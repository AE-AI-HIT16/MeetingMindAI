"""Post-session transcript persistence and Job event publication."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from sqlmodel import Session, select

from meetasr.api.schemas_phase2 import (
    DocDeltaEvent,
    DoneEvent,
    ErrorEvent,
    SpeakerAssignment,
    SpeakerUpdateEvent,
    StatusEvent,
    TranscriptDeltaEvent,
    TranscriptSegmentPayload,
)
from meetasr.db.connection import engine
from meetasr.db.models_phase2 import (
    Document,
    Job,
    JobStage,
    JobStatus,
    Source,
    TranscriptSegment,
)
from meetasr.realtime.events import event_bus
from meetasr.schemas import (
    Segment,
    SentenceInfo,
    SpeakerTurn,
    TargetedRetranscriptionResult,
    TargetedSegmentPlan,
    TranscriptResult,
)
from meetasr.services.document_service import DocumentService
from meetasr.services.inference_coordinator import (
    InferenceCoordinator,
    InferenceKind,
)
from meetasr.services.targeted_transcript_persistence import (
    persist_targeted_transcript,
)
from meetasr.streaming.final_transcript_queue import FinalTranscriptJob
from meetasr.utils.targeted_retranscription import (
    resolve_targeted_retranscription,
)

logger = logging.getLogger(__name__)


class FinalTranscriptWorker:
    """Process completed realtime audio and publish replayable Job events."""

    def __init__(
        self,
        queue: Any,
        pipeline: Any,
        coordinator: InferenceCoordinator | None = None,
    ) -> None:
        self.queue = queue
        self.pipeline = pipeline
        self.coordinator = coordinator

    async def run(self) -> None:
        """Consume final transcript jobs until the worker is cancelled."""
        while True:
            final_job = await self.queue.get()
            try:
                await self._process(final_job)
            finally:
                self.queue.task_done()

    async def _process(self, final_job: FinalTranscriptJob) -> None:
        """Materialize one queued recording only while it is being processed."""
        try:
            await self._process_audio(final_job, final_job.load_audio())
        finally:
            final_job.cleanup()

    async def _process_audio(
        self,
        final_job: FinalTranscriptJob,
        audio: Any,
    ) -> None:
        """Persist one transcript and publish its full Job event sequence."""
        job_id = final_job.job_id
        if not self._update_job(
            job_id,
            status=JobStatus.PROCESSING,
            stage=JobStage.TRANSCRIBING,
            progress=0.05,
            error=None,
        ):
            logger.warning("Job %s not found", job_id)
            await self._publish(
                job_id,
                ErrorEvent(
                    code="job_not_found",
                    message=f"Job '{job_id}' không tồn tại.",
                ),
            )
            return

        await self._publish(
            job_id,
            StatusEvent(stage=JobStage.TRANSCRIBING, progress=0.05),
        )

        try:
            duration_ms = final_job.duration_ms
            if final_job.coverage.complete:
                source_id, persisted = self._load_persisted(job_id)
                sentences, plans = await self._run_inference(
                    InferenceKind.FINALIZE,
                    self._prepare_targeted_retranscription,
                    audio,
                    final_job.coverage,
                    persisted,
                )
                finalized = await self._resolve_targeted_retranscription(
                    audio,
                    sentences,
                    plans,
                )
                persisted_result = persist_targeted_transcript(
                    engine,
                    job_id,
                    persisted,
                    finalized,
                    duration_ms,
                )
                source_id = persisted_result.source_id
                segments = persisted_result.segments
                if persisted_result.speaker_updates:
                    await self._publish(
                        job_id,
                        SpeakerUpdateEvent(
                            updates=persisted_result.speaker_updates,
                        ),
                    )
                logger.info(
                    "Finalizer targeted realtime transcript job=%s "
                    "segments=%d retried=%d replaced=%d fallback=%d "
                    "asr_audio_ms=%d",
                    job_id,
                    len(segments),
                    finalized.stats.targeted_segments,
                    finalized.stats.replaced_segments,
                    finalized.stats.fallback_segments,
                    finalized.stats.asr_audio_ms,
                )
            else:
                logger.warning(
                    "Realtime coverage incomplete job=%s missing=%d "
                    "asr_failures=%d audio_drops=%d flush=%s; running fallback",
                    job_id,
                    len(final_job.coverage.missing_ranges),
                    final_job.coverage.asr_failure_count,
                    final_job.coverage.audio_drop_count,
                    final_job.coverage.flush_completed,
                )
                transcript = await self._run_inference(
                    InferenceKind.FALLBACK,
                    self.pipeline.transcribe,
                    audio,
                )
                duration_ms = max(duration_ms, int(transcript.duration * 1000))
                source_id, segments, new_segments, speaker_updates = (
                    self._merge_fallback_transcript(
                        job_id,
                        transcript,
                        duration_ms,
                    )
                )
                for segment in new_segments:
                    await self._publish(
                        job_id,
                        TranscriptDeltaEvent(segment=segment),
                    )
                if speaker_updates:
                    await self._publish(
                        job_id,
                        SpeakerUpdateEvent(updates=speaker_updates),
                    )

            self._update_job(
                job_id,
                status=JobStatus.PROCESSING,
                stage=JobStage.GENERATING_DOC,
                progress=0.85,
                error=None,
            )
            await self._publish(
                job_id,
                StatusEvent(stage=JobStage.GENERATING_DOC, progress=0.85),
            )

            live_document, markdown = self._create_live_document(source_id)
            self._update_job(
                job_id,
                status=JobStatus.DONE,
                stage=JobStage.GENERATING_DOC,
                progress=1.0,
                error=None,
            )

            await self._publish(
                job_id,
                DocDeltaEvent(section_id="live", markdown=markdown),
            )
            await self._publish(
                job_id,
                DoneEvent(
                    duration_ms=duration_ms,
                    num_segments=len(segments),
                    live_document_id=live_document.id,
                ),
            )
            logger.info(
                "Final transcript completed job=%s segments=%d",
                job_id,
                len(segments),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = str(exc) or "Xử lý transcript cuối thất bại."
            logger.exception("Final transcript failed job=%s", job_id)
            self._update_job(
                job_id,
                status=JobStatus.FAILED,
                error=message,
            )
            await self._publish(
                job_id,
                ErrorEvent(
                    code="final_transcript_failed",
                    message=message,
                ),
            )

    def _load_persisted(
        self,
        job_id: str,
    ) -> tuple[str, list[TranscriptSegmentPayload]]:
        """Load stable realtime transcript records in timeline order."""
        with Session(engine) as db:
            job = db.get(Job, job_id)
            if job is None:
                raise RuntimeError(f"Job '{job_id}' không tồn tại.")
            records = db.exec(
                select(TranscriptSegment)
                .where(TranscriptSegment.job_id == job_id)
                .order_by(TranscriptSegment.start_ms, TranscriptSegment.id)
            ).all()
            return job.source_id, [
                TranscriptSegmentPayload.from_db(record) for record in records
            ]

    def _load_segments(self, job_id: str) -> list[TranscriptSegmentPayload]:
        return self._load_persisted(job_id)[1]

    def _prepare_targeted_retranscription(
        self,
        audio: Any,
        coverage: Any,
        persisted: list[TranscriptSegmentPayload],
    ) -> tuple[list[SentenceInfo], list[TargetedSegmentPlan]]:
        sentences = [
            SentenceInfo(
                text=item.text,
                start=item.start_ms / 1000,
                end=item.end_ms / 1000,
                speaker=item.speaker,
            )
            for item in persisted
        ]
        vad_segments = [
            Segment(item.start_ms, item.end_ms)
            for item in coverage.speech_ranges
        ]
        plans = self.pipeline.prepare_realtime_targeted_retranscription(
            audio,
            sentences,
            vad_segments,
        )
        if len(plans) != len(persisted):
            raise RuntimeError("Targeted plan must cover every persisted segment.")
        return sentences, plans

    async def _resolve_targeted_retranscription(
        self,
        audio: Any,
        sentences: list[SentenceInfo],
        plans: list[TargetedSegmentPlan],
    ) -> TargetedRetranscriptionResult:
        turn_results: dict[int, list[SentenceInfo]] = {}
        for plan in plans:
            if plan.action != "retranscribe":
                continue
            failed = False
            for turn in plan.turns:
                if failed:
                    turn_results[id(turn)] = []
                    continue
                try:
                    turn_results[id(turn)] = await self._run_inference(
                        InferenceKind.TARGETED,
                        self._transcribe_targeted_turn,
                        audio,
                        turn,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    failed = True
                    turn_results[id(turn)] = []
                    logger.exception(
                        "Targeted ASR turn failed segment=%d range=%d-%d",
                        plan.index,
                        turn.start_ms,
                        turn.end_ms,
                    )

        return resolve_targeted_retranscription(
            sentences,
            plans,
            lambda turn: turn_results.get(id(turn), []),
        )

    def _transcribe_targeted_turn(
        self,
        audio: Any,
        turn: SpeakerTurn,
    ) -> list[SentenceInfo]:
        """Decode one diarized turn using the full pipeline ASR model."""
        return self.pipeline.transcribe_vad_segment(
            audio,
            turn.to_segment(),
            language=getattr(
                self.pipeline,
                "transcription_language",
                "auto",
            ),
        )

    async def _run_inference(
        self,
        kind: InferenceKind,
        function: Any,
        *args: Any,
    ) -> Any:
        if self.coordinator is not None:
            return await self.coordinator.submit(kind, function, *args)
        return await asyncio.to_thread(function, *args)

    def _merge_fallback_transcript(
        self,
        job_id: str,
        transcript: TranscriptResult,
        duration_ms: int,
    ) -> tuple[
        str,
        list[TranscriptSegmentPayload],
        list[TranscriptSegmentPayload],
        list[SpeakerAssignment],
    ]:
        """Keep stable live records and add only uncovered fallback output."""
        with Session(engine) as db:
            job = db.get(Job, job_id)
            if job is None:
                raise RuntimeError(f"Job '{job_id}' không tồn tại.")
            source = db.get(Source, job.source_id)
            if source is None:
                raise RuntimeError(f"Source của Job '{job_id}' không tồn tại.")

            records = db.exec(
                select(TranscriptSegment)
                .where(TranscriptSegment.job_id == job_id)
                .order_by(TranscriptSegment.start_ms, TranscriptSegment.id)
            ).all()
            new_records = []
            speaker_updates = []
            for sentence in transcript.sentence_info:
                text = sentence.text.strip()
                if not text:
                    continue
                start_ms = max(0, int(sentence.start * 1000))
                end_ms = max(start_ms, int(sentence.end * 1000))
                matched = _best_covered_record(records, start_ms, end_ms)
                if matched is not None:
                    if matched.speaker != sentence.speaker:
                        matched.speaker = sentence.speaker
                        db.add(matched)
                        speaker_updates.append(
                            SpeakerAssignment(
                                segment_id=matched.id,
                                speaker=matched.speaker,
                            )
                        )
                    continue
                record = TranscriptSegment(
                    job_id=job_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker=sentence.speaker,
                    text=text,
                )
                db.add(record)
                records.append(record)
                new_records.append(record)

            source.duration = max(source.duration or 0.0, duration_ms / 1000)
            db.add(source)
            db.commit()
            new_payloads = []
            all_payloads = []
            for record in records:
                db.refresh(record)
                payload = TranscriptSegmentPayload.from_db(record)
                all_payloads.append(payload)
                if record in new_records:
                    new_payloads.append(payload)
            all_payloads.sort(key=lambda item: (item.start_ms, item.id or 0))
            return source.id, all_payloads, new_payloads, speaker_updates

    def _create_live_document(self, source_id: str) -> tuple[Document, str]:
        """Create the live transcript document after segments are committed."""
        with Session(engine) as db:
            service = DocumentService(db)
            transcript = service.build_transcript(source_id)
            markdown = DocumentService.full_text_markdown(transcript)
            document = service.save_live_document(source_id, markdown)
            db.expunge(document)
            return document, markdown

    def _update_job(
        self,
        job_id: str,
        *,
        status: str,
        stage: str | None = None,
        progress: float | None = None,
        error: str | None = None,
    ) -> bool:
        """Persist one Job state transition and report whether the Job exists."""
        with Session(engine) as db:
            job = db.get(Job, job_id)
            if job is None:
                return False
            job.status = status
            if stage is not None:
                job.stage = stage
            if progress is not None:
                job.progress = progress
            job.error = error
            job.updated_at = datetime.utcnow()
            db.add(job)
            db.commit()
            return True

    async def _publish(self, job_id: str, event: Any) -> None:
        """Publish an event without invalidating already committed Job state."""
        try:
            await event_bus.publish(job_id, event)
        except Exception:
            logger.exception(
                "Failed to publish Job event job=%s type=%s",
                job_id,
                getattr(event, "type", type(event).__name__),
            )


def _best_covered_record(
    records: list[TranscriptSegment],
    start_ms: int,
    end_ms: int,
    *,
    coverage_threshold: float = 0.80,
) -> TranscriptSegment | None:
    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        return None
    candidates = []
    for record in records:
        overlap_ms = min(end_ms, record.end_ms) - max(start_ms, record.start_ms)
        if overlap_ms > 0:
            candidates.append((overlap_ms / duration_ms, record))
    if not candidates:
        return None
    ratio, record = max(candidates, key=lambda item: item[0])
    return record if ratio >= coverage_threshold else None
