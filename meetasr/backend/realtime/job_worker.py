"""Sequential background worker for uploaded Phase 2 media."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Session, select

from meetasr.backend.api.schemas_phase2 import (
    DocDeltaEvent,
    DoneEvent,
    ErrorEvent,
    SpeakerAssignment,
    SpeakerTurn,
    SpeakerUpdateEvent,
    StatusEvent,
    TranscriptDeltaEvent,
    TranscriptSegmentPayload,
)
from meetasr.backend.db.connection import engine
from meetasr.backend.db.models_phase2 import (
    Document,
    Job,
    JobStage,
    JobStatus,
    Source,
    TranscriptSegment,
)
from meetasr.backend.realtime.events import event_bus
from meetasr.backend.schemas import SentenceInfo
from meetasr.backend.services.asr_service import ASRService
from meetasr.backend.services.document_service import DocumentService
from meetasr.backend.storage.backend import StorageBackend
from meetasr.backend.utils.overlap_echo import is_overlap_echo, overlap_ratio

logger = logging.getLogger(__name__)


class JobQueue:
    """One-consumer queue that serializes access to the shared ML pipeline."""

    def __init__(self, maxsize: int = 100) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self._task: asyncio.Task[None] | None = None
        self._storage: StorageBackend | None = None
        self._asr_service: ASRService | None = None

    @property
    def running(self) -> bool:
        return True

    async def start(
        self,
        asr_service: ASRService | None,
        storage: StorageBackend,
    ) -> None:
        self._storage = storage
        self._asr_service = asr_service

    async def stop(self) -> None:
        pass

    async def enqueue(self, job_id: str) -> bool:
        """No-op for serverless deployment: job triggered on websocket connection."""
        return True

    async def _run(self) -> None:
        pass

    async def _process(self, job_id: str) -> None:
        temporary_path: Path | None = None
        try:
            source = self._prepare_job(job_id)
            if source is None:
                return

            await self._publish_status(
                job_id,
                JobStage.EXTRACTING_AUDIO,
                0.05,
            )
            audio_source, temporary_path = await self._materialize_source(source)

            if self._asr_service is None:
                raise RuntimeError("ASR pipeline chưa được cấu hình.")

            await self._publish_status(
                job_id,
                JobStage.TRANSCRIBING,
                0.10,
            )

            # Diarization-first pipeline (works for both RunPod serverless and local):
            #  1. VAD + speaker diarization on the WHOLE file -> consistent speakers
            #  2. transcribe each speaker turn, streaming segments as they complete
            #  3. finalize (punctuation / speaker projection)
            asr_context, num_speakers = self._job_hints(job_id)
            prepared = await self._asr_service.prepare_incremental(
                audio_source,
                num_speakers=num_speakers,
            )
            provisional_sentences: list[SentenceInfo] = []
            persisted: list[TranscriptSegmentPayload] = []
            speaker_first = prepared.speaker_turns is not None
            work_items = (
                [
                    (turn.to_segment(), turn.speaker, turn)
                    for turn in prepared.speaker_turns
                ]
                if speaker_first
                else [
                    (vad_segment, None, None)
                    for vad_segment in prepared.vad_segments
                ]
            )
            chunk_count = max(len(work_items), 1)
            logger.info(
                "Job %s: diarization-first, speaker_first=%s, %d turn(s)",
                job_id, speaker_first, chunk_count,
            )

            # Transcribe turns in batches: each batch is ONE RunPod job (one
            # round-trip on one warm worker — per-turn parallel calls spent most of
            # the time on network/queue overhead and cold-started extra workers).
            # Persist/stream each batch in timeline order so the UI fills
            # top-to-bottom. 24 turns per job (fewer round-trips; Qwen batches
            # 16 on the GPU). Worst case 24 x 15 s FLAC ≈ 10 MB base64, under
            # the 20 MiB /runsync limit.
            BATCH_SIZE = 24
            completed = 0
            transcribed_turns: list[tuple[SpeakerTurn, str]] = []
            for batch_start in range(0, len(work_items), BATCH_SIZE):
                batch = work_items[batch_start:batch_start + BATCH_SIZE]
                batch_results = await self._asr_service.transcribe_segments(
                    prepared,
                    [seg for seg, _spk, _turn in batch],
                    context=asr_context,
                    turns=[turn for _seg, _spk, turn in batch],
                )
                batch_texts = [
                    (turn, " ".join(s.text for s in sentences))
                    for (_seg, _spk, turn), sentences in zip(batch, batch_results)
                    if turn is not None
                ]
                transcribed_turns.extend(batch_texts)

                for (transcription_segment, speaker, turn), chunk_sentences in zip(batch, batch_results):
                    if turn is not None and _is_echo(turn, chunk_sentences, transcribed_turns):
                        logger.info(
                            "Job %s: dropped overlapped turn %d-%dms repeating another speaker",
                            job_id, turn.start_ms, turn.end_ms,
                        )
                        continue
                    if speaker_first:
                        for sentence in chunk_sentences:
                            sentence.speaker = speaker
                    provisional_sentences.extend(chunk_sentences)
                    chunk_payloads = [
                        _sentence_to_payload(
                            sentence,
                            speaker=sentence.speaker if speaker_first else None,
                            overlapped=_has_crosstalk(sentence, turn),
                        )
                        for sentence in chunk_sentences
                    ]
                    persisted_chunk = self._persist_segments(
                        job_id,
                        chunk_payloads,
                    )
                    persisted.extend(persisted_chunk)
                    for segment in persisted_chunk:
                        await event_bus.publish(
                            job_id,
                            TranscriptDeltaEvent(segment=segment),
                        )

                completed += len(batch)
                await self._publish_status(
                    job_id,
                    JobStage.TRANSCRIBING,
                    0.15 + 0.65 * completed / chunk_count,
                )

            finalized_sentences = await self._asr_service.finalize_incremental(
                prepared,
                provisional_sentences,
            )
            finalized = self._apply_finalized_segments(
                persisted,
                finalized_sentences,
            )

            speaker_updates = [
                SpeakerAssignment(
                    segment_id=final.id,
                    speaker=final.speaker,
                )
                for provisional, final in zip(persisted, finalized)
                if final.id is not None
                and provisional.speaker != final.speaker
            ]
            if speaker_updates:
                await event_bus.publish(
                    job_id,
                    SpeakerUpdateEvent(updates=speaker_updates),
                )

            # Punctuation may have changed the text during finalization. Re-send
            # the same stable IDs so clients replace, rather than duplicate, them.
            for provisional, final in zip(persisted, finalized):
                if (
                    provisional.text != final.text
                    or provisional.start_ms != final.start_ms
                    or provisional.end_ms != final.end_ms
                ):
                    await event_bus.publish(
                        job_id,
                        TranscriptDeltaEvent(segment=final),
                    )

            await self._publish_status(
                job_id,
                JobStage.GENERATING_DOC,
                0.85,
            )
            markdown = _transcript_markdown(source.filename, finalized)
            live_document = self._save_live_document(source.id, markdown)
            await event_bus.publish(
                job_id,
                DocDeltaEvent(section_id="live", markdown=markdown),
            )

            duration_ms = (
                prepared.duration_ms
                if prepared is not None
                else max((seg.end_ms for seg in finalized), default=0)
            )
            self._finish_job(job_id, source.id, duration_ms)
            await event_bus.publish(
                job_id,
                DoneEvent(
                    duration_ms=duration_ms,
                    num_segments=len(finalized),
                    live_document_id=live_document.id,
                ),
            )
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self._fail_job(job_id, str(exc))
            await event_bus.publish(
                job_id,
                ErrorEvent(
                    code="processing_failed",
                    message=str(exc) or "Job xử lý thất bại.",
                ),
            )
        finally:
            if temporary_path is not None:
                await asyncio.to_thread(temporary_path.unlink, missing_ok=True)

    def _job_hints(self, job_id: str) -> tuple[str, int | None]:
        """User hints stored on the Job: Qwen3 keywords and speaker count."""
        with Session(engine) as db:
            job = db.get(Job, job_id)
            if job is None:
                return "", None
            return job.asr_context or "", job.num_speakers

    def _prepare_job(self, job_id: str) -> Source | None:
        with Session(engine) as db:
            job = db.get(Job, job_id)
            if job is None or job.status == JobStatus.DONE:
                return None
            source = db.get(Source, job.source_id)
            if source is None:
                raise RuntimeError(f"Source của Job '{job_id}' không tồn tại.")

            old_segments = db.exec(
                select(TranscriptSegment).where(TranscriptSegment.job_id == job_id)
            ).all()
            for segment in old_segments:
                db.delete(segment)

            job.status = JobStatus.PROCESSING
            job.stage = JobStage.EXTRACTING_AUDIO
            job.progress = 0.0
            job.error = None
            job.updated_at = datetime.now(timezone.utc)
            db.add(job)
            db.commit()
            db.refresh(source)
            db.expunge(source)
            return source

    async def _materialize_source(
        self,
        source: Source,
    ) -> tuple[str | bytes, Path | None]:
        if self._storage is None:
            raise RuntimeError("Storage backend chưa được cấu hình.")

        abs_path = getattr(self._storage, "abs_path", None)
        if callable(abs_path):
            return str(abs_path(source.storage_path)), None

        data = await self._storage.load(source.storage_path)
        suffix = Path(source.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            path = Path(tmp.name)
        return str(path), path

    def _persist_segments(
        self,
        job_id: str,
        segments: list[TranscriptSegmentPayload],
    ) -> list[TranscriptSegmentPayload]:
        with Session(engine) as db:
            records = [
                TranscriptSegment(
                    job_id=job_id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    speaker=segment.speaker,
                    text=segment.text,
                    overlapped=segment.overlapped,
                )
                for segment in segments
            ]
            db.add_all(records)
            db.commit()
            payloads = []
            for record in records:
                db.refresh(record)
                payloads.append(TranscriptSegmentPayload.from_db(record))
            return payloads

    def _save_live_document(self, source_id: str, markdown: str) -> Document:
        with Session(engine) as db:
            document = DocumentService(db).save_live_document(
                source_id,
                markdown,
            )
            db.expunge(document)
            return document

    def _apply_finalized_segments(
        self,
        persisted: list[TranscriptSegmentPayload],
        finalized_sentences: list[SentenceInfo],
    ) -> list[TranscriptSegmentPayload]:
        if len(persisted) != len(finalized_sentences):
            raise RuntimeError(
                "Diarization phải giữ nguyên số TranscriptSegment đã lưu."
            )

        with Session(engine) as db:
            records = []
            for payload, sentence in zip(persisted, finalized_sentences):
                if payload.id is None:
                    raise RuntimeError("TranscriptSegment đã commit nhưng chưa có id.")
                record = db.get(TranscriptSegment, payload.id)
                if record is None:
                    raise RuntimeError(
                        f"TranscriptSegment '{payload.id}' không tồn tại."
                    )
                record.start_ms = max(0, int(sentence.start * 1000))
                record.end_ms = max(record.start_ms, int(sentence.end * 1000))
                record.speaker = sentence.speaker
                record.text = sentence.text
                db.add(record)
                records.append(record)

            db.commit()
            finalized = []
            for record in records:
                db.refresh(record)
                finalized.append(TranscriptSegmentPayload.from_db(record))
            return finalized

    async def _publish_status(
        self,
        job_id: str,
        stage: str,
        progress: float,
    ) -> None:
        self._update_job(job_id, stage=stage, progress=progress)
        await event_bus.publish(
            job_id,
            StatusEvent(stage=stage, progress=progress),
        )

    def _update_job(self, job_id: str, *, stage: str, progress: float) -> None:
        with Session(engine) as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            job.status = JobStatus.PROCESSING
            job.stage = stage
            job.progress = progress
            job.updated_at = datetime.now(timezone.utc)
            db.add(job)
            db.commit()

    def _finish_job(self, job_id: str, source_id: str, duration_ms: int) -> None:
        with Session(engine) as db:
            job = db.get(Job, job_id)
            source = db.get(Source, source_id)
            if job is None or source is None:
                return
            source.duration = duration_ms / 1000
            job.status = JobStatus.DONE
            job.stage = JobStage.GENERATING_DOC
            job.progress = 1.0
            job.updated_at = datetime.now(timezone.utc)
            db.add(source)
            db.add(job)
            db.commit()

    def _fail_job(self, job_id: str, message: str) -> None:
        with Session(engine) as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED
            job.error = message or "Job xử lý thất bại."
            job.updated_at = datetime.now(timezone.utc)
            db.add(job)
            db.commit()


def _transcript_markdown(
    filename: str,
    segments: list[TranscriptSegmentPayload],
) -> str:
    lines = [f"# Bản ghi: {filename}", ""]
    for segment in segments:
        minutes, seconds = divmod(segment.start_ms // 1000, 60)
        speaker = (
            f"Người nói {segment.speaker + 1}"
            if segment.speaker is not None
            else "Chưa xác định"
        )
        lines.append(
            f"**[{minutes:02d}:{seconds:02d}] {speaker}:** {segment.text}"
        )
        lines.append("")
    return "\n".join(lines).strip()


def _sentence_to_payload(
    sentence: SentenceInfo,
    *,
    speaker: int | None,
    overlapped: bool = False,
) -> TranscriptSegmentPayload:
    start_ms = max(0, int(sentence.start * 1000))
    return TranscriptSegmentPayload(
        start_ms=start_ms,
        end_ms=max(start_ms, int(sentence.end * 1000)),
        speaker=speaker,
        text=sentence.text,
        overlapped=overlapped,
    )


CROSSTALK_MIN_MS = 300


def _has_crosstalk(sentence: SentenceInfo, turn: SpeakerTurn | None) -> bool:
    """True if someone else talks over this sentence for >= 300 ms."""
    if turn is None or not turn.overlaps:
        return False
    start_ms, end_ms = int(sentence.start * 1000), int(sentence.end * 1000)
    shared = sum(
        max(0, min(end_ms, b) - max(start_ms, a)) for a, b in turn.overlaps
    )
    return shared >= CROSSTALK_MIN_MS


job_queue = JobQueue()


async def run_job_processing(job_id: str, asr_service: ASRService, storage: StorageBackend) -> None:
    """Run transcription job on-demand, scoped to a connection/request."""
    job_queue._asr_service = asr_service
    job_queue._storage = storage
    try:
        await job_queue._process(job_id)
    except asyncio.CancelledError:
        logger.info(f"Job processing {job_id} cancelled.")
        raise


def recover_interrupted_jobs(db_engine=None) -> tuple[int, int]:
    """Call once at startup: no job can still be running in this process.

    A restart (e.g. a deploy) killed in-flight jobs, which then stayed
    "processing" forever. Jobs whose audio is stored go back to QUEUED (they
    restart when the user opens the source — not all at once, to avoid GPU
    cost for old files nobody looks at); jobs without audio (realtime session
    cut before its audio was saved) can never run and are marked FAILED.
    Returns (requeued, failed).
    """
    requeued = failed = 0
    with Session(db_engine or engine) as db:
        rows = db.exec(
            select(Job, Source)
            .join(Source, Source.id == Job.source_id)
            .where(Job.status.in_([JobStatus.PROCESSING, JobStatus.QUEUED]))
        ).all()
        for job, source in rows:
            if source.storage_path:
                if job.status == JobStatus.PROCESSING:
                    job.status = JobStatus.QUEUED
                    requeued += 1
                else:
                    continue
            else:
                job.status = JobStatus.FAILED
                job.error = "Phiên ghi âm bị gián đoạn trước khi lưu xong âm thanh."
                failed += 1
            job.updated_at = datetime.now(timezone.utc)
            db.add(job)
        db.commit()
    if requeued or failed:
        logger.info("Recovered interrupted jobs: %d requeued, %d failed (no audio).", requeued, failed)
    return requeued, failed


def _is_echo(
    turn: SpeakerTurn,
    sentences: list[SentenceInfo],
    transcribed: list[tuple[SpeakerTurn, str]],
) -> bool:
    """A mostly-overlapped turn whose words repeat a concurrent speaker."""
    concurrent = [
        text for other, text in transcribed
        if other.speaker != turn.speaker
        and other.start_ms < turn.end_ms
        and other.end_ms > turn.start_ms
    ]
    return is_overlap_echo(
        " ".join(s.text for s in sentences),
        overlap_ratio(turn.start_ms, turn.end_ms, turn.overlaps),
        concurrent,
    )
