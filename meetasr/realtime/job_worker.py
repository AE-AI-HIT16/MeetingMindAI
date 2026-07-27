"""Sequential background worker for uploaded Phase 2 media."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from datetime import datetime
from pathlib import Path

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
from meetasr.schemas import SentenceInfo
from meetasr.services.asr_service import ASRService
from meetasr.services.document_service import DocumentService
from meetasr.storage.backend import StorageBackend


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
        return self._task is not None and not self._task.done()

    async def start(
        self,
        asr_service: ASRService | None,
        storage: StorageBackend,
    ) -> None:
        if self.running:
            return
        self._storage = storage
        self._asr_service = asr_service
        self._task = asyncio.create_task(self._run(), name="phase2-job-worker")

        with Session(engine) as db:
            pending = db.exec(
                select(Job).where(
                    Job.status.in_([JobStatus.QUEUED, JobStatus.PROCESSING])
                )
            ).all()
        for job in pending:
            await self._queue.put(job.id)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def enqueue(self, job_id: str) -> bool:
        """Queue a Job if the application worker is running."""
        if not self.running:
            logger.warning(
                "Job %s remains queued because the worker is not running.",
                job_id,
            )
            return False
        await self._queue.put(job_id)
        return True

    async def _run(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._process(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unhandled failure while processing Job %s", job_id)
            finally:
                self._queue.task_done()

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

            prepared = await self._asr_service.prepare_incremental(audio_source)
            await self._publish_status(
                job_id,
                JobStage.TRANSCRIBING,
                0.15,
            )
            provisional_sentences: list[SentenceInfo] = []
            persisted: list[TranscriptSegmentPayload] = []
            chunk_count = max(len(prepared.vad_segments), 1)

            for index, vad_segment in enumerate(
                prepared.vad_segments,
                start=1,
            ):
                chunk_sentences = await self._asr_service.transcribe_segment(
                    prepared,
                    vad_segment,
                    key=f"{source.filename}:chunk-{index}",
                )
                provisional_sentences.extend(chunk_sentences)
                chunk_payloads = [
                    _sentence_to_payload(sentence, speaker=None)
                    for sentence in chunk_sentences
                ]
                persisted_chunk = self._persist_segments(
                    job_id,
                    chunk_payloads,
                )
                persisted.extend(persisted_chunk)

                # Persist first. A reconnect can now replay every event that the
                # frontend is about to receive.
                for segment in persisted_chunk:
                    await event_bus.publish(
                        job_id,
                        TranscriptDeltaEvent(segment=segment),
                    )
                await self._publish_status(
                    job_id,
                    JobStage.TRANSCRIBING,
                    0.15 + 0.65 * index / chunk_count,
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

            duration_ms = prepared.duration_ms
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
            job.updated_at = datetime.utcnow()
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
            job.updated_at = datetime.utcnow()
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
            job.updated_at = datetime.utcnow()
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
            job.updated_at = datetime.utcnow()
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
) -> TranscriptSegmentPayload:
    start_ms = max(0, int(sentence.start * 1000))
    return TranscriptSegmentPayload(
        start_ms=start_ms,
        end_ms=max(start_ms, int(sentence.end * 1000)),
        speaker=speaker,
        text=sentence.text,
    )


job_queue = JobQueue()
