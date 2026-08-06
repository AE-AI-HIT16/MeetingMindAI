"""Background, idempotent document finalization for Phase 2."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from meetasr.api.schemas_phase2 import (
    DocumentGenerationDoneEvent,
    DocumentGenerationErrorEvent,
    DocumentGenerationStatusEvent,
)
from meetasr.db.connection import engine
from meetasr.db.models_phase2 import (
    Document,
    DocumentGenerationJob,
    DocumentGenerationStage,
    DocumentGenerationStatus,
    DocumentMode,
)
from meetasr.realtime.events import event_bus
from meetasr.services.document_service import (
    DocumentGenerationError,
    DocumentService,
    FinalDocumentMode,
    InvalidDocumentStateError,
    PlannerUnavailableError,
)

logger = logging.getLogger(__name__)


class DocumentGenerationQueue:
    """One-consumer queue with per-source/mode submission locks."""

    def __init__(self, *, database_engine=engine, maxsize: int = 100) -> None:
        self._engine = database_engine
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self._task: asyncio.Task[None] | None = None
        self._planner: Any | None = None
        self._submission_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._submission_locks_guard = asyncio.Lock()
        self._queued_ids: set[str] = set()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, planner: Any | None) -> None:
        """Start the consumer and resume unfinished persisted jobs."""
        self._planner = planner

    async def stop(self) -> None:
        pass

    async def join(self) -> None:
        """Wait until all queued generation jobs finish (mainly for tests)."""
        await self._queue.join()

    async def submit(
        self,
        db: Session,
        live_document_id: str,
        mode: FinalDocumentMode,
        *,
        planner: Any | None,
    ) -> DocumentGenerationJob:
        """Atomically reuse or create one generation job for source/mode."""
        service = DocumentService(db, planner=planner)
        live_document = service.get_document(live_document_id)
        if live_document.mode != DocumentMode.LIVE:
            raise InvalidDocumentStateError(
                "Finalize phải bắt đầu từ Document có mode='live'."
            )
        if mode not in (DocumentMode.SUMMARY, DocumentMode.FULL_TEXT):
            raise InvalidDocumentStateError(
                f"Document mode '{mode}' không hỗ trợ finalize."
            )
        if mode == DocumentMode.SUMMARY and planner is None:
            raise PlannerUnavailableError(
                "DocumentPlanner chưa được cấu hình."
            )

        # Validate that generation is possible before creating queued rows.
        service.build_transcript(live_document.source_id)
        key = (live_document.source_id, mode)
        lock = await self._submission_lock(key)
        should_enqueue = False

        async with lock:
            # Re-read inside the lock. Another request may have completed while
            # this request was waiting.
            document = db.exec(
                select(Document)
                .where(Document.source_id == live_document.source_id)
                .where(Document.mode == mode)
            ).first()
            generation = db.exec(
                select(DocumentGenerationJob)
                .where(
                    DocumentGenerationJob.source_id
                    == live_document.source_id
                )
                .where(DocumentGenerationJob.mode == mode)
            ).first()

            if document is not None and document.markdown.strip():
                if generation is None:
                    generation = DocumentGenerationJob(
                        document_id=document.id,
                        source_id=live_document.source_id,
                        mode=mode,
                        status=DocumentGenerationStatus.DONE,
                        stage=DocumentGenerationStage.DONE,
                        progress=1.0,
                    )
                else:
                    generation.document_id = document.id
                    generation.status = DocumentGenerationStatus.DONE
                    generation.stage = DocumentGenerationStage.DONE
                    generation.progress = 1.0
                    generation.error = None
                    generation.updated_at = datetime.utcnow()
                db.add(generation)
                db.commit()
                db.refresh(generation)
                return generation

            if document is None:
                document = Document(
                    source_id=live_document.source_id,
                    mode=mode,
                )
                db.add(document)
                try:
                    db.flush()
                except IntegrityError:
                    # Database constraint is the final guard when requests came
                    # from different Python processes with different locks.
                    db.rollback()
                    document = db.exec(
                        select(Document)
                        .where(
                            Document.source_id
                            == live_document.source_id
                        )
                        .where(Document.mode == mode)
                    ).one()

            if generation is None:
                generation = DocumentGenerationJob(
                    document_id=document.id,
                    source_id=live_document.source_id,
                    mode=mode,
                )
                should_enqueue = True
            elif generation.status == DocumentGenerationStatus.FAILED:
                generation.document_id = document.id
                generation.status = DocumentGenerationStatus.QUEUED
                generation.stage = DocumentGenerationStage.QUEUED
                generation.progress = 0.0
                generation.error = None
                generation.updated_at = datetime.utcnow()
                should_enqueue = True

            db.add(generation)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                generation = db.exec(
                    select(DocumentGenerationJob)
                    .where(
                        DocumentGenerationJob.source_id
                        == live_document.source_id
                    )
                    .where(DocumentGenerationJob.mode == mode)
                ).one()
                should_enqueue = False
            db.refresh(generation)

            if should_enqueue:
                await self.enqueue(generation.id)
            return generation

    async def enqueue(self, generation_id: str) -> bool:
        """No-op for serverless deployment: generation triggered on websocket connection."""
        return True

    async def _submission_lock(
        self,
        key: tuple[str, str],
    ) -> asyncio.Lock:
        async with self._submission_locks_guard:
            return self._submission_locks.setdefault(key, asyncio.Lock())

    async def _run(self) -> None:
        pass

    async def _process(self, generation_id: str) -> None:
        try:
            generation = self._mark_processing(generation_id)
            if generation is None:
                return
            await self._publish_status(generation, progress=0.1)

            loop = asyncio.get_running_loop()
            progress_updates = []

            def report_progress(planner_progress: float) -> None:
                external_progress = 0.1 + 0.75 * planner_progress
                progress_updates.append(
                    asyncio.run_coroutine_threadsafe(
                        self._record_generation_progress(
                            generation_id,
                            external_progress,
                        ),
                        loop,
                    )
                )

            with Session(self._engine) as db:
                live_document = db.exec(
                    select(Document)
                    .where(Document.source_id == generation.source_id)
                    .where(Document.mode == DocumentMode.LIVE)
                ).first()
                if live_document is None:
                    raise InvalidDocumentStateError(
                        "Source không còn tài liệu live để finalize."
                    )
                service = DocumentService(db, planner=self._planner)
                document = await service.finalize_document(
                    live_document.id,
                    generation.mode,
                    progress_callback=report_progress,
                )

            if progress_updates:
                await asyncio.gather(
                    *(
                        asyncio.wrap_future(update)
                        for update in progress_updates
                    )
                )
            generation = self._update_stage(
                generation_id,
                DocumentGenerationStage.SAVING,
                0.9,
            )
            if generation is None:
                return
            await self._publish_status(generation, progress=0.9)
            generation = self._finish(generation_id, document.id)
            if generation is None:
                return
            await event_bus.publish(
                generation.id,
                DocumentGenerationDoneEvent(
                    generation_job_id=generation.id,
                    document_id=generation.document_id,
                    mode=generation.mode,
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Document generation %s failed",
                generation_id,
            )
            generation = self._fail(generation_id, exc)
            if generation is not None:
                await event_bus.publish(
                    generation.id,
                    DocumentGenerationErrorEvent(
                        generation_job_id=generation.id,
                        document_id=generation.document_id,
                        code="document_generation_failed",
                        message=generation.error
                        or "Không thể tạo tài liệu.",
                        retry_after=(
                            60
                            if isinstance(exc, DocumentGenerationError)
                            else None
                        ),
                    ),
                )

    def _mark_processing(
        self,
        generation_id: str,
    ) -> DocumentGenerationJob | None:
        with Session(self._engine) as db:
            generation = db.get(DocumentGenerationJob, generation_id)
            if (
                generation is None
                or generation.status != DocumentGenerationStatus.QUEUED
            ):
                return None
            generation.status = DocumentGenerationStatus.PROCESSING
            generation.stage = DocumentGenerationStage.GENERATING
            generation.progress = 0.1
            generation.error = None
            generation.updated_at = datetime.utcnow()
            db.add(generation)
            db.commit()
            db.refresh(generation)
            db.expunge(generation)
            return generation

    def _update_stage(
        self,
        generation_id: str,
        stage: str,
        progress: float,
    ) -> DocumentGenerationJob | None:
        with Session(self._engine) as db:
            generation = db.get(DocumentGenerationJob, generation_id)
            if generation is None:
                return None
            if (
                stage
                in {
                    DocumentGenerationStage.GENERATING,
                    DocumentGenerationStage.SAVING,
                }
                and generation.status
                != DocumentGenerationStatus.PROCESSING
            ):
                return None
            generation.stage = stage
            generation.progress = progress
            generation.updated_at = datetime.utcnow()
            db.add(generation)
            db.commit()
            db.refresh(generation)
            db.expunge(generation)
            return generation

    def _finish(
        self,
        generation_id: str,
        document_id: str,
    ) -> DocumentGenerationJob | None:
        with Session(self._engine) as db:
            generation = db.get(DocumentGenerationJob, generation_id)
            if generation is None:
                return None
            generation.document_id = document_id
            generation.status = DocumentGenerationStatus.DONE
            generation.stage = DocumentGenerationStage.DONE
            generation.progress = 1.0
            generation.error = None
            generation.updated_at = datetime.utcnow()
            db.add(generation)
            db.commit()
            db.refresh(generation)
            db.expunge(generation)
            return generation

    def _fail(
        self,
        generation_id: str,
        exc: Exception,
    ) -> DocumentGenerationJob | None:
        if isinstance(
            exc,
            (
                DocumentGenerationError,
                InvalidDocumentStateError,
                PlannerUnavailableError,
            ),
        ):
            message = str(exc)
        else:
            message = "Không thể tạo tài liệu do lỗi nội bộ."

        with Session(self._engine) as db:
            generation = db.get(DocumentGenerationJob, generation_id)
            if generation is None:
                return None
            generation.status = DocumentGenerationStatus.FAILED
            generation.stage = DocumentGenerationStage.FAILED
            generation.error = message
            generation.updated_at = datetime.utcnow()
            db.add(generation)
            db.commit()
            db.refresh(generation)
            db.expunge(generation)
            return generation

    async def _publish_status(
        self,
        generation: DocumentGenerationJob,
        *,
        progress: float,
    ) -> None:
        await event_bus.publish(
            generation.id,
            DocumentGenerationStatusEvent(
                generation_job_id=generation.id,
                document_id=generation.document_id,
                stage=generation.stage,
                progress=progress,
            ),
        )

    async def _record_generation_progress(
        self,
        generation_id: str,
        progress: float,
    ) -> None:
        generation = self._update_stage(
            generation_id,
            DocumentGenerationStage.GENERATING,
            min(max(progress, 0.1), 0.85),
        )
        if generation is not None:
            await self._publish_status(
                generation,
                progress=generation.progress,
            )


document_generation_queue = DocumentGenerationQueue()


async def run_document_generation(generation_id: str, planner: Any) -> None:
    """Run document generation on-demand, scoped to a connection/request."""
    document_generation_queue._planner = planner
    try:
        await document_generation_queue._process(generation_id)
    except asyncio.CancelledError:
        logger.info(f"Document generation {generation_id} cancelled.")
        document_generation_queue._fail(generation_id, Exception("Cancelled: Client disconnected."))
        raise
