"""Durable publication of confirmed realtime transcript segments."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Protocol

from sqlmodel import Session, select

from meetasr.api.schemas_phase2 import (
    TranscriptDeltaEvent,
    TranscriptSegmentPayload,
)
from meetasr.db.connection import engine
from meetasr.db.models_phase2 import (
    Job,
    JobStage,
    JobStatus,
    Source,
    TranscriptSegment,
)
from meetasr.realtime.events import event_bus

logger = logging.getLogger(__name__)


class JsonWebSocket(Protocol):
    """Minimum recording WebSocket contract needed by the service."""

    async def send_json(self, payload: dict[str, Any]) -> None: ...


class RealtimeTranscriptService:
    """Commit confirmed ASR output before publishing its canonical event."""

    async def persist_and_publish(
        self,
        job_id: str,
        segment: TranscriptSegmentPayload,
        websocket: JsonWebSocket,
    ) -> TranscriptSegmentPayload:
        """Persist one idempotent delta and fan it out after commit."""
        persisted = await asyncio.to_thread(
            self._persist_confirmed,
            job_id,
            segment,
        )
        event = TranscriptDeltaEvent(segment=persisted)
        await self._publish_job_event(job_id, event)
        await self._publish_recording_event(websocket, event)
        return persisted

    def _persist_confirmed(
        self,
        job_id: str,
        segment: TranscriptSegmentPayload,
    ) -> TranscriptSegmentPayload:
        text = segment.text.strip()
        if not text:
            raise ValueError("confirmed transcript text must not be blank")

        with Session(engine) as db:
            job = db.get(Job, job_id)
            if job is None:
                raise RuntimeError(f"Job '{job_id}' không tồn tại.")

            existing = db.exec(
                select(TranscriptSegment)
                .where(TranscriptSegment.job_id == job_id)
                .where(TranscriptSegment.start_ms == segment.start_ms)
                .where(TranscriptSegment.end_ms == segment.end_ms)
                .order_by(TranscriptSegment.id)
            ).first()
            if existing is not None:
                return TranscriptSegmentPayload.from_db(existing)

            if job.status in {JobStatus.DONE, JobStatus.FAILED}:
                raise RuntimeError(
                    f"Job '{job_id}' đã kết thúc, không thể thêm transcript."
                )
            source = db.get(Source, job.source_id)
            if source is None:
                raise RuntimeError(f"Source của Job '{job_id}' không tồn tại.")

            record = TranscriptSegment(
                job_id=job_id,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                speaker=segment.speaker,
                text=text,
            )
            source.duration = max(source.duration or 0.0, segment.end_ms / 1000)
            job.status = JobStatus.PROCESSING
            job.stage = JobStage.TRANSCRIBING
            job.progress = max(job.progress, 0.05)
            job.error = None
            job.updated_at = datetime.utcnow()
            db.add(source)
            db.add(job)
            db.add(record)
            db.commit()
            db.refresh(record)
            return TranscriptSegmentPayload.from_db(record)

    async def _publish_job_event(
        self,
        job_id: str,
        event: TranscriptDeltaEvent,
    ) -> None:
        try:
            await event_bus.publish(job_id, event)
        except Exception:
            logger.exception(
                "Confirmed transcript persisted but Job event publish failed job=%s",
                job_id,
            )

    async def _publish_recording_event(
        self,
        websocket: JsonWebSocket,
        event: TranscriptDeltaEvent,
    ) -> None:
        try:
            await websocket.send_json(event.model_dump(mode="json"))
        except Exception:
            logger.exception(
                "Confirmed transcript persisted but recording WebSocket publish failed"
            )
