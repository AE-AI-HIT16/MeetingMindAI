"""REST/WebSocket endpoints for observing Phase 2 processing Jobs."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from meetasr.api.schemas_phase2 import (
    DocDeltaEvent,
    DoneEvent,
    ErrorEvent,
    StatusEvent,
    TranscriptDeltaEvent,
    TranscriptSegmentPayload,
)
from meetasr.db.connection import engine
from meetasr.db.models_phase2 import Document, DocumentMode, Job, JobStatus, TranscriptSegment
from meetasr.realtime.events import event_bus


router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _job_snapshot(job_id: str) -> list[dict]:
    """Rebuild replayable events from persistent Job state."""
    with Session(engine) as db:
        job = db.get(Job, job_id)
        if job is None:
            return [
                ErrorEvent(
                    code="job_not_found",
                    message=f"Job '{job_id}' không tồn tại.",
                ).model_dump(mode="json")
            ]

        events: list[dict] = []
        if job.stage:
            events.append(
                StatusEvent(
                    stage=job.stage,
                    progress=job.progress,
                ).model_dump(mode="json")
            )

        segments = db.exec(
            select(TranscriptSegment)
            .where(TranscriptSegment.job_id == job_id)
            .order_by(TranscriptSegment.id)
        ).all()
        events.extend(
            TranscriptDeltaEvent(
                segment=TranscriptSegmentPayload.from_db(segment)
            ).model_dump(mode="json")
            for segment in segments
        )

        live_document = db.exec(
            select(Document)
            .where(Document.source_id == job.source_id)
            .where(Document.mode == DocumentMode.LIVE)
            .order_by(Document.updated_at.desc())
        ).first()
        if live_document is not None and live_document.markdown:
            events.append(
                DocDeltaEvent(
                    section_id="live",
                    markdown=live_document.markdown,
                ).model_dump(mode="json")
            )

        if job.status == JobStatus.DONE:
            duration_ms = (
                int(job.source.duration * 1000)
                if job.source is not None and job.source.duration is not None
                else max((segment.end_ms for segment in segments), default=0)
            )
            events.append(
                DoneEvent(
                    duration_ms=duration_ms,
                    num_segments=len(segments),
                    live_document_id=(
                        live_document.id if live_document is not None else None
                    ),
                ).model_dump(mode="json")
            )
        elif job.status == JobStatus.FAILED:
            events.append(
                ErrorEvent(
                    code="job_failed",
                    message=job.error or "Job xử lý thất bại.",
                ).model_dump(mode="json")
            )
        return events


@router.websocket("/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    """Replay persisted state, then stream new events for one Job."""
    await websocket.accept()
    queue = await event_bus.subscribe(job_id)
    try:
        snapshot = _job_snapshot(job_id)
        for event in snapshot:
            await websocket.send_json(event)
        if snapshot and snapshot[-1].get("code") == "job_not_found":
            await websocket.close(code=4404)
            return

        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    finally:
        await event_bus.unsubscribe(job_id, queue)
