import asyncio
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel
from sqlmodel import Session, select

from meetasr.backend.api.schemas_phase2 import (
    DocDeltaEvent,
    DoneEvent,
    ErrorEvent,
    StatusEvent,
    TranscriptDeltaEvent,
    TranscriptSegmentPayload,
)
from meetasr.backend.db.connection import engine
from meetasr.backend.db.models_phase2 import Document, DocumentMode, Job, JobStatus, Source, TranscriptSegment
from meetasr.backend.realtime.events import event_bus
from meetasr.backend.api.routes import sources
from meetasr.backend.realtime.job_worker import run_job_processing
from meetasr.backend.api.auth_deps import get_current_user
from meetasr.backend.db.user_model import User


router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


class SegmentUpdate(BaseModel):
    """Body cho sửa transcript — text và/hoặc speaker."""
    text: Optional[str] = None
    speaker: Optional[int] = None


@router.patch("/{job_id}/segments/{segment_id}", response_model=TranscriptSegmentPayload)
def update_segment(
    job_id: str,
    segment_id: int,
    payload: SegmentUpdate,
    current_user: Annotated[Optional[User], Depends(get_current_user)],
) -> TranscriptSegmentPayload:
    """Sửa nội dung (hoặc người nói) của một câu transcript rồi lưu."""
    with Session(engine) as db:
        segment = db.get(TranscriptSegment, segment_id)
        if segment is None or segment.job_id != job_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Câu transcript không tồn tại.")

        # Quyền: chỉ chủ sở hữu source mới được sửa
        job = db.get(Job, job_id)
        source = db.get(Source, job.source_id) if job else None
        if source and source.user_id is not None:
            if current_user is None or source.user_id != current_user.id:
                raise HTTPException(status.HTTP_403_FORBIDDEN, "Không có quyền sửa.")

        if payload.text is not None:
            segment.text = payload.text
        if payload.speaker is not None:
            segment.speaker = payload.speaker
        db.add(segment)
        db.commit()
        db.refresh(segment)
        return TranscriptSegmentPayload.from_db(segment)


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

        _STREAMABLE_STAGES = {"extracting_audio", "transcribing", "generating_doc"}
        events: list[dict] = []
        if job.stage and job.stage in _STREAMABLE_STAGES:
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

    # Try to trigger job processing on-demand if it is in QUEUED or FAILED state
    asr_service = getattr(websocket.app.state, "asr_service", None)
    storage = sources.get_storage_backend()
    processing_task = None

    if asr_service is not None:
        with Session(engine) as db:
            job = db.get(Job, job_id)
            if job and job.status in [JobStatus.QUEUED, JobStatus.FAILED]:
                job.status = JobStatus.PROCESSING
                db.add(job)
                db.commit()

                processing_task = asyncio.create_task(
                    run_job_processing(job_id, asr_service, storage)
                )

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
            try:
                await websocket.send_json(event)
            except Exception:
                break
    except WebSocketDisconnect:
        return
    finally:
        await event_bus.unsubscribe(job_id, queue)
        # Do NOT cancel processing_task on disconnect — RunPod job must run to
        # completion regardless of WebSocket lifecycle. The next reconnect will
        # find status=PROCESSING and skip re-triggering; _job_snapshot will
        # replay all persisted state when the client reconnects.
