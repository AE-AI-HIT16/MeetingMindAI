import asyncio
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from sqlmodel import Session

from meetasr.backend.api.schemas_phase2 import (
    DocumentGenerationDoneEvent,
    DocumentGenerationErrorEvent,
    DocumentGenerationResponse,
    DocumentGenerationStatusEvent,
)
from meetasr.backend.db.connection import engine
from meetasr.backend.db.models_phase2 import (
    DocumentGenerationJob,
    DocumentGenerationStatus,
)
from meetasr.backend.realtime.events import event_bus
from meetasr.backend.realtime.document_generation import run_document_generation

router = APIRouter(prefix="/v1/document-jobs", tags=["document-jobs"])


def _get_generation(generation_id: str) -> DocumentGenerationJob | None:
    with Session(engine) as db:
        generation = db.get(DocumentGenerationJob, generation_id)
        if generation is not None:
            db.expunge(generation)
        return generation


def _generation_snapshot(generation_id: str) -> list[dict]:
    generation = _get_generation(generation_id)
    if generation is None:
        return [
            DocumentGenerationErrorEvent(
                generation_job_id=generation_id,
                document_id="",
                code="document_generation_not_found",
                message=f"Document generation job '{generation_id}' không tồn tại.",
            ).model_dump(mode="json")
        ]
    if generation.status == DocumentGenerationStatus.DONE:
        return [
            DocumentGenerationDoneEvent(
                generation_job_id=generation.id,
                document_id=generation.document_id,
                mode=generation.mode,
            ).model_dump(mode="json")
        ]
    if generation.status == DocumentGenerationStatus.FAILED:
        return [
            DocumentGenerationErrorEvent(
                generation_job_id=generation.id,
                document_id=generation.document_id,
                code="document_generation_failed",
                message=generation.error or "Không thể tạo tài liệu.",
                retry_after=60,
            ).model_dump(mode="json")
        ]
    return [
        DocumentGenerationStatusEvent(
            generation_job_id=generation.id,
            document_id=generation.document_id,
            stage=generation.stage,
            progress=generation.progress,
        ).model_dump(mode="json")
    ]


@router.get("/{generation_id}", response_model=DocumentGenerationResponse)
def get_document_generation(
    generation_id: str,
) -> DocumentGenerationResponse:
    """Return the persisted state used to recover after a disconnect."""
    generation = _get_generation(generation_id)
    if generation is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document generation job '{generation_id}' không tồn tại.",
        )
    return DocumentGenerationResponse.from_db(generation)


@router.websocket("/{generation_id}/events")
async def document_generation_events(
    websocket: WebSocket,
    generation_id: str,
) -> None:
    """Replay persisted generation state, then stream live updates."""
    await websocket.accept()

    # Try to trigger document generation on-demand if it is in QUEUED or FAILED state
    pipeline = getattr(websocket.app.state, "pipeline", None)
    planner = getattr(pipeline, "doc_planner", None)
    processing_task = None

    if planner is not None:
        with Session(engine) as db:
            generation = db.get(DocumentGenerationJob, generation_id)
            if generation and generation.status in [DocumentGenerationStatus.QUEUED, DocumentGenerationStatus.FAILED]:
                generation.status = DocumentGenerationStatus.PROCESSING
                db.add(generation)
                db.commit()

                processing_task = asyncio.create_task(
                    run_document_generation(generation_id, planner)
                )

    queue = await event_bus.subscribe(generation_id)
    try:
        snapshot = _generation_snapshot(generation_id)
        for event in snapshot:
            await websocket.send_json(event)
        if (
            snapshot
            and snapshot[-1].get("code")
            == "document_generation_not_found"
        ):
            await websocket.close(code=4404)
            return
        if snapshot and snapshot[-1].get("type") in {
            "document_done",
            "document_error",
        }:
            await websocket.close()
            return

        while True:
            event = await queue.get()
            try:
                await websocket.send_json(event)
            except Exception:
                break
            if event.get("type") in {"document_done", "document_error"}:
                try:
                    await websocket.close()
                except Exception:
                    pass
                return
    except WebSocketDisconnect:
        return
    finally:
        await event_bus.unsubscribe(generation_id, queue)
        # If this is the last client connected, cancel the processing task
        if generation_id not in event_bus._subscribers:
            if processing_task and not processing_task.done():
                processing_task.cancel()
