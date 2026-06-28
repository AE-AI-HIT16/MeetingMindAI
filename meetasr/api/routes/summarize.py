"""Meeting summarization endpoint with background processing.
Implements the async pattern:
1. Accept file upload → save to temp dir
2. Create meeting record with status 'pending'
3. Return 202 Accepted immediately
4. Process ASR + LLM in background worker
5. Client polls GET /v1/meeting/{id}/status
"""
from __future__ import annotations
import uuid
import logging
from fastapi import (
    APIRouter, BackgroundTasks, Depends, HTTPException,
    UploadFile, status, File, Form,
)
from meetasr.api.mock_db import create_meeting, update_status, get_meeting
from meetasr.api.dependencies import get_pipeline, save_upload, safe_remove


logger = logging.getLogger(__name__)
router = APIRouter(tags=["Meeting"])


def _process_meeting(meeting_id: str, audio_path: str, language: str) -> None:
    """Background worker: run full ASR + LLM pipeline.
    Args:
        meeting_id: Meeting identifier for status tracking.
        audio_path: Path to the temporary audio file.
        language: Output language for LLM summarization.
    """
    try:
        update_status(meeting_id, "processing")
        pipeline = get_pipeline()
        report = pipeline.summarize_meeting(audio_path, language=language)
        update_status(meeting_id, "completed", result=report.to_dict())
        logger.info(f"Meeting {meeting_id} processed")
    except Exception as e:
        logger.error(f"Meeting {meeting_id} failed: {e}")
        update_status(meeting_id, "failed", result={"error": str(e)})
    finally:
        safe_remove(audio_path)
        logger.info(f"Meeting {meeting_id} cleanup")


@router.post("/v1/meeting/summarize", status_code=status.HTTP_202_ACCEPTED)
async def summarize_meeting(
    background_tasks: BackgroundTasks,
    pipeline=Depends(get_pipeline),
    file: UploadFile = File(..., description="Audio file to summarize"),
    language: str = Form("vi", description="Output language: vi | en | zh"),
    llm_model: str = Form("gpt-4o-mini", description="LLM model name (default: gpt-4o-mini)"),
    asr_model: str = Form("sensevoice", description="ASR model: sensevoice (default) | paraformer"),
    include_transcript: bool = Form(True, description="Include full transcript in result"),
    include_topics: bool = Form(True, description="Include key topics extraction"),
    include_actions: bool = Form(True, description="Include action items extraction"),
    include_decisions: bool = Form(True, description="Include decisions extraction"),
) -> dict:
    """Full pipeline: transcribe + LLM summarization (async).
    Returns 202 Accepted with a meeting_id for status polling.
    The actual processing runs in the background.
    """
    if pipeline.summarizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "code": "llm_not_configured",
                    "message": "LLM summarizer is not configured. Add 'llm' section to config YAML.",
                }
            },
        )
    audio_path = await save_upload(file)
    meeting_id = str(uuid.uuid4())
    # NOTE: llm_model, asr_model, include_* fields are accepted per spec
    # but not yet passed to pipeline (pending dynamic model selection support).
    create_meeting(meeting_id, file.filename or "audio.mp3", audio_path)
    background_tasks.add_task(_process_meeting, meeting_id, audio_path, language)
    return {"meeting_id": meeting_id, "status": "pending"}


@router.get("/v1/meeting/{meeting_id}/status")
async def get_status(meeting_id: str) -> dict:
    """Poll the processing status of a meeting.
    Status values: pending → processing → completed | failed
    """
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "meeting_not_found",
                    "message": f"Meeting '{meeting_id}' not found.",
                }
            },
        )
    return meeting