"""Meeting summarization endpoint with background processing.
Implements the async pattern:
1. Accept file upload → save to temp dir
2. Create meeting record with status 'pending'
3. Return 202 Accepted immediately
4. Process ASR + LLM in background worker
5. Client polls GET /v1/meeting/{id}/status
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
)
from pydantic import BaseModel, Field

from meetasr.api.dependencies import get_pipeline, safe_remove, save_upload
from meetasr.api.mock_db import create_meeting, get_meeting, update_status
from meetasr.schemas import SentenceInfo, TranscriptResult

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Meeting"])


def _process_meeting(meeting_id: str, audio_path: str, language: str, include_flags: dict[str, bool]) -> None:
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
        result_dict = report.to_dict()
        if not include_flags.get("transcript"): result_dict.pop("transcript", None)
        if not include_flags.get("topics"): result_dict.pop("topics", None)
        if not include_flags.get("actions"): result_dict.pop("action_items", None)
        if not include_flags.get("decisions"): result_dict.pop("decisions", None)
        update_status(meeting_id, "completed", result=result_dict)
        logger.info(f"Meeting {meeting_id} processed")
    except Exception as e:
        logger.error(f"Meeting {meeting_id} failed: {e}")
        update_status(meeting_id, "failed", result={"error": str(e)})
    finally:
        safe_remove(audio_path)
        logger.info(f"Meeting {meeting_id} cleanup")


def _process_text_meeting(meeting_id: str, transcript_data: dict, language: str, include_flags: dict[str, bool]) -> None:
    """Background worker: run LLM pipeline directly from text data."""
    try:
        update_status(meeting_id, "processing")
        pipeline = get_pipeline()
        
        # 1. Convert Dictionary to TranscriptResult Object
        sentences = [SentenceInfo(**s) for s in transcript_data.get("sentence_info", [])]
        transcript = TranscriptResult(
            key=transcript_data.get("key", "mock-test"),
            text=transcript_data.get("text", ""),
            duration=transcript_data.get("duration", 0.0),
            sentence_info=sentences
        )
        
        # 2. Bypass ASR, call LLM directly
        report = pipeline.summarizer.summarize(transcript)
        
        result_dict = report.to_dict()
        if not include_flags.get("transcript"): result_dict.pop("transcript", None)
        if not include_flags.get("topics"): result_dict.pop("topics", None)
        if not include_flags.get("actions"): result_dict.pop("action_items", None)
        if not include_flags.get("decisions"): result_dict.pop("decisions", None)
        
        update_status(meeting_id, "completed", result=result_dict)
        logger.info(f"Text Meeting {meeting_id} processed")
    except Exception as e:
        logger.error(f"Text Meeting {meeting_id} failed: {e}")
        update_status(meeting_id, "failed", result={"error": str(e)})


@router.post("/v1/meeting/summarize", status_code=status.HTTP_202_ACCEPTED)
async def summarize_meeting(
    background_tasks: BackgroundTasks,
    pipeline: Any = Depends(get_pipeline),
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
    flags = {
        "transcript": include_transcript,
        "topics": include_topics,
        "actions": include_actions,
        "decisions": include_decisions,
    }
    create_meeting(meeting_id, file.filename or "audio.mp3", audio_path)
    background_tasks.add_task(_process_meeting, meeting_id, audio_path, language, flags)
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

class SummarizeTextRequest(BaseModel):
    """Request model for summarizing a pre-existing meeting transcript."""
    transcript_data: dict = Field(..., description="Nội dung file JSON chứa kịch bản cuộc họp")
    language: str = Field("vi", description="Ngôn ngữ đầu ra (vi, en)")
    include_transcript: bool = Field(True, description="Trả về kèm kịch bản")
    include_topics: bool = Field(True, description="Trả về kèm chủ đề")
    include_actions: bool = Field(True, description="Trả về kèm task")
    include_decisions: bool = Field(True, description="Trả về kèm quyết định")

@router.post("/v1/meeting/summarize-text", status_code=status.HTTP_202_ACCEPTED)
async def summarize_meeting_text(
    request: SummarizeTextRequest,
    background_tasks: BackgroundTasks,
    pipeline: Any = Depends(get_pipeline), 
) -> dict:
    """Bypass ASR: Run LLM summarization directly from a JSON transcript."""
    if pipeline.summarizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"code": "llm_not_configured", "message": "LLM is not configured."}}
        )

    meeting_id = str(uuid.uuid4())
    flags = {
        "transcript": request.include_transcript,
        "topics": request.include_topics,
        "actions": request.include_actions,
        "decisions": request.include_decisions,
    }
    
    create_meeting(meeting_id, "text-upload", "")
    
    background_tasks.add_task(
        _process_text_meeting, 
        meeting_id, 
        request.transcript_data, 
        request.language, 
        flags
    )
    
    return {"meeting_id": meeting_id, "status": "pending"}