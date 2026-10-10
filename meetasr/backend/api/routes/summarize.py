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
from pydantic import BaseModel, Field, ValidationError
from sqlmodel import Session

from meetasr.backend.api.dependencies import get_pipeline, safe_remove, save_upload
from meetasr.backend.db.connection import engine, get_db
from meetasr.backend.db import repository
from meetasr.backend.schemas import SentenceInfo, TranscriptResult

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Meeting"])


def _process_meeting(
    meeting_id: str, 
    audio_path: str, 
    language: str, 
    include_flags: dict[str, bool]
) -> None:
    """Background worker: run full ASR + LLM pipeline.
    Uses its own database session (not request-scoped).
    """
    with Session(engine) as db:
        try:
            repository.update_meeting_status(db, meeting_id, "processing")
            
            pipeline = get_pipeline()
            report = pipeline.summarize_meeting(
                audio_path, 
                language=language
            )
            
            # Save results to DB
            repository.save_meeting_result(db, meeting_id, report)
            repository.update_meeting_status(db, meeting_id, "completed")
            
            logger.info(f"Meeting {meeting_id} processed successfully and saved to DB.")
        except Exception as e:
            logger.error(f"Meeting {meeting_id} failed during processing: {e}", exc_info=True)
            db.rollback()
            try:
                repository.update_meeting_status(db, meeting_id, "failed")
            except Exception as inner_e:
                logger.critical(f"Failed to update status to 'failed' for {meeting_id}: {inner_e}")
        finally:
            safe_remove(audio_path)


def _process_text_meeting(
    meeting_id: str, 
    transcript_result: TranscriptResult, 
    language: str, 
    include_flags: dict[str, bool]
) -> None:
    """Background worker: run LLM pipeline directly from a validated TranscriptResult."""
    with Session(engine) as db:
        try:
            repository.update_meeting_status(db, meeting_id, "processing")
            
            pipeline = get_pipeline()
            
            # Chạy summarizer trực tiếp
            report = pipeline.summarizer.summarize(transcript_result)
            
            # Lưu kết quả
            repository.save_meeting_result(db, meeting_id, report)
            
            # Cập nhật thời lượng (duration) ngược lại cho meeting record vì ta không chạy qua ASR
            meeting = repository.get_meeting(db, meeting_id)
            if meeting and transcript_result.duration > 0:
                meeting.duration = transcript_result.duration
                db.add(meeting)
                db.commit()
                
            repository.update_meeting_status(db, meeting_id, "completed")
            
            logger.info(f"Text Meeting {meeting_id} processed successfully.")
        except Exception as e:
            logger.error(f"Text Meeting {meeting_id} failed: {e}", exc_info=True)
            db.rollback()
            try:
                repository.update_meeting_status(db, meeting_id, "failed")
            except Exception as inner_e:
                logger.critical(f"Failed to update status to 'failed' for {meeting_id}: {inner_e}")


@router.post("/v1/meeting/summarize", status_code=status.HTTP_202_ACCEPTED)
async def summarize_meeting(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    pipeline = Depends(get_pipeline),
    file: UploadFile = File(..., description="Audio file to summarize"),
    language: str = Form("vi", description="Output language: vi | en | zh"),
    include_transcript: bool = Form(True, description="Include full transcript in result"),
    include_topics: bool = Form(True, description="Include key topics extraction"),
    include_actions: bool = Form(True, description="Include action items extraction"),
    include_decisions: bool = Form(True, description="Include decisions extraction"),
) -> dict:
    """Full pipeline: transcribe + LLM summarization (async)."""
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
    
    try:
        repository.create_meeting(
            db=db,
            id=meeting_id,
            title=file.filename or "audio.mp3",
            audio_path=audio_path
        )
    except Exception as e:
        safe_remove(audio_path)
        logger.error(f"Failed to create meeting record: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "db_error", "message": "Failed to create meeting record"}}
        )
        
    background_tasks.add_task(
        _process_meeting, 
        meeting_id, 
        audio_path, 
        language, 
        flags
    )
    return {"meeting_id": meeting_id, "status": "pending"}


@router.get("/v1/meeting/{meeting_id}/status")
async def get_status(meeting_id: str, db: Session = Depends(get_db)) -> dict:
    """Poll the processing status of a meeting."""
    meeting = repository.get_meeting(db, meeting_id)
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
        
    return {
        "id": meeting.id,
        "title": meeting.title,
        "status": meeting.status,
        "duration": meeting.duration,
        "created_at": meeting.created_at.isoformat(),
        "updated_at": meeting.updated_at.isoformat()
    }


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
    db: Session = Depends(get_db),
    pipeline = Depends(get_pipeline)
) -> dict:
    """Bypass ASR: Run LLM summarization directly from a JSON transcript."""
    if pipeline.summarizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"code": "llm_not_configured", "message": "LLM is not configured."}}
        )

    # 1. PARSE & VALIDATE JSON NGAY TẠI ROUTE SYNCHRONOUSLY
    # Nếu payload JSON gửi lên bị sai form (thiếu key, sai kiểu dữ liệu), 
    # API sẽ trả về lỗi HTTP 400 Bad Request ngay lập tức cho client biết!
    try:
        t_data = request.transcript_data
        sentences = [SentenceInfo(**s) for s in t_data.get("sentence_info", [])]
        transcript_result = TranscriptResult(
            key=t_data.get("key", "text-upload"),
            text=t_data.get("text", ""),
            duration=float(t_data.get("duration", 0.0)),
            sentence_info=sentences
        )
    except (ValidationError, TypeError, ValueError) as e:
        logger.warning(f"Invalid transcript payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "invalid_payload", "message": f"Malformed transcript schema: {str(e)}"}}
        )

    meeting_id = str(uuid.uuid4())
    flags = {
        "transcript": request.include_transcript,
        "topics": request.include_topics,
        "actions": request.include_actions,
        "decisions": request.include_decisions,
    }
    
    try:
        repository.create_meeting(
            db=db,
            id=meeting_id,
            title=t_data.get("key", "text-upload"),
            audio_path="",
            duration=transcript_result.duration
        )
    except Exception as e:
        logger.error(f"Failed to create text meeting record: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "db_error", "message": "Failed to create meeting record"}}
        )
    
    background_tasks.add_task(
        _process_text_meeting, 
        meeting_id, 
        transcript_result, 
        request.language, 
        flags
    )
    
    return {"meeting_id": meeting_id, "status": "pending"}