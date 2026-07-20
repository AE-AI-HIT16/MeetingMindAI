"""Phase 2 document generation endpoint using DocumentPlanner.

Provides /v2/documents endpoints for content-agnostic structured
summarization. Does NOT modify any Phase 1 routes.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import Session

from meetasr.api.dependencies import get_pipeline
from meetasr.db.connection import engine, get_db
from meetasr.db import repository
from meetasr.schemas import SentenceInfo, TranscriptResult

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Document"])


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class GenerateDocumentRequest(BaseModel):
    """Request body for Phase 2 document generation."""

    transcript_data: dict = Field(
        ..., description="JSON transcript (same format as /v1/meeting/summarize-text)"
    )


class GenerateDocumentResponse(BaseModel):
    """Immediate response with meeting ID for polling."""

    meeting_id: str
    status: str = "pending"


class DocumentReportResponse(BaseModel):
    """Full document report returned after processing."""

    meeting_id: str
    content_kind: str = ""
    sections: list[dict] = Field(default_factory=list)
    language: str = "vi"
    llm_model: str = ""
    processing_time: float = 0.0


# ------------------------------------------------------------------
# Background worker
# ------------------------------------------------------------------

def _process_document(
    meeting_id: str,
    transcript_result: TranscriptResult,
) -> None:
    """Background worker: run DocumentPlanner pipeline.

    Uses its own database session (not request-scoped).
    """
    with Session(engine) as db:
        try:
            repository.update_meeting_status(db, meeting_id, "processing")

            pipeline = get_pipeline()
            planner = pipeline.doc_planner
            if planner is None:
                raise RuntimeError("DocumentPlanner not configured")

            report = planner.plan_and_write(transcript_result)

            # Save DocumentReport as JSON in the Report table
            from meetasr.db.models import Report
            db_report = Report(
                meeting_id=meeting_id,
                summary=report.to_json(),
                processing_time=report.processing_time,
                llm_model=report.llm_model,
            )
            db.add(db_report)

            repository.update_meeting_status(db, meeting_id, "completed")
            logger.info("Document %s generated successfully.", meeting_id)
        except Exception as e:
            logger.error("Document %s failed: %s", meeting_id, e, exc_info=True)
            db.rollback()
            try:
                repository.update_meeting_status(db, meeting_id, "failed")
            except Exception as inner_e:
                logger.critical(
                    "Failed to update status to 'failed' for %s: %s",
                    meeting_id, inner_e,
                )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.post(
    "/v2/documents/generate",
    response_model=GenerateDocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_document(
    request: GenerateDocumentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    pipeline=Depends(get_pipeline),
) -> GenerateDocumentResponse:
    """Generate a structured document using DocumentPlanner (async).

    Accepts a pre-existing transcript payload, validates it, then
    dispatches background processing via DocumentPlanner.
    Returns 202 with a meeting_id for status polling.
    """
    if pipeline.doc_planner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "code": "planner_not_configured",
                    "message": (
                        "DocumentPlanner is not configured. "
                        "Add 'llm' section to config YAML."
                    ),
                }
            },
        )

    # Parse transcript payload
    try:
        t_data = request.transcript_data
        sentences = [SentenceInfo(**s) for s in t_data.get("sentence_info", [])]
        transcript_result = TranscriptResult(
            key=t_data.get("key", "doc-upload"),
            text=t_data.get("text", ""),
            duration=float(t_data.get("duration", 0.0)),
            sentence_info=sentences,
        )
    except (TypeError, ValueError) as e:
        logger.warning("Invalid transcript payload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "invalid_payload",
                    "message": f"Malformed transcript schema: {e}",
                }
            },
        )

    meeting_id = str(uuid.uuid4())
    try:
        repository.create_meeting(
            db=db,
            id=meeting_id,
            title=t_data.get("key", "doc-upload"),
            audio_path="",
            duration=transcript_result.duration,
        )
    except Exception as e:
        logger.error("Failed to create document record: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "db_error",
                    "message": "Failed to create document record.",
                }
            },
        )

    background_tasks.add_task(
        _process_document,
        meeting_id,
        transcript_result,
    )
    return GenerateDocumentResponse(meeting_id=meeting_id, status="pending")


@router.get("/v2/documents/{meeting_id}/status")
async def get_document_status(
    meeting_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """Poll the processing status of a document generation job."""
    meeting = repository.get_meeting(db, meeting_id)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "document_not_found",
                    "message": f"Document job '{meeting_id}' not found.",
                }
            },
        )
    return {
        "id": meeting.id,
        "title": meeting.title,
        "status": meeting.status,
        "created_at": meeting.created_at.isoformat(),
        "updated_at": meeting.updated_at.isoformat(),
    }


@router.get("/v2/documents/{meeting_id}/report")
async def get_document_report(
    meeting_id: str,
    format: str = "json",
    db: Session = Depends(get_db),
) -> dict:
    """Retrieve the generated document report.

    Args:
        meeting_id: The meeting/document job ID.
        format: Output format — "json" (structured) or "markdown" (readable).
    """
    from meetasr.db.models import Report
    meeting = repository.get_meeting(db, meeting_id)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "document_not_found",
                    "message": f"Document '{meeting_id}' not found.",
                }
            },
        )

    if meeting.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "code": "not_ready",
                    "message": f"Document is '{meeting.status}', not completed yet.",
                }
            },
        )

    report_row = db.get(Report, meeting_id)
    if not report_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "report_not_found",
                    "message": "Report record not found in database.",
                }
            },
        )

    import json as _json
    try:
        report_data = _json.loads(report_row.summary)
    except (ValueError, TypeError):
        report_data = {"raw": report_row.summary}

    if format == "markdown":
        # Build markdown from sections
        lines = [f"# {report_data.get('content_kind', 'Document')}", ""]
        for sec in report_data.get("sections", []):
            if sec.get("found", True) and sec.get("markdown", "").strip():
                lines += [f"## {sec.get('heading', '')}", "", sec["markdown"], ""]
        return {
            "meeting_id": meeting_id,
            "format": "markdown",
            "content": "\n".join(lines).strip(),
        }

    return {
        "meeting_id": meeting_id,
        "format": "json",
        "content_kind": report_data.get("content_kind", ""),
        "sections": report_data.get("sections", []),
        "language": report_data.get("language", "vi"),
        "llm_model": report_data.get("llm_model", ""),
        "processing_time": report_data.get("processing_time", 0),
    }
