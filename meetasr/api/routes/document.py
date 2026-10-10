"""Phase 2 document generation endpoint using DocumentPlanner.

Provides /v2/documents endpoints for content-agnostic structured
summarization. Does NOT modify any Phase 1 routes.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal, Union

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
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

class SentencePayload(BaseModel):
    """Validated sentence from a transcript payload."""

    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: int | None = Field(default=None, ge=0)
    char_timestamps: list[list[int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_time_range(self) -> "SentencePayload":
        """Ensure sentence timestamps form a valid range."""
        if self.end < self.start:
            raise ValueError("sentence end must be greater than or equal to start")
        return self


class TranscriptPayload(BaseModel):
    """Validated transcript accepted by the document endpoint."""

    key: str = "doc-upload"
    text: str = ""
    duration: float = Field(default=0.0, ge=0)
    language: str = Field(default="vi", min_length=2, max_length=16)
    sentence_info: list[SentencePayload] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content(self) -> "TranscriptPayload":
        """Reject empty transcripts before an LLM call can hallucinate content."""
        if not self.text.strip() and not any(
            sentence.text.strip() for sentence in self.sentence_info
        ):
            raise ValueError("transcript must contain text or at least one sentence")
        return self


class GenerateDocumentRequest(BaseModel):
    """Request body for Phase 2 document generation."""

    transcript_data: TranscriptPayload = Field(
        ..., description="JSON transcript (same format as /v1/meeting/summarize-text)"
    )


class GenerateDocumentResponse(BaseModel):
    """Immediate response with meeting ID for polling."""

    meeting_id: str
    status: str = "pending"


class DocumentReportResponse(BaseModel):
    """Full document report returned after processing."""

    meeting_id: str
    format: Literal["json"] = "json"
    content_kind: str = ""
    sections: list[dict[str, Any]] = Field(default_factory=list)
    language: str = "vi"
    llm_model: str = ""
    processing_time: float = 0.0


class MarkdownDocumentResponse(BaseModel):
    """Markdown representation of a generated document."""

    meeting_id: str
    format: Literal["markdown"] = "markdown"
    content: str


class DocumentStatusResponse(BaseModel):
    """Current status of a document generation job."""

    id: str
    title: str
    status: str
    created_at: str
    updated_at: str


# ------------------------------------------------------------------
# Background worker
# ------------------------------------------------------------------

def _process_document(
    meeting_id: str,
    transcript_result: TranscriptResult,
    planner: Any,
) -> None:
    """Background worker: run DocumentPlanner pipeline.

    Uses its own database session (not request-scoped).
    """
    with Session(engine) as db:
        try:
            repository.update_meeting_status(db, meeting_id, "processing")

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

    t_data = request.transcript_data
    sentences = [
        SentenceInfo(**sentence.model_dump()) for sentence in t_data.sentence_info
    ]
    transcript_result = TranscriptResult(
        key=t_data.key,
        text=t_data.text,
        duration=t_data.duration,
        language=t_data.language,
        sentence_info=sentences,
    )

    meeting_id = str(uuid.uuid4())
    try:
        repository.create_meeting(
            db=db,
            id=meeting_id,
            title=t_data.key,
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
        pipeline.doc_planner,
    )
    return GenerateDocumentResponse(meeting_id=meeting_id, status="pending")


@router.get(
    "/v2/documents/{meeting_id}/status",
    response_model=DocumentStatusResponse,
)
async def get_document_status(
    meeting_id: str,
    db: Session = Depends(get_db),
) -> DocumentStatusResponse:
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
    return DocumentStatusResponse(
        id=meeting.id,
        title=meeting.title,
        status=meeting.status,
        created_at=meeting.created_at.isoformat(),
        updated_at=meeting.updated_at.isoformat(),
    )


@router.get(
    "/v2/documents/{meeting_id}/report",
    response_model=Union[DocumentReportResponse, MarkdownDocumentResponse],
)
async def get_document_report(
    meeting_id: str,
    format: Literal["json", "markdown"] = "json",
    db: Session = Depends(get_db),
) -> DocumentReportResponse | MarkdownDocumentResponse:
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
        report_data = {}
    if not isinstance(report_data, dict):
        report_data = {}

    raw_sections = report_data.get("sections", [])
    sections = (
        [section for section in raw_sections if isinstance(section, dict)]
        if isinstance(raw_sections, list)
        else []
    )
    content_kind = _string_value(report_data.get("content_kind"), "Tài liệu")

    if format == "markdown":
        # Build markdown from sections
        lines = [f"# {content_kind}", ""]
        for section in sections:
            markdown = _string_value(section.get("markdown"), "")
            if section.get("found", True) and markdown:
                heading = _string_value(section.get("heading"), "Tóm tắt")
                lines += [f"## {heading}", "", markdown, ""]
        return MarkdownDocumentResponse(
            meeting_id=meeting_id,
            content="\n".join(lines).strip(),
        )

    return DocumentReportResponse(
        meeting_id=meeting_id,
        content_kind=content_kind,
        sections=sections,
        language=_string_value(report_data.get("language"), "vi"),
        llm_model=_string_value(report_data.get("llm_model"), ""),
        processing_time=_float_value(report_data.get("processing_time")),
    )


def _string_value(value: object, default: str) -> str:
    """Return a safe response string for persisted, potentially stale JSON."""
    return value.strip() if isinstance(value, str) and value.strip() else default


def _float_value(value: object) -> float:
    """Return a safe numeric response value for persisted JSON."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0
