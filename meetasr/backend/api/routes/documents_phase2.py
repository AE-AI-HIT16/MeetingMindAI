"""Finalize and export persisted Phase 2 documents."""

from __future__ import annotations

from typing import Annotated, Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlmodel import Session

from meetasr.backend.api.schemas_phase2 import (
    DocumentGenerationResponse,
    DocumentResponse,
    FinalizeDocumentRequest,
)
from meetasr.backend.api.auth_deps import get_current_user
from meetasr.backend.db.connection import get_db
from meetasr.backend.db.models_phase2 import Document, Source
from meetasr.backend.db.user_model import User
from meetasr.backend.export import (
    MissingExportDependency,
    UnsupportedExportFormat,
    UnsupportedExportPreset,
)
from meetasr.backend.realtime.document_generation import document_generation_queue
from meetasr.backend.services.document_service import (
    DocumentGenerationError,
    DocumentNotFoundError,
    DocumentService,
    InvalidDocumentStateError,
    PlannerUnavailableError,
    TranscriptUnavailableError,
)

router = APIRouter(prefix="/v1/documents", tags=["documents"])


def _raise_service_http_error(exc: Exception) -> None:
    if isinstance(exc, DocumentNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, PlannerUnavailableError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    if isinstance(exc, DocumentGenerationError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Retry-After": "60"},
        ) from exc
    if isinstance(
        exc,
        (InvalidDocumentStateError, TranscriptUnavailableError),
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    raise exc


def _authorize_document(db: Session, document_id: str, user: Optional[User]) -> None:
    """Same rule as sources: a document whose Source has an owner is only
    visible to that owner; guest-mode documents (no owner) stay open."""
    document = db.get(Document, document_id)
    if document is None:
        return  # let the service raise its usual 404
    source = db.get(Source, document.source_id)
    if source is not None and source.user_id is not None:
        if user is None or source.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Bạn không có quyền truy cập tài liệu này.",
            )


CurrentUser = Annotated[Optional[User], Depends(get_current_user)]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: CurrentUser,
) -> DocumentResponse:
    """Return one persisted document by Document ID."""
    _authorize_document(db, document_id, current_user)
    try:
        document = DocumentService(db).get_document(document_id)
    except DocumentNotFoundError as exc:
        _raise_service_http_error(exc)
    return DocumentResponse.from_db(document)


@router.post(
    "/{document_id}/finalize",
    response_model=DocumentGenerationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def finalize_document(
    document_id: str,
    payload: FinalizeDocumentRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    current_user: CurrentUser,
) -> DocumentGenerationResponse:
    """Accept an idempotent background summary/full-text generation job."""
    _authorize_document(db, document_id, current_user)
    pipeline = getattr(request.app.state, "pipeline", None)
    planner = getattr(pipeline, "doc_planner", None)
    try:
        generation = await document_generation_queue.submit(
            db,
            document_id,
            payload.mode,
            planner=planner,
        )
    except (
        DocumentNotFoundError,
        InvalidDocumentStateError,
        TranscriptUnavailableError,
        PlannerUnavailableError,
        DocumentGenerationError,
    ) as exc:
        _raise_service_http_error(exc)
    return DocumentGenerationResponse.from_db(generation)


@router.get("/{document_id}/export")
def export_document(
    document_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: CurrentUser,
    format: Literal["md", "docx", "pdf"] = Query(default="md"),
    preset: str | None = Query(default=None),
) -> Response:
    """Download a persisted document in a supported format/template preset."""
    _authorize_document(db, document_id, current_user)
    try:
        artifact = DocumentService(db).export_document(
            document_id,
            format,
            preset,
        )
    except DocumentNotFoundError as exc:
        _raise_service_http_error(exc)
    except (UnsupportedExportFormat, UnsupportedExportPreset) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MissingExportDependency as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    encoded_filename = quote(artifact.filename)
    return Response(
        content=artifact.content,
        media_type=artifact.media_type,
        headers={
            "Content-Disposition": (
                f"attachment; filename*=UTF-8''{encoded_filename}"
            )
        },
    )
