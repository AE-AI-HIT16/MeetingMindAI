"""Finalize and export persisted Phase 2 documents."""

from __future__ import annotations

from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlmodel import Session

from meetasr.api.schemas_phase2 import (
    DocumentGenerationResponse,
    DocumentResponse,
    FinalizeDocumentRequest,
)
from meetasr.db.connection import get_db
from meetasr.export import (
    MissingExportDependency,
    UnsupportedExportFormat,
)
from meetasr.realtime.document_generation import document_generation_queue
from meetasr.services.document_service import (
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


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> DocumentResponse:
    """Return one persisted document by Document ID."""
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
) -> DocumentGenerationResponse:
    """Accept an idempotent background summary/full-text generation job."""
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
    format: Literal["md", "docx", "pdf"] = Query(default="md"),
) -> Response:
    """Download a persisted document in a supported format."""
    try:
        artifact = DocumentService(db).export_document(
            document_id,
            format,
        )
    except DocumentNotFoundError as exc:
        _raise_service_http_error(exc)
    except UnsupportedExportFormat as exc:
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
