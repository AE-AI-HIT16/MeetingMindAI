"""API routes cho Sources — Phase 2.

Cung cấp 5 endpoint để frontend-next tương tác với tệp media:

    GET    /v1/sources          → danh sách tất cả Source (trang Library)
    POST   /v1/sources          → upload file, tạo Source + Job mới
    GET    /v1/sources/{id}     → chi tiết 1 Source
    GET    /v1/sources/{id}/media → stream file audio/video (HTTP Range)
    DELETE /v1/sources/{id}     → xóa Source cascade (file + DB)

Frontend-next gọi /v1/... và Next.js proxy tự forward sang FastAPI port 8000.
"""
from __future__ import annotations

import logging
import mimetypes
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from meetasr.backend.api.schemas_phase2 import CreateSourceResponse
from meetasr.backend.db.connection import get_db
from meetasr.backend.db.models_phase2 import DocumentMode, Job, JobStatus, MediaType, Source
from meetasr.backend.realtime.job_worker import job_queue
from meetasr.backend.storage import get_storage
from meetasr.backend.storage.backend import StorageBackend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/sources", tags=["sources"])

# ---------------------------------------------------------------------------
# Dependency: storage backend (singleton per process)
# ---------------------------------------------------------------------------

_storage: Optional[StorageBackend] = None


def get_storage_backend() -> StorageBackend:
    """Trả về StorageBackend singleton, khởi tạo lần đầu khi cần."""
    global _storage
    if _storage is None:
        _storage = get_storage()
    return _storage


# ---------------------------------------------------------------------------
# Pydantic response schemas (mirror types.ts của frontend)
# ---------------------------------------------------------------------------

class SourceDocumentRef(BaseModel):
    """Document ID + mode để Library có thể mở lại tài liệu đã lưu."""

    id: str
    mode: str


class SourceResponse(BaseModel):
    """Schema trả về cho một Source — khớp với interface Source trong types.ts."""

    id: str
    title: str                  # tên file gốc (filename)
    mediaType: str              # "audio" | "video"
    durationMs: Optional[int]   # duration * 1000; None nếu chưa trích xuất
    createdAt: str              # ISO 8601 string
    status: str                 # "processing" | "done" | "failed"
    jobId: Optional[str]        # Job để frontend kết nối WebSocket
    docs: List[str]             # ["live", "summary", "full_text"]
    documents: List[SourceDocumentRef]

    @classmethod
    def from_orm(cls, source: Source) -> "SourceResponse":
        """Chuyển đổi SQLModel Source → Pydantic SourceResponse."""
        job = source.job
        if job is None:
            job_status = "failed"
        elif job.status == JobStatus.DONE:
            job_status = "done"
        elif job.status == JobStatus.FAILED:
            job_status = "failed"
        else:
            job_status = "processing"

        visible_documents = [
            doc
            for doc in source.documents
            if doc.mode == DocumentMode.LIVE or doc.markdown.strip()
        ]
        docs = [
            doc.mode
            for doc in visible_documents
            if doc.mode != DocumentMode.LIVE
        ]
        documents = [
            SourceDocumentRef(id=doc.id, mode=doc.mode)
            for doc in visible_documents
        ]

        return cls(
            id=source.id,
            title=source.filename,
            mediaType=source.media_type,
            durationMs=(
                int(source.duration * 1000)
                if source.duration is not None
                else None
            ),
            createdAt=source.created_at.isoformat(),
            status=job_status,
            jobId=job.id if job is not None else None,
            docs=docs,
            documents=documents,
        )


# ---------------------------------------------------------------------------
# GET /v1/sources — danh sách tất cả Source (trang Library)
# ---------------------------------------------------------------------------

from meetasr.backend.api.auth_deps import get_current_user
from meetasr.backend.db.user_model import User

@router.get("", response_model=List[SourceResponse])
def list_sources(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[Optional[User], Depends(get_current_user)],
) -> List[SourceResponse]:
    """Trả về danh sách tất cả Source của người dùng hiện tại."""
    if not current_user:
        return []
    sources = db.exec(
        select(Source)
        .where(Source.user_id == current_user.id)
        .order_by(Source.created_at.desc())
    ).all()
    return [SourceResponse.from_orm(s) for s in sources]


# ---------------------------------------------------------------------------
# POST /v1/sources — upload file, tạo Source + Job
# ---------------------------------------------------------------------------

@router.post("", response_model=CreateSourceResponse, status_code=status.HTTP_201_CREATED)
async def create_source(
    file: Annotated[UploadFile, File(description="File audio hoặc video cần xử lý.")],
    db: Annotated[Session, Depends(get_db)],
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
    current_user: Annotated[Optional[User], Depends(get_current_user)],
) -> CreateSourceResponse:
    """Upload file media, lưu vào Storage, tạo Source và Job trong DB.

    Sau khi trả về, frontend chuyển hướng tới ``/sources/{id}`` để xem
    tiến trình xử lý thời gian thực.

    Args:
        file:    File upload từ form-data (UploadFile của FastAPI).
        db:      DB session (Dependency Injection).
        storage: Storage backend — local hoặc MinIO (từ biến môi trường).

    Returns:
        ``sourceId`` để mở Source và ``jobId`` để theo dõi tiến trình xử lý.

    Raises:
        HTTPException 400: Nếu không có file hoặc tên file trống.
        HTTPException 500: Nếu lưu file vào storage thất bại.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên file không được để trống.",
        )

    # Xác định loại media từ content-type hoặc phần mở rộng file
    content_type = file.content_type or ""
    if content_type.startswith("video") or file.filename.lower().endswith(
        (".mp4", ".mkv", ".webm", ".mov", ".avi")
    ):
        media_type = MediaType.VIDEO
    else:
        media_type = MediaType.AUDIO

    # Lưu file lên Storage (stream để tránh OOM với file lớn)
    try:
        storage_key = await storage.save(file.file, file.filename)
    except Exception as exc:
        logger.error("Lỗi lưu file '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không thể lưu file: {exc}",
        ) from exc

    # Tạo Source trong DB
    source = Source(
        filename=file.filename,
        media_type=media_type,
        storage_path=storage_key,
        user_id=current_user.id if current_user else None,
    )
    db.add(source)
    db.flush()  # để có source.id trước khi tạo Job

    # Tạo Job với trạng thái queued — Worker sẽ xử lý sau
    job = Job(source_id=source.id, status=JobStatus.QUEUED)
    db.add(job)
    db.commit()
    db.refresh(source)
    await job_queue.enqueue(job.id)

    logger.info("Đã tạo Source '%s' (id=%s) và Job (id=%s).", file.filename, source.id, job.id)
    return CreateSourceResponse(
        sourceId=source.id,
        jobId=job.id,
        status=JobStatus.QUEUED,
    )


# ---------------------------------------------------------------------------
# GET /v1/sources/{source_id} — chi tiết 1 Source
# ---------------------------------------------------------------------------

@router.get("/{source_id}", response_model=SourceResponse)
def get_source(
    source_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[Optional[User], Depends(get_current_user)],
) -> SourceResponse:
    """Trả về chi tiết một Source theo ID."""
    source = db.get(Source, source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source '{source_id}' không tồn tại.",
        )
    if source.user_id is not None:
        if current_user is None or source.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Bạn không có quyền truy cập file này.",
            )
    return SourceResponse.from_orm(source)


# ---------------------------------------------------------------------------
# GET /v1/sources/{source_id}/media — stream file (hỗ trợ HTTP Range)
# ---------------------------------------------------------------------------

@router.get("/{source_id}/media")
async def stream_media(
    source_id: str,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
):
    """Phu vu file audio/video.

    Hanh vi phu thuoc vao backend:
      - S3/MinIO (needs_redirect=True): HTTP 307 redirect sang presigned URL.
        Trinh duyet stream thang tu MinIO, FastAPI khong can doc bytes.
      - Local (needs_redirect=False): doc bytes tu disk, stream voi HTTP Range.

    Args:
        source_id: UUID cua Source.
        request:   Request goc (doc header Range cho local backend).
        db:        DB session.
        storage:   Storage backend hien tai.

    Raises:
        HTTPException 404: Source khong ton tai hoac file khong tim thay.
        HTTPException 416: Range header khong hop le (chi local backend).
    """
    source = db.get(Source, source_id)
    if source is None:
        raise HTTPException(
            status_code=404, detail=f"Source '{source_id}' khong ton tai."
        )

    # --- S3/MinIO: redirect sang presigned URL ---
    if storage.needs_redirect():
        try:
            url = storage.public_url(source.storage_path)
        except Exception as exc:
            logger.error(
                "Khong the tao presigned URL cho '%s': %s", source.storage_path, exc
            )
            raise HTTPException(
                status_code=500, detail="Khong the tao URL xem media."
            ) from exc
        logger.info("Media redirect: source=%s -> presigned URL", source_id)
        return RedirectResponse(url=url, status_code=307)

    # --- Local: doc bytes tu disk va stream voi HTTP Range ---
    try:
        data = await storage.load(source.storage_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="File khong tim thay trong storage."
        )

    total_size = len(data)
    mime_type = (
        mimetypes.guess_type(source.filename)[0] or "application/octet-stream"
    )

    range_header = request.headers.get("Range")
    if range_header:
        try:
            range_val = range_header.strip().replace("bytes=", "")
            start_str, end_str = range_val.split("-")
            start = int(start_str)
            end = int(end_str) if end_str else total_size - 1
        except ValueError:
            raise HTTPException(
                status_code=416, detail="Range header khong hop le."
            )

        if start >= total_size or end >= total_size or start > end:
            raise HTTPException(
                status_code=416, detail="Range vuot qua kich thuoc file."
            )

        chunk = data[start : end + 1]
        return StreamingResponse(
            iter([chunk]),
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{total_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(chunk)),
                "Content-Type": mime_type,
            },
            media_type=mime_type,
        )

    return StreamingResponse(
        iter([data]),
        status_code=200,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(total_size),
            "Content-Type": mime_type,
        },
        media_type=mime_type,
    )



# ---------------------------------------------------------------------------
# DELETE /v1/sources/{source_id} — xóa cascade Source
# ---------------------------------------------------------------------------

@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(
    source_id: str,
    db: Annotated[Session, Depends(get_db)],
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
    current_user: Annotated[Optional[User], Depends(get_current_user)],
) -> None:
    """Xóa Source cùng toàn bộ dữ liệu liên quan (cascade)."""
    source = db.get(Source, source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source '{source_id}' không tồn tại.",
        )
    if source.user_id is not None:
        if current_user is None or source.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Bạn không có quyền xóa file này.",
            )

    # Bước 1: Xóa file khỏi storage (idempotent — không lỗi nếu file đã mất)
    try:
        await storage.delete(source.storage_path)
    except Exception as exc:
        logger.warning("Không thể xóa file storage '%s': %s", source.storage_path, exc)

    # Bước 2: Xóa Source khỏi DB (SQLModel cascade xóa Job, Segments, Documents)
    db.delete(source)
    db.commit()

    logger.info("Đã xóa Source '%s' (id=%s).", source.filename, source_id)
