"""
Phase 2 — Backend API & Realtime Channel.
Trách nhiệm: AI Engineer 3

Routes:
  POST   /v1/sources                      — Upload media file (stream to disk)
  GET    /v1/sources                      — List sources
  GET    /v1/sources/{source_id}          — Get source detail
  GET    /v1/sources/{source_id}/media    — Serve media với HTTP Range (video tua được)
  GET    /v1/jobs/{job_id}                — Get job status
  GET    /v1/jobs/{job_id}/queue_position — Vị trí hàng đợi
  GET    /v1/documents/{doc_id}           — Get document
  POST   /v1/documents/{doc_id}/finalize  — Finalize: chọn summary | full_text
  WS     /v1/jobs/{job_id}/events         — WebSocket realtime events + replay từ DB

===========================================================================
MOCK — CÁC DEPENDENCY CỦA KỸ SƯ KHÁC
===========================================================================

[MOCK-1] In-memory dict _mock_sources/_mock_jobs/_mock_segments/_mock_documents
         (import từ job_queue.py) thay cho SQLModel DB của Data Engineer.
         Thay bằng Session + repository khi Data Engineer thêm realtime_models.py.
         Xem: meet_docs/docs/15_ai_eng3_mock_dependencies.md

[MOCK-4] Hàm _stream_save_to_disk() thay cho StorageBackend của Data Engineer.
         Khi Data Engineer xây meetasr/storage/backend.py, thay bằng:
             from meetasr.storage.backend import get_storage
             path = await get_storage().save_stream(file.file, dest, MAX_UPLOAD_BYTES)
         Xem: meet_docs/docs/15_ai_eng3_mock_dependencies.md
===========================================================================
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Optional

from fastapi import (
    APIRouter, File, HTTPException, Request, UploadFile, WebSocket,
    WebSocketDisconnect, status,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from meetasr.realtime.events import EventBus
from meetasr.realtime.job_queue import (
    RealtimeJobQueue,
    _mock_documents,
    _mock_jobs,
    _mock_segments,
    _mock_sources,
    _finalize_mock,
    create_document,
    create_source_and_job,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Phase2-Realtime"])

# ---------------------------------------------------------------
# Cấu hình upload
# ---------------------------------------------------------------
SUPPORTED_MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".webm",        # video
    ".wav", ".mp3", ".m4a", ".ogg", ".flac",  # audio
}
# Giới hạn dung lượng upload (2 GB)
MAX_UPLOAD_BYTES: int = int(os.environ.get("MEETASR_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
# Kích thước chunk khi stream ghi đĩa (64 KB)
UPLOAD_CHUNK_SIZE: int = 64 * 1024
# Thư mục lưu media — TODO (Data Engineer): đọc từ StorageBackend config
MEDIA_DIR: str = os.environ.get("MEETASR_MEDIA_DIR", "data/media")

# ---------------------------------------------------------------
# Helper — lấy EventBus và JobQueue từ app.state
# ---------------------------------------------------------------
def _get_event_bus(request: Request) -> EventBus:
    bus: EventBus | None = getattr(request.app.state, "event_bus", None)
    if bus is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"code": "bus_not_ready", "message": "EventBus chưa khởi động."}},
        )
    return bus


def _get_job_queue(request: Request) -> RealtimeJobQueue:
    q: RealtimeJobQueue | None = getattr(request.app.state, "job_queue", None)
    if q is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"code": "queue_not_ready", "message": "JobQueue chưa khởi động."}},
        )
    return q


# ---------------------------------------------------------------
# MOCK-4: Stream ghi file vào đĩa, kiểm tra dung lượng từng chunk
# ---------------------------------------------------------------
async def _stream_save_to_disk(
    file: UploadFile,
    dest_path: str,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> int:
    """
    Ghi file upload vào đĩa theo từng chunk 64KB.
    Kiểm tra dung lượng TRONG LÚC ghi — không đọc hết vào RAM trước.

    TODO (Data Engineer): thay toàn bộ hàm này bằng:
        from meetasr.storage.backend import get_storage
        return await get_storage().save_stream(file.file, dest_path, max_bytes)

    Args:
        file: UploadFile từ FastAPI.
        dest_path: Đường dẫn lưu file đích.
        max_bytes: Giới hạn dung lượng tối đa cho phép.

    Returns:
        int: Tổng số byte đã ghi.

    Raises:
        HTTPException 413: Nếu file vượt max_bytes.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    total_written = 0

    with open(dest_path, "wb") as f:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total_written += len(chunk)
            if total_written > max_bytes:
                # Xoá file dở, trả lỗi ngay lập tức
                f.close()
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={
                        "error": {
                            "code": "file_too_large",
                            "message": (
                                f"File vượt giới hạn {max_bytes // 1024 // 1024} MB. "
                                f"Đã nhận: {total_written // 1024 // 1024} MB."
                            ),
                        }
                    },
                )
            f.write(chunk)

    return total_written


# ---------------------------------------------------------------
# POST /v1/sources — Upload media
# ---------------------------------------------------------------
@router.post("/v1/sources", status_code=status.HTTP_202_ACCEPTED)
async def upload_source(
    request: Request,
    file: UploadFile = File(..., description="Video (mp4/mkv/webm) hoặc Audio (wav/mp3/m4a)"),
) -> dict:
    """
    Upload file media, tạo Source + Job queued, đẩy vào job queue.

    Trả về ngay lập tức (202 Accepted) — không chờ xử lý xong.

    Response:
        {
            "source_id": "...",
            "job_id": "...",
            "status": "queued"
        }
    """
    # Kiểm tra extension
    filename = file.filename or "upload.mp4"
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in SUPPORTED_MEDIA_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "unsupported_format",
                    "message": (
                        f"Định dạng '{ext}' không được hỗ trợ. "
                        f"Hỗ trợ: {', '.join(sorted(SUPPORTED_MEDIA_EXTENSIONS))}"
                    ),
                }
            },
        )

    # Xác định media_type
    media_type = "video/mp4" if ext in {".mp4", ".mkv", ".webm"} else "audio/mpeg"

    # Chuẩn bị đường dẫn lưu file
    file_id = str(uuid.uuid4())
    dest_path = os.path.join(MEDIA_DIR, f"{file_id}{ext}")

    # MOCK-4: Stream ghi đĩa (thay bằng StorageBackend khi Data Engineer xong)
    try:
        bytes_written = await _stream_save_to_disk(file, dest_path, MAX_UPLOAD_BYTES)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "upload_failed", "message": str(exc)}},
        )

    logger.info("Uploaded %s → %s (%d bytes)", filename, dest_path, bytes_written)

    # MOCK-1: Tạo Source + Job trong mock DB (thay bằng repository khi Data Engineer xong)
    source_id, job_id = create_source_and_job(
        filename=filename,
        media_type=media_type,
        storage_path=dest_path,
    )

    # Đẩy vào job queue
    job_queue = _get_job_queue(request)
    try:
        await job_queue.enqueue(job_id)
    except asyncio.QueueFull:
        # Xoá file + record vừa tạo
        try:
            os.remove(dest_path)
        except OSError:
            pass
        _mock_sources.pop(source_id, None)
        _mock_jobs.pop(job_id, None)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "code": "queue_full",
                    "message": f"Hàng đợi đã đầy ({job_queue.queue_size()} jobs). Thử lại sau.",
                }
            },
        )

    return {
        "source_id": source_id,
        "job_id": job_id,
        "status": "queued",
        "filename": filename,
    }


# ---------------------------------------------------------------
# GET /v1/sources — List sources
# ---------------------------------------------------------------
@router.get("/v1/sources")
async def list_sources(skip: int = 0, limit: int = 20) -> dict:
    """
    Trả về danh sách sources đã upload.

    TODO (Data Engineer): thay body bằng:
        sources = repository.get_sources(db, skip=skip, limit=limit)
        return {"items": [s.model_dump() for s in sources], "total": ...}
    """
    # MOCK-1: đọc từ dict (thay bằng DB query)
    all_sources = list(_mock_sources.values())
    # Sort theo created_at DESC
    all_sources.sort(key=lambda s: s["created_at"], reverse=True)
    paged = all_sources[skip : skip + limit]
    return {"items": paged, "total": len(all_sources)}


# ---------------------------------------------------------------
# GET /v1/sources/{source_id} — Source detail
# ---------------------------------------------------------------
@router.get("/v1/sources/{source_id}")
async def get_source(source_id: str) -> dict:
    """
    Chi tiết 1 source (bao gồm job liên quan).

    TODO (Data Engineer): thay bằng repository.get_source(db, source_id)
    """
    source = _mock_sources.get(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "source_not_found", "message": f"Source '{source_id}' không tồn tại."}},
        )

    # Tìm job gắn với source này
    related_jobs = [j for j in _mock_jobs.values() if j["source_id"] == source_id]

    return {**source, "jobs": related_jobs}


# ---------------------------------------------------------------
# GET /v1/sources/{source_id}/media — Serve media với HTTP Range
# ---------------------------------------------------------------
@router.get("/v1/sources/{source_id}/media")
async def get_source_media(
    source_id: str,
    request: Request,
) -> StreamingResponse:
    """
    Serve file media với hỗ trợ HTTP Range để trình duyệt tua được.

    Trả về:
      - 200 OK + toàn bộ file nếu không có Range header
      - 206 Partial Content nếu có Range header

    TODO (Data Engineer — MinIO/S3):
        Khi chuyển sang S3Storage, thay endpoint này để trả presigned URL:
            url = await get_storage().get_presigned_url(source.storage_path, expires=3600)
            return RedirectResponse(url)
        Component frontend (api.mediaUrl()) không cần sửa.
    """
    source = _mock_sources.get(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "source_not_found", "message": f"Source '{source_id}' không tồn tại."}},
        )

    file_path: str = source["storage_path"]
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "file_not_found", "message": "File media không tồn tại trên disk."}},
        )

    media_type: str = source.get("media_type", "application/octet-stream")
    file_size: int = os.path.getsize(file_path)

    range_header: str | None = request.headers.get("range")

    if range_header:
        # Parse "bytes=start-end"
        start, end = _parse_range_header(range_header, file_size)

        def _iter_range():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(UPLOAD_CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            _iter_range(),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range":  f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges":  "bytes",
                "Content-Length": str(end - start + 1),
            },
        )
    else:
        # Trả toàn bộ file
        def _iter_full():
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            _iter_full(),
            status_code=200,
            media_type=media_type,
            headers={
                "Accept-Ranges":  "bytes",
                "Content-Length": str(file_size),
            },
        )


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int]:
    """
    Parse HTTP Range header dạng "bytes=start-end".
    Hỗ trợ "bytes=start-" (đến cuối file).

    Raises:
        HTTPException 416: Range không hợp lệ.
    """
    try:
        unit, ranges = range_header.strip().split("=", 1)
        if unit.lower() != "bytes":
            raise ValueError("Only bytes range supported")
        start_str, end_str = ranges.split("-", 1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start < 0:
            raise ValueError("Invalid range")
        return start, end
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail={
                "error": {
                    "code": "invalid_range",
                    "message": f"Range header không hợp lệ: {range_header}. Lỗi: {exc}",
                }
            },
        ) from exc


# ---------------------------------------------------------------
# GET /v1/jobs/{job_id} — Job status
# ---------------------------------------------------------------
@router.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    """
    Trả về trạng thái và tiến độ của job.

    TODO (Data Engineer): thay bằng repository.get_job(db, job_id)
    """
    job = _mock_jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "job_not_found", "message": f"Job '{job_id}' không tồn tại."}},
        )
    return job


# ---------------------------------------------------------------
# GET /v1/jobs/{job_id}/queue_position — Vị trí hàng đợi
# ---------------------------------------------------------------
@router.get("/v1/jobs/{job_id}/queue_position")
async def get_queue_position(job_id: str, request: Request) -> dict:
    """
    Trả về vị trí của job trong hàng đợi.
    Hữu ích để frontend hiển thị "Bạn đang ở vị trí thứ N trong hàng đợi".

    Response:
        {
            "job_id": "...",
            "position": 2,       // 0 = đang xử lý, None = không trong queue
            "queue_size": 5
        }
    """
    job = _mock_jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "job_not_found", "message": f"Job '{job_id}' không tồn tại."}},
        )

    job_queue = _get_job_queue(request)
    position = job_queue.queue_position(job_id)

    return {
        "job_id": job_id,
        "status": job["status"],
        "position": position,
        "queue_size": job_queue.queue_size(),
    }


# ---------------------------------------------------------------
# GET /v1/documents/{doc_id} — Get document
# ---------------------------------------------------------------
@router.get("/v1/documents/{doc_id}")
async def get_document(doc_id: str) -> dict:
    """
    Trả về document (markdown + metadata).

    TODO (Data Engineer): thay bằng repository.get_document(db, doc_id)
    """
    doc = _mock_documents.get(doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "document_not_found", "message": f"Document '{doc_id}' không tồn tại."}},
        )
    return doc


# ---------------------------------------------------------------
# POST /v1/documents/{doc_id}/finalize — Finalize document
# ---------------------------------------------------------------
class FinalizeRequest(BaseModel):
    mode: str  # "summary" | "full_text"


@router.post("/v1/documents/{doc_id}/finalize", status_code=status.HTTP_200_OK)
async def finalize_document(doc_id: str, body: FinalizeRequest) -> dict:
    """
    Finalize document sau khi job done.
    Client gọi sau khi nhận sự kiện WS {"type": "done"}.

    Body: {"mode": "summary" | "full_text"}

    Response: document đã finalize với markdown đầy đủ.

    Lưu ý: Endpoint nhận doc_id nhưng hiện tại sẽ tìm document qua source để
    lấy segments. Cần source_id từ document record.
    """
    if body.mode not in ("summary", "full_text"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "invalid_mode",
                    "message": "mode phải là 'summary' hoặc 'full_text'.",
                }
            },
        )

    doc = _mock_documents.get(doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "document_not_found", "message": f"Document '{doc_id}' không tồn tại."}},
        )

    source_id = doc["source_id"]

    # Tìm job của source
    related_jobs = [j for j in _mock_jobs.values() if j["source_id"] == source_id]
    if not related_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "job_not_found", "message": "Không tìm thấy job cho source này."}},
        )

    job = related_jobs[-1]  # lấy job mới nhất
    if job["status"] != "done":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "code": "job_not_done",
                    "message": f"Job chưa xử lý xong (status={job['status']}). Vui lòng đợi sự kiện 'done'.",
                }
            },
        )

    # Lấy segments từ mock DB
    segments = _mock_segments.get(job["id"], [])

    # MOCK-3: Gọi finalizer giả lập (thay bằng DocumentPlanner AI Eng 2)
    markdown = await _finalize_mock(segments, body.mode)

    # Cập nhật document
    doc["mode"] = body.mode
    doc["markdown"] = markdown
    # TODO (Data Engineer): repository.update_document(db, doc_id, mode=body.mode, markdown=markdown)

    logger.info("Document %s finalized (mode=%s)", doc_id, body.mode)
    return doc


# ---------------------------------------------------------------
# POST /v1/sources/{source_id}/documents — Tạo document live
# (Được tạo khi job bắt đầu, để có doc_id cho frontend)
# ---------------------------------------------------------------
@router.post("/v1/sources/{source_id}/documents", status_code=status.HTTP_201_CREATED)
async def create_live_document(source_id: str) -> dict:
    """
    Tạo document placeholder (mode=live) cho source ngay khi job bắt đầu.
    Frontend dùng doc_id này để theo dõi và sau đó gọi /finalize.

    TODO (Data Engineer): thay bằng repository.create_document(db, source_id, mode="live", markdown="")
    """
    source = _mock_sources.get(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "source_not_found", "message": f"Source '{source_id}' không tồn tại."}},
        )

    doc_id = create_document(source_id=source_id, mode="live", markdown="")
    doc = _mock_documents[doc_id]
    logger.info("Live document created: doc_id=%s source_id=%s", doc_id, source_id)
    return doc


# ---------------------------------------------------------------
# WS /v1/jobs/{job_id}/events — Realtime event stream
# ---------------------------------------------------------------
@router.websocket("/v1/jobs/{job_id}/events")
async def job_events_ws(websocket: WebSocket, job_id: str) -> None:
    """
    WebSocket endpoint phát realtime events cho 1 job.

    Protocol (Server → Client):
        {"type": "status",           "stage": "extracting_audio|transcribing|generating_doc", "progress": 0.42}
        {"type": "transcript_delta", "segment": {"start_ms": ..., "end_ms": ..., "speaker": 1, "text": "..."}}
        {"type": "doc_delta",        "section_id": "s1", "markdown": "..."}
        {"type": "done",             "duration_ms": ..., "num_segments": ...}
        {"type": "error",            "code": "...", "message": "..."}
        {"type": "ping"}             -- keepalive mỗi 30s nếu không có event mới

    Khi reconnect: server tự động replay toàn bộ transcript_delta đã có trong DB
    để client không bị mất dữ liệu.

    TODO (Data Engineer): thay phần "replay từ mock DB" bằng:
        segments = repository.get_segments_by_job(db, job_id)
        for seg in segments:
            await websocket.send_json({"type": "transcript_delta", "segment": seg.model_dump()})
    """
    await websocket.accept()
    logger.info("WS connected: job=%s client=%s", job_id, websocket.client)

    # Kiểm tra job tồn tại
    job = _mock_jobs.get(job_id)
    if job is None:
        await websocket.send_json({
            "type": "error",
            "code": "job_not_found",
            "message": f"Job '{job_id}' không tồn tại.",
        })
        await websocket.close(code=1008)
        return

    # ----------------------------------------------------------------
    # REPLAY — gửi lại transcript_delta đã có trong DB khi reconnect
    # ----------------------------------------------------------------
    # MOCK-1: đọc từ dict (thay bằng repository.get_segments_by_job)
    existing_segments = _mock_segments.get(job_id, [])
    if existing_segments:
        logger.info("WS replay: %d segments for job=%s", len(existing_segments), job_id)
        for seg in existing_segments:
            try:
                await websocket.send_json({
                    "type": "transcript_delta",
                    "segment": {
                        "start_ms": seg["start_ms"],
                        "end_ms":   seg["end_ms"],
                        "speaker":  seg["speaker"],
                        "text":     seg["text"],
                    },
                })
            except Exception:
                # Client đã ngắt trong lúc replay
                logger.warning("WS replay interrupted for job=%s", job_id)
                return

    # ----------------------------------------------------------------
    # Nếu job đã xong — gửi event cuối rồi đóng
    # ----------------------------------------------------------------
    if job["status"] == "done":
        await websocket.send_json({
            "type": "done",
            "duration_ms": int(_mock_sources.get(job["source_id"], {}).get("duration_s", 0) * 1000),
            "num_segments": len(existing_segments),
        })
        await websocket.close()
        return

    if job["status"] == "failed":
        await websocket.send_json({
            "type": "error",
            "code": "job_failed",
            "message": job.get("error") or "Job thất bại.",
        })
        await websocket.close(code=1011)
        return

    # ----------------------------------------------------------------
    # Subscribe vào EventBus — nhận sự kiện realtime
    # ----------------------------------------------------------------
    bus: EventBus = websocket.app.state.event_bus
    q = await bus.subscribe(job_id)

    try:
        while True:
            try:
                # Chờ sự kiện tối đa 30 giây, sau đó gửi ping keepalive
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                await websocket.send_json(event)

                # Kết thúc stream khi nhận done/error
                if event.get("type") in ("done", "error"):
                    logger.info("WS stream ended (type=%s) for job=%s", event.get("type"), job_id)
                    break

            except asyncio.TimeoutError:
                # Gửi ping để giữ kết nối
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break  # client đã đóng

    except WebSocketDisconnect:
        logger.info("WS disconnected: job=%s (client disconnect)", job_id)
    except Exception as exc:
        logger.exception("WS error job=%s: %s", job_id, exc)
    finally:
        await bus.unsubscribe(job_id, q)
        logger.info("WS cleanup done: job=%s", job_id)