"""
RealtimeJobQueue — Hàng đợi xử lý job tuần tự Phase 2.

Trách nhiệm (AI Engineer 3):
  - Nhận job mới (push vào asyncio.Queue)
  - 1 consumer coroutine duy nhất chạy nền → xử lý tuần tự, không tranh GPU
  - Cập nhật trạng thái job (queued → processing → done/failed)
  - Phát sự kiện qua EventBus ở mỗi bước
  - Ghi transcript segments vào mock DB sau khi có kết quả từ AI Eng 1
  - Theo dõi vị trí hàng đợi (queue_position)

===========================================================================
MOCK — CÁC DEPENDENCY CỦA KỸ SƯ KHÁC
===========================================================================

[MOCK-1] _mock_db (dict trong module này) thay cho SQLModel DB của Data Engineer.
         Khi Data Engineer thêm meetasr/db/realtime_models.py, thay các thao tác
         _mock_db bằng gọi repository tương ứng.
         Xem: meet_docs/docs/15_ai_eng3_mock_dependencies.md — mục 1 (DB Models)

[MOCK-2] _process_job_mock() thay cho logic ASR thật của AI Engineer 1.
         Khi AI Engineer 1 hoàn thiện meetasr/realtime/worker.py (process_job_real),
         thay dòng:
             result = await _process_job_mock(job_id, source_path, event_bus)
         bằng:
             from meetasr.realtime.worker import process_job_real
             result = await process_job_real(job_id, source_path, event_bus, pipeline)
         Xem: meet_docs/docs/15_ai_eng3_mock_dependencies.md — mục 2 (Job Processor)

[MOCK-3] _finalize_mock() thay cho DocumentPlanner của AI Engineer 2.
         Khi AI Engineer 2 hoàn thiện meetasr/llm/planner.py, thay bằng:
             from meetasr.llm.planner import DocumentPlanner
             planner = DocumentPlanner(llm)
             markdown = await planner.finalize(segments, mode)
         Xem: meet_docs/docs/15_ai_eng3_mock_dependencies.md — mục 3 (DocumentPlanner)

[MOCK-4] Lưu file dùng os.makedirs + open() thay cho StorageBackend của Data Engineer.
         Khi Data Engineer xây meetasr/storage/backend.py, thay bằng:
             from meetasr.storage.backend import get_storage
             storage = get_storage()
             await storage.save_stream(file_obj, dest_path, max_bytes=MAX_UPLOAD_BYTES)
         Xem: meet_docs/docs/15_ai_eng3_mock_dependencies.md — mục 4 (StorageBackend)
===========================================================================
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any

from meetasr.realtime.events import EventBus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------
MAX_JOB_QUEUE_SIZE: int = int(os.environ.get("MEETASR_MAX_QUEUE", "50"))

# ---------------------------------------------------------------
# MOCK-1: In-memory DB thay thế SQLModel Phase 2
# ---------------------------------------------------------------
# Thay toàn bộ dict này bằng gọi repository khi Data Engineer thêm realtime_models.py
# Cấu trúc chi tiết: meet_docs/docs/15_ai_eng3_mock_dependencies.md

_mock_sources: dict[str, dict[str, Any]] = {}
# Ví dụ record Source:
# {
#   "id": "src-uuid",
#   "filename": "video.mp4",
#   "media_type": "video/mp4",
#   "duration_s": 0.0,       # cập nhật sau khi ffmpeg chạy xong
#   "storage_path": "data/media/src-uuid.mp4",
#   "created_at": "2026-07-18T14:00:00",
# }

_mock_jobs: dict[str, dict[str, Any]] = {}
# Ví dụ record Job:
# {
#   "id": "job-uuid",
#   "source_id": "src-uuid",
#   "status": "queued",       # queued | processing | done | failed
#   "progress": 0.0,          # 0.0–1.0
#   "error": None,
#   "created_at": "2026-07-18T14:00:00",
#   "updated_at": "2026-07-18T14:00:00",
# }

_mock_segments: dict[str, list[dict[str, Any]]] = {}
# job_id → list of segment records
# Ví dụ segment:
# {
#   "id": 1,
#   "job_id": "job-uuid",
#   "start_ms": 0,
#   "end_ms": 3200,
#   "speaker": 0,
#   "text": "Xin chào mọi người.",
# }

_mock_documents: dict[str, dict[str, Any]] = {}
# Ví dụ Document:
# {
#   "id": "doc-uuid",
#   "source_id": "src-uuid",
#   "mode": "summary",        # live | summary | full_text
#   "markdown": "# Tóm tắt\n...",
#   "created_at": "2026-07-18T14:00:00",
# }


# ---------------------------------------------------------------
# MOCK-2: Hàm xử lý job giả lập (thay bằng AI Engineer 1)
# ---------------------------------------------------------------
async def _process_job_mock(
    job_id: str,
    source_path: str,
    event_bus: EventBus,
) -> dict[str, Any]:
    """
    MOCK — Giả lập pipeline ASR xử lý file media.

    Thay hàm này bằng import từ AI Engineer 1 khi sẵn sàng:
        from meetasr.realtime.worker import process_job_real
        return await process_job_real(job_id, source_path, event_bus, pipeline)

    Trả về:
        {
            "num_segments": int,
            "duration_ms": int,
            "segments": [{"start_ms", "end_ms", "speaker", "text"}, ...]
        }
    """
    # Giai đoạn 1 — giả lập ffmpeg tách audio
    await event_bus.publish(job_id, {
        "type": "status",
        "stage": "extracting_audio",
        "progress": 0.05,
    })
    await asyncio.sleep(1.0)  # MOCK: ffmpeg thật sẽ mất 2-10s

    # Giai đoạn 2 — giả lập transcribe chunk
    await event_bus.publish(job_id, {
        "type": "status",
        "stage": "transcribing",
        "progress": 0.10,
    })

    # MOCK: 5 câu giả (thật: ASR sinh ra từ chunks 30s)
    mock_segments = [
        {"start_ms": 0,     "end_ms": 3200,  "speaker": 0, "text": "[MOCK] Xin chào, đây là bản demo giả lập."},
        {"start_ms": 3500,  "end_ms": 7000,  "speaker": 1, "text": "[MOCK] Chúng ta bắt đầu cuộc họp hôm nay."},
        {"start_ms": 7500,  "end_ms": 12000, "speaker": 0, "text": "[MOCK] Agenda gồm 3 điểm chính."},
        {"start_ms": 12500, "end_ms": 18000, "speaker": 1, "text": "[MOCK] Điểm đầu tiên là ngân sách Q3."},
        {"start_ms": 18500, "end_ms": 24000, "speaker": 0, "text": "[MOCK] Chúng ta cần quyết định trước tuần tới."},
    ]

    for i, seg in enumerate(mock_segments):
        await asyncio.sleep(0.5)  # MOCK: độ trễ thật phụ thuộc tốc độ ASR

        # Lưu segment vào mock DB
        seg_record = {**seg, "id": i + 1, "job_id": job_id}
        _mock_segments.setdefault(job_id, []).append(seg_record)

        # Phát transcript_delta cho WS clients
        await event_bus.publish(job_id, {
            "type": "transcript_delta",
            "segment": {
                "start_ms": seg["start_ms"],
                "end_ms":   seg["end_ms"],
                "speaker":  seg["speaker"],
                "text":     seg["text"],
            },
        })

        # Cập nhật progress
        progress = 0.10 + 0.70 * ((i + 1) / len(mock_segments))
        await event_bus.publish(job_id, {
            "type": "status",
            "stage": "transcribing",
            "progress": round(progress, 2),
        })

    # Giai đoạn 3 — giả lập sinh doc thô (doc_delta text)
    await event_bus.publish(job_id, {
        "type": "status",
        "stage": "generating_doc",
        "progress": 0.85,
    })
    await asyncio.sleep(0.5)

    # MOCK doc_delta — AI Engineer 2 sẽ thay bằng LLM thật
    await event_bus.publish(job_id, {
        "type": "doc_delta",
        "section_id": "s1",
        "markdown": (
            "## [MOCK] Nội dung phiên làm việc\n\n"
            + "\n".join(
                f"- **Người nói {s['speaker']}** ({s['start_ms']}ms): {s['text']}"
                for s in mock_segments
            )
        ),
    })

    duration_ms = mock_segments[-1]["end_ms"] if mock_segments else 0
    return {
        "num_segments": len(mock_segments),
        "duration_ms": duration_ms,
        "segments": mock_segments,
    }


# ---------------------------------------------------------------
# MOCK-3: Finalize document giả lập (thay bằng AI Engineer 2)
# ---------------------------------------------------------------
async def _finalize_mock(
    segments: list[dict],
    mode: str,  # "summary" | "full_text"
) -> str:
    """
    MOCK — Sinh markdown cuối cùng từ transcript segments.

    Thay hàm này bằng import từ AI Engineer 2 khi sẵn sàng:
        from meetasr.llm.planner import DocumentPlanner
        planner = DocumentPlanner(llm=pipeline.llm)
        return await planner.finalize(segments, mode)

    Args:
        segments: Danh sách transcript segments đã lưu trong DB.
        mode: "summary" — LLM tóm tắt; "full_text" — ghép toàn văn.

    Returns:
        str: Markdown content của document.
    """
    await asyncio.sleep(0.3)  # MOCK: LLM call thật sẽ mất 2-30s

    full_text = "\n".join(
        f"**Người nói {s['speaker']}** ({s['start_ms']}ms – {s['end_ms']}ms): {s['text']}"
        for s in segments
    )

    if mode == "summary":
        return (
            f"# [MOCK] Tóm tắt phiên\n\n"
            f"> ⚠️ Đây là mock — chưa gọi LLM thật (AI Engineer 2 implement)\n\n"
            f"## Nội dung chính\n\n{full_text}\n\n"
            f"## Quyết định\n\n_Chưa phân tích (mock)_\n\n"
            f"## Action items\n\n_Chưa phân tích (mock)_\n"
        )
    else:  # full_text
        return (
            f"# [MOCK] Toàn văn\n\n"
            f"> ⚠️ Đây là mock — chưa gọi LLM thật (AI Engineer 2 implement)\n\n"
            f"{full_text}\n"
        )


# ---------------------------------------------------------------
# RealtimeJobQueue — AI Engineer 3 chịu trách nhiệm
# ---------------------------------------------------------------
class RealtimeJobQueue:
    """
    Hàng đợi xử lý job tuần tự, 1 consumer duy nhất.

    Mục đích: không tranh GPU khi nhiều user upload cùng lúc.
    Consumer chạy nền (asyncio task), nhận job lần lượt và gọi processor.

    Khởi động:
        q = RealtimeJobQueue(event_bus=bus)
        q.start()          # trong app lifespan startup
        await q.stop()     # trong app lifespan shutdown
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        # Queue nội bộ — chứa job_id chờ xử lý
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_JOB_QUEUE_SIZE)
        # Thứ tự job trong queue để tính queue_position
        self._pending_order: list[str] = []
        self._consumer_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Khởi động consumer coroutine nền. Gọi 1 lần khi app startup."""
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(
                self._consume(), name="realtime-job-consumer"
            )
            logger.info("RealtimeJobQueue: consumer started (maxsize=%d)", MAX_JOB_QUEUE_SIZE)

    async def stop(self) -> None:
        """Dừng consumer gracefully. Gọi khi app shutdown."""
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        logger.info("RealtimeJobQueue: consumer stopped")

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------
    async def enqueue(self, job_id: str) -> None:
        """
        Đẩy job vào cuối hàng đợi.

        Raises:
            asyncio.QueueFull: Nếu queue đã đầy (MAX_JOB_QUEUE_SIZE).
                Caller (route) bắt exception này và trả 429.
        """
        await self._queue.put(job_id)  # raises QueueFull nếu đầy
        self._pending_order.append(job_id)
        logger.info(
            "RealtimeJobQueue: enqueued job=%s (queue_size=%d)",
            job_id, self._queue.qsize(),
        )

    def queue_position(self, job_id: str) -> int | None:
        """
        Vị trí job trong hàng đợi (0-indexed; 0 = đang xử lý / đã xong).
        Trả về None nếu job_id không còn trong pending list.
        """
        try:
            return self._pending_order.index(job_id)
        except ValueError:
            return None

    def queue_size(self) -> int:
        """Số job đang chờ trong queue."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Consumer — vòng lặp tuần tự
    # ------------------------------------------------------------------
    async def _consume(self) -> None:
        """
        Consumer loop: lấy từng job từ queue và xử lý tuần tự.
        Chạy liên tục đến khi bị cancel.
        """
        logger.info("RealtimeJobQueue: consumer loop running")
        while True:
            try:
                job_id = await self._queue.get()
                self._pending_order.remove(job_id)
            except asyncio.CancelledError:
                logger.info("RealtimeJobQueue: consumer cancelled")
                return

            await self._run_job(job_id)
            self._queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        """Xử lý 1 job: update status → process → update done/failed."""
        job = _mock_jobs.get(job_id)
        if job is None:
            logger.error("RealtimeJobQueue: job=%s not found in mock DB", job_id)
            return

        source_id = job["source_id"]
        source = _mock_sources.get(source_id)
        if source is None:
            logger.error("RealtimeJobQueue: source=%s not found for job=%s", source_id, job_id)
            self._set_job_status(job_id, "failed", error="Source not found")
            await self._event_bus.publish(job_id, {
                "type": "error",
                "code": "source_not_found",
                "message": f"Source {source_id} not found",
            })
            return

        logger.info("RealtimeJobQueue: starting job=%s source=%s", job_id, source_id)

        # Cập nhật status → processing
        self._set_job_status(job_id, "processing", progress=0.0)
        await self._event_bus.publish(job_id, {
            "type": "status",
            "stage": "extracting_audio",
            "progress": 0.0,
        })

        try:
            # ============================================================
            # MOCK-2: Gọi processor giả lập
            # Thay bằng: from meetasr.realtime.worker import process_job_real
            #            result = await process_job_real(job_id, source["storage_path"],
            #                                           self._event_bus, pipeline)
            # ============================================================
            result = await _process_job_mock(
                job_id=job_id,
                source_path=source["storage_path"],
                event_bus=self._event_bus,
            )

            # Cập nhật source duration sau khi biết (AI Eng 1 sẽ trả về)
            source["duration_s"] = result["duration_ms"] / 1000.0

            # Cập nhật job → done
            self._set_job_status(job_id, "done", progress=1.0)

            # Phát sự kiện "done"
            await self._event_bus.publish(job_id, {
                "type": "done",
                "duration_ms": result["duration_ms"],
                "num_segments": result["num_segments"],
            })
            logger.info(
                "RealtimeJobQueue: job=%s done (%d segments, %dms)",
                job_id, result["num_segments"], result["duration_ms"],
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("RealtimeJobQueue: job=%s failed: %s", job_id, exc)
            self._set_job_status(job_id, "failed", error=str(exc))
            await self._event_bus.publish(job_id, {
                "type": "error",
                "code": "processing_failed",
                "message": str(exc),
            })

    # ------------------------------------------------------------------
    # Helper — cập nhật mock DB
    # ------------------------------------------------------------------
    @staticmethod
    def _set_job_status(
        job_id: str,
        status: str,
        *,
        progress: float | None = None,
        error: str | None = None,
    ) -> None:
        """
        Cập nhật trạng thái job trong mock DB.
        TODO (Data Engineer): thay bằng repository.update_job_status(db, job_id, status, ...)
        """
        job = _mock_jobs.get(job_id)
        if job:
            job["status"] = status
            job["updated_at"] = datetime.utcnow().isoformat()
            if progress is not None:
                job["progress"] = progress
            if error is not None:
                job["error"] = error


# ---------------------------------------------------------------
# Helper — tạo Source + Job record trong mock DB
# ---------------------------------------------------------------
def create_source_and_job(
    filename: str,
    media_type: str,
    storage_path: str,
) -> tuple[str, str]:
    """
    Tạo record Source và Job trong mock DB.

    TODO (Data Engineer): thay nội dung hàm này bằng:
        source = repository.create_source(db, filename, media_type, storage_path)
        job = repository.create_job(db, source.id)
        return source.id, job.id

    Returns:
        (source_id, job_id)
    """
    now = datetime.utcnow().isoformat()
    source_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    _mock_sources[source_id] = {
        "id": source_id,
        "filename": filename,
        "media_type": media_type,
        "duration_s": 0.0,
        "storage_path": storage_path,
        "created_at": now,
    }

    _mock_jobs[job_id] = {
        "id": job_id,
        "source_id": source_id,
        "status": "queued",
        "progress": 0.0,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }

    return source_id, job_id


def create_document(
    source_id: str,
    mode: str,
    markdown: str,
) -> str:
    """
    Lưu Document vào mock DB.

    TODO (Data Engineer): thay bằng:
        doc = repository.create_document(db, source_id, mode, markdown)
        return doc.id

    Returns:
        doc_id
    """
    doc_id = str(uuid.uuid4())
    _mock_documents[doc_id] = {
        "id": doc_id,
        "source_id": source_id,
        "mode": mode,
        "markdown": markdown,
        "created_at": datetime.utcnow().isoformat(),
    }
    return doc_id
