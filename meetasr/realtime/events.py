"""
EventBus — Kênh publish/subscribe realtime cho Phase 2.

Trách nhiệm (AI Engineer 3):
  - Backpressure: mỗi WS subscriber có asyncio.Queue(maxsize=MAX_QUEUE_PER_SUBSCRIBER)
  - Khi queue đầy → drop sự kiện CŨ NHẤT loại "status"/"transcript_delta"
    (client không mất dữ liệu vì sẽ replay từ DB khi reconnect)
  - Sự kiện "done"/"error" KHÔNG drop — chờ tối đa CRITICAL_TIMEOUT giây
  - Không phụ thuộc vào DB hay model AI nào

Không phụ thuộc thêm gì từ kỹ sư khác.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Hằng số backpressure
# ---------------------------------------------------------------
# Số sự kiện tối đa buffered cho mỗi WS subscriber trước khi drop
MAX_QUEUE_PER_SUBSCRIBER: int = 100

# Các loại sự kiện có thể drop khi queue đầy.
# Client sẽ replay transcript_delta từ DB khi reconnect nên không mất dữ liệu.
DROPPABLE_TYPES: frozenset[str] = frozenset({"status", "transcript_delta", "doc_delta"})

# Timeout (giây) chờ gửi sự kiện quan trọng (done/error) khi queue vẫn đầy
CRITICAL_EVENT_TIMEOUT: float = 10.0


class EventBus:
    """
    Per-job publish/subscribe event bus với backpressure.

    Cách dùng:
        # Trong startup app:
        bus = EventBus()
        app.state.event_bus = bus

        # Worker phát sự kiện:
        await bus.publish("job-abc", {"type": "transcript_delta", "segment": {...}})

        # WS handler đăng ký:
        q = await bus.subscribe("job-abc")
        event = await q.get()           # chờ sự kiện kế tiếp
        await bus.unsubscribe("job-abc", q)   # khi WS disconnect
    """

    def __init__(self) -> None:
        # job_id → danh sách queue của các WS client đang kết nối
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Subscribe / Unsubscribe
    # ------------------------------------------------------------------
    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """
        Tạo queue mới và đăng ký nhận sự kiện của job_id.

        Trả về:
            asyncio.Queue — client đọc từ queue này để nhận sự kiện.
        """
        async with self._lock:
            q: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_PER_SUBSCRIBER)
            self._subscribers.setdefault(job_id, []).append(q)
            logger.debug(
                "EventBus: subscriber added job=%s (total=%d)",
                job_id, len(self._subscribers[job_id]),
            )
            return q

    async def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        """
        Huỷ đăng ký queue khi WS disconnect.
        An toàn nếu gọi nhiều lần.
        """
        async with self._lock:
            subs = self._subscribers.get(job_id, [])
            if q in subs:
                subs.remove(q)
            if not subs:
                self._subscribers.pop(job_id, None)
            logger.debug("EventBus: subscriber removed job=%s", job_id)

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------
    async def publish(self, job_id: str, event: dict) -> None:
        """
        Phát sự kiện tới TẤT CẢ subscriber của job_id.

        Chiến lược backpressure:
          - Droppable (status/transcript_delta/doc_delta):
              Nếu queue đầy → pop phần tử cũ nhất, push mới vào.
          - Critical (done/error):
              Nếu queue đầy → chờ tối đa CRITICAL_EVENT_TIMEOUT giây.
              Nếu vẫn đầy sau timeout → log lỗi nhưng không crash.
        """
        async with self._lock:
            subs = list(self._subscribers.get(job_id, []))

        if not subs:
            logger.debug("EventBus: no subscribers for job=%s, event dropped", job_id)
            return

        event_type: str = event.get("type", "")

        for q in subs:
            if q.full():
                if event_type in DROPPABLE_TYPES:
                    # Drop sự kiện cũ nhất để nhường chỗ cho sự kiện mới
                    try:
                        dropped = q.get_nowait()
                        logger.warning(
                            "EventBus: queue full job=%s — dropped old '%s' event",
                            job_id, dropped.get("type", "?"),
                        )
                    except asyncio.QueueEmpty:
                        pass
                else:
                    # done/error: cố chờ có chỗ
                    try:
                        # Tạo task chờ ít nhất 1 slot trống
                        async def _wait_slot(queue: asyncio.Queue) -> None:
                            while queue.full():
                                await asyncio.sleep(0.05)

                        await asyncio.wait_for(
                            _wait_slot(q), timeout=CRITICAL_EVENT_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            "EventBus: timeout waiting to deliver critical '%s' event job=%s",
                            event_type, job_id,
                        )
                        # Vẫn cố push (sẽ raise QueueFull, bắt bên dưới)

            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.error(
                    "EventBus: FAILED to deliver '%s' event job=%s (queue still full after wait)",
                    event_type, job_id,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def subscriber_count(self, job_id: str) -> int:
        """Số WS client đang listen job_id."""
        return len(self._subscribers.get(job_id, []))

    def active_jobs(self) -> list[str]:
        """Danh sách job_id đang có ít nhất 1 subscriber."""
        return list(self._subscribers.keys())


# ---------------------------------------------------------------
# Singleton — được gán vào app.state.event_bus khi startup
# ---------------------------------------------------------------
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Trả về singleton EventBus. Tạo mới nếu chưa có."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def set_event_bus(bus: EventBus) -> None:
    """Gán EventBus từ app startup (app.state.event_bus)."""
    global _event_bus
    _event_bus = bus
