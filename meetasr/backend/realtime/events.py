"""Bounded in-process publish/subscribe channel for Job WebSockets."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


EventPayload = dict[str, Any]


class EventBus:
    """Fan Job events out to connected clients without blocking workers."""

    def __init__(self, max_queue_size: int = 100) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self._max_queue_size = max_queue_size
        self._subscribers: dict[str, set[asyncio.Queue[EventPayload]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> asyncio.Queue[EventPayload]:
        queue: asyncio.Queue[EventPayload] = asyncio.Queue(
            maxsize=self._max_queue_size
        )
        async with self._lock:
            self._subscribers[job_id].add(queue)
        return queue

    async def unsubscribe(
        self,
        job_id: str,
        queue: asyncio.Queue[EventPayload],
    ) -> None:
        async with self._lock:
            subscribers = self._subscribers.get(job_id)
            if subscribers is None:
                return
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(job_id, None)

    async def publish(
        self,
        job_id: str,
        event: BaseModel | Mapping[str, Any],
    ) -> None:
        payload = (
            event.model_dump(mode="json")
            if isinstance(event, BaseModel)
            else dict(event)
        )
        async with self._lock:
            subscribers = tuple(self._subscribers.get(job_id, ()))

        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                # Another producer may have filled the queue between the checks.
                continue


event_bus = EventBus()
