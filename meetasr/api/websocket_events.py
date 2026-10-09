"""Helpers for finite, disconnect-aware event WebSockets."""

from __future__ import annotations

import asyncio
from collections.abc import Set
from typing import Any

from fastapi import WebSocket


async def close_websocket(
    websocket: WebSocket,
    *,
    code: int = 1000,
    timeout_seconds: float = 0.25,
) -> None:
    """Attempt a graceful close without waiting forever for its handshake."""
    try:
        await asyncio.wait_for(
            websocket.close(code=code),
            timeout=timeout_seconds,
        )
    except Exception:
        # A dead peer cannot acknowledge the close frame. Returning lets the
        # ASGI request finish and Uvicorn release the underlying connection.
        return


async def stream_event_queue(
    websocket: WebSocket,
    queue: asyncio.Queue[dict[str, Any]],
    *,
    terminal_types: Set[str],
) -> None:
    """Forward queued events until the client disconnects or a terminal event.

    Waiting only on ``queue.get()`` prevents an ASGI disconnect from being
    consumed. Keeping one receive task active lets browser disconnects and
    server shutdown release the route immediately.
    """
    receive_task = asyncio.create_task(websocket.receive())
    event_task = asyncio.create_task(queue.get())
    try:
        while True:
            completed, _ = await asyncio.wait(
                {receive_task, event_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if receive_task in completed:
                message = receive_task.result()
                if message.get("type") == "websocket.disconnect":
                    return
                receive_task = asyncio.create_task(websocket.receive())

            if event_task in completed:
                event = event_task.result()
                try:
                    await websocket.send_json(event)
                except Exception:
                    return
                if event.get("type") in terminal_types:
                    await close_websocket(websocket)
                    return
                event_task = asyncio.create_task(queue.get())
    finally:
        for task in (receive_task, event_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            receive_task,
            event_task,
            return_exceptions=True,
        )
