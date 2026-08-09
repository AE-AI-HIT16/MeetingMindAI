"""Lossless queue draining for a completed realtime recording session."""

from __future__ import annotations

import asyncio
from typing import Protocol

DEFAULT_DRAIN_TIMEOUT_SECONDS = 120.0


class JoinableQueue(Protocol):
    async def join(self) -> None: ...


class DrainSession(Protocol):
    audio_queue: JoinableQueue
    asr_queue: JoinableQueue
    partial_cut_queue: JoinableQueue


class Flushable(Protocol):
    async def flush(self) -> None: ...


async def drain_realtime_session(
    session: DrainSession,
    receiver: Flushable,
    worker: Flushable,
    *,
    timeout_seconds: float = DEFAULT_DRAIN_TIMEOUT_SECONDS,
) -> None:
    """Flush accepted audio through confirmed ASR before workers are cancelled."""

    async def drain() -> None:
        await receiver.flush()
        await session.audio_queue.join()
        await worker.flush()
        await session.asr_queue.join()
        await session.partial_cut_queue.join()

    await asyncio.wait_for(drain(), timeout=timeout_seconds)
