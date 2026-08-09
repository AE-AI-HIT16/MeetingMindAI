"""Bounded queue for post-session realtime finalization."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

from meetasr.streaming.audio_archive import ArchivedAudio
from meetasr.streaming.coverage import RealtimeCoverageSnapshot

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FinalTranscriptJob:
    """One complete realtime recording waiting for final processing."""

    job_id: str
    audio: np.ndarray | ArchivedAudio
    coverage: RealtimeCoverageSnapshot = field(
        default_factory=RealtimeCoverageSnapshot
    )

    @property
    def duration_ms(self) -> int:
        if isinstance(self.audio, ArchivedAudio):
            return self.audio.duration_ms
        return len(self.audio) * 1000 // 16000

    def load_audio(self) -> np.ndarray:
        if isinstance(self.audio, ArchivedAudio):
            return self.audio.load_numpy()
        return self.audio

    def cleanup(self) -> None:
        if isinstance(self.audio, ArchivedAudio):
            self.audio.cleanup()


class FinalTranscriptQueue:
    """Bounded producer/consumer queue for completed realtime sessions."""

    def __init__(self, maxsize: int = 100) -> None:
        self.queue: asyncio.Queue[FinalTranscriptJob] = asyncio.Queue(
            maxsize=maxsize
        )

    async def put(self, job: FinalTranscriptJob) -> bool:
        """Apply producer backpressure instead of dropping finalization work."""
        await self.queue.put(job)
        return True

    async def get(self) -> FinalTranscriptJob:
        """Return the next finalization job."""
        return await self.queue.get()

    def task_done(self) -> None:
        """Mark the current finalization job as processed."""
        self.queue.task_done()

    async def join(self) -> None:
        """Wait until every enqueued job is processed."""
        await self.queue.join()

    async def clear(self) -> None:
        """Discard queued jobs during application shutdown."""
        cleared = 0
        while not self.queue.empty():
            try:
                job = self.queue.get_nowait()
                job.cleanup()
                self.queue.task_done()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        logger.info("Cleared %d final transcript jobs", cleared)

    def qsize(self) -> int:
        return self.queue.qsize()

    def empty(self) -> bool:
        return self.queue.empty()

    def full(self) -> bool:
        return self.queue.full()
