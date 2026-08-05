import asyncio
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class FinalTranscriptJob:
    job_id: int
    audio: np.ndarray

class FinalTranscriptQueue:
    """
    Queue chứa các job xử lý transcript cuối cùng.

    Producer:
        WebSocket session

    Consumer:
        FinalTranscriptWorker
    """

    def __init__(self, maxsize: int = 100):
        self.queue = asyncio.Queue(maxsize=maxsize)

    async def put(self, job):
        """
        Đưa một job vào queue.
        Nếu queue đầy thì bỏ job mới.
        """

        if self.queue.full():

            logger.warning(
                "Final transcript queue full (qsize=%d), dropping job",
                self.queue.qsize(),
            )
            return

        await self.queue.put(job)

    async def get(self):
        """
        Lấy job tiếp theo.
        """
        return await self.queue.get()

    def task_done(self):
        self.queue.task_done()

    async def join(self):
        await self.queue.join()

    async def clear(self):
        """
        Xóa toàn bộ job còn tồn đọng.
        """

        cleared = 0

        while not self.queue.empty():

            try:
                self.queue.get_nowait()
                self.queue.task_done()
                cleared += 1

            except asyncio.QueueEmpty:
                break

        logger.info(
            "Cleared %d final transcript jobs",
            cleared,
        )

    def qsize(self):
        return self.queue.qsize()

    def empty(self):
        return self.queue.empty()

    def full(self):
        return self.queue.full()
