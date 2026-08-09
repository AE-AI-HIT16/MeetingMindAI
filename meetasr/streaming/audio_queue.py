import asyncio
import logging

logger = logging.getLogger(__name__)

# Định nghĩa kiểu dữ liệu hàng chờ (queue)
class AudioQueue:

    def __init__(self, maxsize=20):
        self.queue = asyncio.Queue(maxsize=maxsize)

    async def put(self, audio):
        """Enqueue audio; drop newest chunk if full (do not steal consumer items)."""

        if self.queue.full():
            logger.warning(
                "Queue full (qsize=%d); dropping incoming chunk (%d bytes)",
                self.queue.qsize(),
                len(audio) if audio is not None else 0,
            )
            return

        await self.queue.put(audio)

    async def get(self):
        return await self.queue.get()

    def task_done(self):
        """Đánh dấu một item lấy từ queue đã được xử lý xong."""
        self.queue.task_done()

    async def join(self):
        """Chờ cho đến khi tất cả item trong queue được xử lý."""
        await self.queue.join()

    # Dùng để xóa toàn bộ dữ liệu trong queue
    async def clear(self):
        """
        Xóa toàn bộ audio đang chờ trong queue.
        Dùng khi session disconnect/cleanup.
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
            "Cleared %d audio chunks from queue",
            cleared
        )

    def qsize(self):
        return self.queue.qsize()

    def empty(self):
        return self.queue.empty()

    def full(self):
        return self.queue.full()
