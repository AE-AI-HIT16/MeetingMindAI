import asyncio
import logging

logger = logging.getLogger(__name__)

# Định nghĩa kiểu dữ liệu hàng chờ (queue)
class AudioQueue:

    def __init__(self, maxsize=20):
        self.queue = asyncio.Queue(maxsize=maxsize)

    async def put(self, audio):

        if self.queue.full():
            logger.warning("Queue full")

            prev = await self.queue.get()
            audio = prev + audio

        await self.queue.put(audio)

    async def get(self):
        return await self.queue.get()

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