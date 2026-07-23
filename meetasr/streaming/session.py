from collections import deque
from meetasr.streaming.audio_queue import AudioQueue
import asyncio

# Lưu trữ 1 phiên hoạt động của websocket
class StreamSession:
    """
    Lưu toàn bộ trạng thái của một WebSocket session.
    Không chứa logic xử lý AI.
    """

    def __init__(
        self,
        websocket,
        *,
        ring_buffer_size: int = 200,
        queue_size: int = 20,
    ):
        self.websocket = websocket

        # Ring buffer lưu các chunk audio gần nhất
        self.audio_buffer = deque(maxlen=ring_buffer_size)

        # Queue để worker lấy audio xử lý
        self.audio_queue = AudioQueue(maxsize=queue_size)

        # State của VAD (do module VAD quản lý)
        self.vad_state = None

        # State của ASR (cache/context nếu cần)
        self.asr_state = None

        # State của Diarizer
        self.diarizer_state = None

        # Transcript đang được cập nhật
        self.partial_transcript = ""

        # Transcript đã hoàn thành (nếu cần)
        self.final_transcripts = []

        self.worker_task = None

        self.closed = False

    async def close(self):
        """
        Cleanup toàn bộ tài nguyên của session.
        Gọi khi websocket disconnect hoặc timeout.
        """

        # tránh cleanup nhiều lần
        if self.closed:
            return

        self.closed = True

        # 1. Dừng worker
        if self.worker_task:

            self.worker_task.cancel()

            try:
                await self.worker_task

            except asyncio.CancelledError:
                pass

        # 2. Clear audio queue
        await self.audio_queue.clear()

        # 3. Clear ring buffer
        self.audio_buffer.clear()

        # 4. Reset AI states
        self.vad_state = None
        self.asr_state = None
        self.diarizer_state = None

        # 5. Reset transcript
        self.partial_transcript = ""
        self.final_transcripts.clear()