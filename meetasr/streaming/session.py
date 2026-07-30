from collections import deque
import asyncio
import time

import numpy as np

from meetasr.streaming.audio_queue import AudioQueue


class StreamSession:
    """
    Runtime state của một websocket session.

    Session KHÔNG chứa logic AI.
    Chỉ lưu trạng thái để StreamingProcessor sử dụng.
    """

    def __init__(
        self,
        websocket,
        *,
        ring_buffer_size: int = 300,
        queue_size: int = 1000,
    ):
        self.websocket = websocket

        # ==========================================================
        # Queue
        # ==========================================================

        # Queue nhận chunk từ websocket
        self.audio_queue = AudioQueue(maxsize=queue_size)

        # Queue chứa window audio chờ ASR
        self.asr_queue = asyncio.Queue(
            maxsize=50
        )

        # Chứa các đoạn audio để xử lý trước, đưa ngay kết quả cho fe
        self.temp_asr_queue = asyncio.Queue(maxsize=20)

        # ==========================================================
        # Raw audio
        # ==========================================================

        # Ring buffer lưu chunk gần nhất (debug/reconnect)
        self.audio_buffer = deque(maxlen=ring_buffer_size)

        # Toàn bộ audio CHƯA xử lý
        self.pending_audio = np.empty(0, dtype=np.float32)

        # ==========================================================
        # VAD
        # ==========================================================

        # State nội bộ của Streaming VAD
        self.vad_state = None

        # Các speech segment đã hoàn chỉnh
        # mỗi phần tử là np.ndarray
        self.ready_segments = deque()

        # ==========================================================
        # ASR
        # ==========================================================

        # Window tối đa
        self.max_window_seconds = 30.0

        # Window tối thiểu để bắt đầu infer
        self.min_window_seconds = 4.0

        # Chu kỳ infer
        self.infer_interval = 0.5

        # Thời điểm infer gần nhất
        self.last_infer_time = time.monotonic()

        self.asr_task = None

        # ==========================================================
        # Transcript
        # ==========================================================

        self.partial_transcript = ""

        self.final_transcript = ""

        # ==========================================================
        # Speaker / Model State
        # ==========================================================

        self.asr_state = None

        self.diarizer_state = None

        # ==========================================================
        # Worker
        # ==========================================================

        self.worker_task = None

        self.asr_task = None

        self.temp_asr_task = None

        self.closed = False

    async def close(self):
        """
        Cleanup toàn bộ session.
        """

        if self.closed:
            return
        self.closed = True

        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        if self.asr_task:
            self.asr_task.cancel()
            try:
                await self.asr_task
            except asyncio.CancelledError:
                pass

        if self.temp_asr_task:
            self.temp_asr_task.cancel()
            try:
                await self.temp_asr_task
            except asyncio.CancelledError:
                pass

        await self.audio_queue.clear()

        self.audio_buffer.clear()

        self.pending_audio = np.empty(0, dtype=np.float32)

        self.ready_segments.clear()

        self.partial_transcript = ""

        self.final_transcript = ""

        self.vad_state = None

        self.asr_state = None

        self.diarizer_state = None

        while not self.asr_queue.empty():
            try:
                self.asr_queue.get_nowait()
                self.asr_queue.task_done()

            except asyncio.QueueEmpty:
                break

        while not self.temp_asr_queue.empty():
            try:
                self.temp_asr_queue.get_nowait()
                self.temp_asr_queue.task_done()

            except asyncio.QueueEmpty:
                break