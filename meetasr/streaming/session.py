import asyncio
from collections import deque

import numpy as np

from meetasr.streaming.audio_archive import AudioArchive
from meetasr.streaming.audio_queue import AudioQueue
from meetasr.streaming.coverage import RealtimeCoverageTracker


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

        # Queue phát sự kiện đã xác nhận transcript
        self.partial_cut_queue = asyncio.Queue(maxsize=20)

        # Chứa các đoạn audio để xử lý trước, đưa ngay kết quả cho fe
        # Preview is disposable: keep only the newest pending request.
        self.temp_asr_queue = asyncio.Queue(maxsize=1)

        # ==========================================================
        # Raw audio
        # ==========================================================

        # Ring buffer lưu chunk gần nhất (debug/reconnect)
        self.audio_buffer = deque(maxlen=ring_buffer_size)

        # Toàn bộ audio CHƯA xử lý
        self.pending_audio = np.empty(0, dtype=np.float32)

        # Buffer dành riêng cho transcript_partial
        self.partial_buffer = np.empty(
            0,
            dtype=np.float32,
        )

        self.partial_buffer_lock = asyncio.Lock()

        # Timeline tuyệt đối của sample đầu tiên
        # trong partial_buffer
        self.partial_buffer_start_ms = 0

        self.audio_archive = AudioArchive()
        self.coverage = RealtimeCoverageTracker()

        # ==========================================================
        # VAD
        # ==========================================================

        # State nội bộ của Streaming VAD
        self.vad_state = None

        # Các SpeechUtterance đã được VAD chốt.
        self.ready_segments = deque()

        # ==========================================================
        # ASR
        # ==========================================================

        # Timeline của lần gần nhất yêu cầu transcript preview.
        self.last_partial_request_ms = 0
        self.first_audio_received_at = None
        self.first_partial_emitted_at = None
        self.last_partial_emitted_at = None
        self.partial_emitted_count = 0
        self.partial_utterance_start_ms = None
        self.last_partial_utterance_emitted_at = None
        self.partial_utterance_emitted_count = 0

        # ==========================================================
        # Transcript
        # ==========================================================

        self.partial_transcript = ""

        self.final_transcript = ""

        # Delta đã xác nhận đến thời điểm này
        # (timeline tuyệt đối)
        self.confirmed_end_ms = 0

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

        self.partial_cleaner_task = None

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

        if self.partial_cleaner_task:
            self.partial_cleaner_task.cancel()
            try:
                await self.partial_cleaner_task
            except asyncio.CancelledError:
                pass

        await self.audio_queue.clear()

        self.audio_buffer.clear()

        self.pending_audio = np.empty(0, dtype=np.float32)

        self.partial_buffer = np.empty(0, dtype=np.float32,)

        self.partial_buffer_start_ms = 0

        self.confirmed_end_ms = 0
        self.last_partial_request_ms = 0
        self.first_audio_received_at = None
        self.first_partial_emitted_at = None
        self.last_partial_emitted_at = None
        self.partial_emitted_count = 0
        self.partial_utterance_start_ms = None
        self.last_partial_utterance_emitted_at = None
        self.partial_utterance_emitted_count = 0

        self.ready_segments.clear()

        self.partial_transcript = ""

        self.final_transcript = ""

        self.vad_state = None

        self.asr_state = None

        self.diarizer_state = None

        self.audio_archive.reset()

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

        while not self.partial_cut_queue.empty():
            try:
                self.partial_cut_queue.get_nowait()
                self.partial_cut_queue.task_done()

            except asyncio.QueueEmpty:
                break
