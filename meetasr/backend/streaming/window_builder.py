import logging

import numpy as np


SAMPLE_RATE = 16000

MIN_WINDOW_SECONDS = 1
MAX_WINDOW_SECONDS = 30

logger = logging.getLogger(__name__)



class SegmentWindowBuilder:
    """
    Gom speech segment thành ASR window.

    Không chạy ASR.
    Chỉ tạo window và đẩy vào asr_queue.
    """


    def __init__(self, session,):
        self.session = session

        # Các đoạn speech đang chờ ghép
        self.buffer = []

        self.duration = 0.0


    async def process(self):
        """
        Lấy segment từ VAD.
        Build window 4-30s.
        """

        while self.session.ready_segments:
            segment = (self.session.ready_segments.popleft())

            await self._append_segment(segment)


    async def _append_segment(self, segment: np.ndarray):
        """
        Thêm segment vào buffer.

        - Không cắt segment nếu còn đủ chỗ trong window.
        - Nếu segment tiếp theo làm vượt MAX_WINDOW_SECONDS thì flush window hiện tại.
        - Chỉ cắt khi chính segment lớn hơn MAX_WINDOW_SECONDS.
        """

        max_samples = int(MAX_WINDOW_SECONDS * SAMPLE_RATE)
        remaining = segment

        while len(remaining) > 0:
            current_samples = sum(len(x) for x in self.buffer)

            # --------------------------------------------------
            # Buffer đang rỗng nhưng segment > MAX_WINDOW_SECONDS
            # -> phải cắt thành nhiều phần
            # --------------------------------------------------
            if current_samples == 0 and len(remaining) > max_samples:
                part = remaining[:max_samples]

                self.buffer.append(part)
                self.duration += len(part) / SAMPLE_RATE

                remaining = remaining[max_samples:]

                await self.flush()
                continue

            # --------------------------------------------------
            # Thêm cả segment vẫn không vượt MAX_WINDOW_SECONDS
            # --------------------------------------------------
            if current_samples + len(remaining) <= max_samples:
                self.buffer.append(remaining)
                self.duration += len(remaining) / SAMPLE_RATE
                break

            # --------------------------------------------------
            # Nếu thêm segment sẽ vượt MAX_WINDOW_SECONDS
            # -> gửi window hiện tại trước
            # --------------------------------------------------
            if current_samples > 0:
                await self.flush()


    async def flush(self):

        if not self.buffer:
            return

        audio = np.concatenate(self.buffer)

        duration = (len(audio) / SAMPLE_RATE)

        if duration < MIN_WINDOW_SECONDS:
            return

        print(
            f"Window: enqueue ASR window={duration:.2f}s "
            f"asr_q={self.session.asr_queue.qsize()}"
        )

        await self.session.asr_queue.put(audio)

        self.buffer.clear()
        self.duration = 0.0
