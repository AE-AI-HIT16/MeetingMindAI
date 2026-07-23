import logging

import numpy as np


SAMPLE_RATE = 16000

MIN_WINDOW_SECONDS = 4
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


    async def _append_segment(self, segment: np.ndarray,):
        """
        Thêm segment vào buffer.
        Có xử lý segment >30s.
        """

        remaining = segment

        while len(remaining) > 0:
            max_samples = int(MAX_WINDOW_SECONDS * SAMPLE_RATE)

            current_samples = sum(len(x) for x in self.buffer)

            remain_capacity = (max_samples - current_samples)

            # ==================================================
            # Segment còn dư lớn hơn phần window còn lại
            # ==================================================
            if len(remaining) >= remain_capacity:

                part = remaining[:remain_capacity]

                self.buffer.append(part)

                self.duration += (len(part) / SAMPLE_RATE)

                remaining = remaining[remain_capacity:]

                # Window đủ 30s
                await self.flush()

            else:
                self.buffer.append(remaining)

                self.duration += (len(remaining) / SAMPLE_RATE)

                remaining = np.empty(0, dtype=np.float32)

                # Window đạt tối thiểu 4s
                if (self.duration >= MIN_WINDOW_SECONDS):
                    await self.flush()


    async def flush(self):

        if not self.buffer:
            return

        audio = np.concatenate(self.buffer)

        duration = (len(audio) / SAMPLE_RATE)

        if duration < MIN_WINDOW_SECONDS:
            return

        print(
            f"Window: enqueue asr window=%.2fs asr_q=%d",
            duration,
            self.session.asr_queue.qsize(),
        )

        await self.session.asr_queue.put(audio)

        self.buffer.clear()
        self.duration = 0.0
