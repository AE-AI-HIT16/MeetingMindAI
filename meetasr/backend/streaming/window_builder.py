import logging

import numpy as np


SAMPLE_RATE = 16000

MIN_WINDOW_SECONDS = 1
MAX_WINDOW_SECONDS = 30

# Tìm điểm cắt window trong ngần này giây cuối, theo khung 50 ms
QUIET_SEARCH_SECONDS = 3.0
QUIET_FRAME_SAMPLES = 800

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

        max_seconds = getattr(self.session, "max_window_seconds", MAX_WINDOW_SECONDS)
        max_samples = int(max_seconds * SAMPLE_RATE)
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
            # -> gửi window hiện tại trước, cắt tại chỗ yên lặng nhất
            #    (cắt cứng giữa từ làm ASR ra chữ linh tinh ở 2 đầu window)
            # --------------------------------------------------
            if current_samples > 0:
                await self._flush_at_quiet_point()


    async def _flush_at_quiet_point(self):
        """Gửi phần đầu buffer tới điểm năng lượng thấp nhất trong
        QUIET_SEARCH_SECONDS cuối; phần còn lại giữ cho window sau."""

        audio = np.concatenate(self.buffer)
        cut = quietest_cut(audio)  # luôn >= MIN_WINDOW_SECONDS → flush() gửi được

        self.buffer = [audio[:cut]]
        await self.flush()

        rest = audio[cut:]
        if len(rest):
            self.buffer = [rest]
            self.duration = len(rest) / SAMPLE_RATE

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


def quietest_cut(audio: np.ndarray) -> int:
    """Sample index of the lowest-energy 50 ms frame within the last
    QUIET_SEARCH_SECONDS (never earlier than MIN_WINDOW_SECONDS)."""

    search_start = max(
        int(MIN_WINDOW_SECONDS * SAMPLE_RATE),
        len(audio) - int(QUIET_SEARCH_SECONDS * SAMPLE_RATE),
    )
    n_frames = (len(audio) - search_start) // QUIET_FRAME_SAMPLES
    if n_frames <= 0:
        return len(audio)

    frames = audio[search_start:search_start + n_frames * QUIET_FRAME_SAMPLES]
    energy = np.square(frames.reshape(n_frames, QUIET_FRAME_SAMPLES)).mean(axis=1)
    best = int(np.argmin(energy))
    # cut in the middle of the quietest frame
    return search_start + best * QUIET_FRAME_SAMPLES + QUIET_FRAME_SAMPLES // 2
