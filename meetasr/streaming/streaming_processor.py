import logging

import numpy as np

from meetasr.schemas import Segment
import asyncio


SAMPLE_RATE = 16000
SECOND = 1

logger = logging.getLogger(__name__)


class StreamingProcessor:
    """
    Chỉ xử lý Streaming VAD.

    Chức năng:

    - Đọc pending_audio trong session.
    - Chạy VAD trên toàn bộ pending_audio.
    - Khi VAD đã tách được >=2 segment:
        + Segment đầu tiên được coi là hoàn chỉnh.
        + Cắt audio của segment đầu.
        + Đưa vào ready_segments.
        + Xóa khỏi pending_audio.

    Không thực hiện ASR.
    Không build window.
    """

    def __init__(self, session, pipeline):
        self.session = session
        self.pipeline = pipeline

        # DEBUG: đếm số lần process được gọi
        self._debug_chunk_count = 0

    # ----------------------------------------------------------

    def process(self):
        """
        Được gọi mỗi khi có chunk mới.
        """

        if self.session.pending_audio.size == 0:
            return

        if self.pipeline.vad is None:
            return

        if self.session.vad_state is None:
            self.session.vad_state = {}

        segments = split_fixed_segments(
            self.session.pending_audio,
            segment_ms=SECOND * 1000,
        )

        if len(segments) < 2:
            return

        first_segment = segments[0]

        audio = self._extract_segment(
            self.session.pending_audio,
            first_segment,
        )

        self.session.ready_segments.append(audio)

        # Đẩy dữ liệu vào queue, để có kết quả sớm cho fe
        try:
            self.session.temp_asr_queue.put_nowait(audio)

        except asyncio.QueueFull:
            logger.warning(
                "Temp ASR queue full, dropping segment"
            )

        self._remove_processed_audio(
                first_segment,
                segments[1]
            )

        # Timeline of pending_audio shifted — invalidate streaming cache
        self.session.vad_state = None

    # ----------------------------------------------------------

    def _extract_segment(
        self,
        audio: np.ndarray,
        segment: Segment,
    ) -> np.ndarray:

        start = int(segment.start_ms / 1000 * SAMPLE_RATE)
        end = int(segment.end_ms / 1000 * SAMPLE_RATE)

        return audio[start:end].copy()

    # ----------------------------------------------------------

    def _remove_processed_audio(
            self,
            processed_segment: Segment,
            next_segment: Segment | None = None,
    ):
        """
        Remove speech đã xử lý + silence phía sau.
        """

        if next_segment is None:

            end = int(
                processed_segment.end_ms
                / 1000
                * SAMPLE_RATE
            )

        else:

            """
            Remove processed speech
            and silence before next speech.
            """
            end = int(
                next_segment.start_ms
                / 1000
                * SAMPLE_RATE
            )

        self.session.pending_audio = (
            self.session.pending_audio[end:].copy()
        )

    # ----------------------------------------------------------

# DEBUG

from meetasr.schemas import Segment

SAMPLE_RATE = 16000


def split_fixed_segments(
    audio: np.ndarray,
    segment_ms: int = 3000,
) -> list[Segment]:
    """
    Chia audio thành các segment cố định.

    Args:
        audio:
            Float32 mono audio 16kHz.
        segment_ms:
            Độ dài mỗi segment (ms), mặc định 3 giây.

    Returns:
        List[Segment] với start_ms/end_ms.
    """

    total_ms = int(
        len(audio) / SAMPLE_RATE * 1000
    )

    segments = []

    start_ms = 0

    while start_ms < total_ms:
        end_ms = min(
            start_ms + segment_ms,
            total_ms,
        )

        segments.append(
            Segment(
                start_ms,
                end_ms,
            )
        )

        start_ms = end_ms

    return segments