import numpy as np

from meetasr.schemas import Segment


SAMPLE_RATE = 16000


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

    # ----------------------------------------------------------

    def process(self):
        """
        Được gọi mỗi khi có chunk mới.
        """

        if self.session.pending_audio.size == 0:
            return

        if self.pipeline.vad is None:
            return

        segments = self.pipeline.vad.detect(self.session.pending_audio)

        if len(segments) < 2:
            return

        first_segment = segments[0]

        audio = self._extract_segment(
            self.session.pending_audio,
            first_segment,
        )

        self.session.ready_segments.append(audio)

        self._remove_processed_audio(
                first_segment,
                segments[1]
            )

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