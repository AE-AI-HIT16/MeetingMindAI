import time
import numpy as np

from meetasr.pipeline import MeetPipeline


class StreamingProcessor:
    """
    Điều phối luồng realtime.

    Không sửa MeetPipeline.
    Không thay đổi logic AI.

    Chỉ quản lý:

    - audio buffer
    - trigger infer
    - gọi pipeline
    - lưu transcript gần nhất
    """

    def __init__(
        self,
        session,
        pipeline: MeetPipeline,
        *,
        infer_interval: float = 0.5,
    ):

        self.session = session
        self.pipeline = pipeline

        # Bao lâu chạy pipeline một lần
        self.infer_interval = infer_interval

        self.last_infer_time = 0.0

    async def feed(self, chunk: np.ndarray):

        # ----------------------------------------------------------
        # 1. Lưu chunk vào session
        # ----------------------------------------------------------

        self.session.audio_buffer.append(chunk)

        # ----------------------------------------------------------
        # 2. Kiểm tra thời gian
        # ----------------------------------------------------------

        now = time.monotonic()

        if now - self.last_infer_time < self.infer_interval:
            return None

        self.last_infer_time = now

        # ----------------------------------------------------------
        # 3. Ghép toàn bộ audio
        # ----------------------------------------------------------

        if not self.session.audio_buffer:
            return None

        audio = np.concatenate(
            list(self.session.audio_buffer),
            axis=0,
        )

        # ----------------------------------------------------------
        # 4. Gọi pipeline cũ
        # ----------------------------------------------------------

        result = self.pipeline.transcribe(audio)

        # ----------------------------------------------------------
        # 5. Cập nhật state
        # ----------------------------------------------------------

        self.session.last_result = result

        if result is not None:
            self.session.partial_transcript = result.text

        # ----------------------------------------------------------
        # 6. Trả kết quả
        # ----------------------------------------------------------

        return result