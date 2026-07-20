import asyncio
import numpy as np
from meetasr.pipeline import MeetPipeline


# Nơi khởi chạy luồng logic
# Sử dụng luồng hoạt động của pipeline.py luôn
class AudioWorker:

    def __init__(self, session, pipeline : MeetPipeline):

        self.session = session
        self.pipeline = pipeline

    async def run(self):

        loop = asyncio.get_running_loop()

        while True:

            # Lấy dữ liệu từ trong queue
            audio = await self.session.audio_queue.get()

            # Convert PCM16 bytes từ microphone thành numpy float32
            audio = np.frombuffer(
                audio,
                dtype=np.int16,
            ).astype(np.float32) / 32768.0

            # Đưa gữ liệu cho processer xử lý và nhận kết quả
            # (chỗ này cần gọi nhiều và thực hiện tuần tự nếu xử lý cần nhiều bước logic
            # Gọi các model vào xử lý ở đây
            # process có kiểu dữ liệu là AudioProcessor
            result = self.pipeline.transcribe(audio)

            # Gửi kết quả cho websocket
            if result is not None:
                await self.session.websocket.send_json(result)

            self.session.audio_queue.task_done()