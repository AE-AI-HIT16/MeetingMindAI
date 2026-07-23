import asyncio
import numpy as np
from meetasr.pipeline import MeetPipeline


class AudioWorker:

    def __init__(self, session, pipeline: MeetPipeline):
        self.session = session
        self.pipeline = pipeline

    async def run(self):

        while True:

            # Lấy dữ liệu từ queue
            audio = await self.session.audio_queue.get()

            # Convert PCM16 bytes thành numpy float32
            audio = np.frombuffer(
                audio,
                dtype=np.int16,
            ).astype(np.float32) / 32768.0

            # Xử lý ASR
            result = self.pipeline.transcribe(audio)

            # Gửi kết quả qua websocket
            if result is not None:
                await self.session.websocket.send_json(result.to_dict())

            self.session.audio_queue.task_done()