import asyncio
import logging
import numpy as np
from meetasr.pipeline import MeetPipeline
import wave
import os


logger = logging.getLogger("meetasr.streaming.worker")


# Nơi khởi chạy luồng logic
# Sử dụng luồng hoạt động của pipeline.py luôn
class AudioWorker:

    def __init__(self, session, pipeline : MeetPipeline):

        self.session = session
        self.pipeline = pipeline
        self.chunk_index = 0

        logger.info("=" * 80)
        logger.info("AudioWorker initialized")
        logger.info("Pipeline type        : %s", type(self.pipeline).__name__)

    async def run(self):

        logger.info("=" * 80)
        logger.info("AudioWorker started")

        loop = asyncio.get_running_loop()

        logger.info("Event loop           : %s", type(loop).__name__)

        while True:

            logger.info("-" * 80)
            logger.info("Waiting for audio chunk...")
            logger.info(
                "Current queue size   : %d",
                self.session.audio_queue.qsize(),
            )

            # Lấy dữ liệu từ trong queue
            audio = await self.session.audio_queue.get()

            logger.info("Chunk received")
            logger.info("Chunk python type    : %s", type(audio).__name__)
            logger.info("Chunk size           : %d bytes", len(audio))

            # Convert PCM16 bytes từ microphone thành numpy float32
            audio = np.frombuffer(
                audio,
                dtype=np.int16,
            ).astype(np.float32) / 32768.0

            os.makedirs("debug_audio", exist_ok=True)

            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

            filename = f"debug_audio/chunk_{self.chunk_index:05d}.wav"

            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16.tobytes())

            logger.info(
                "Saved %s",
                os.path.abspath(filename),
            )

            self.chunk_index += 1

            logger.info("Converted to numpy")
            logger.info("Numpy dtype          : %s", audio.dtype)
            logger.info("Numpy shape          : %s", audio.shape)
            logger.info("Number of samples    : %d", audio.size)

            if audio.size > 0:
                logger.info(
                    "Amplitude range      : [%.5f, %.5f]",
                    float(audio.min()),
                    float(audio.max()),
                )
                logger.info(
                    "Mean amplitude       : %.5f",
                    float(audio.mean()),
                )

            # Đưa gữ liệu cho processer xử lý và nhận kết quả
            # (chỗ này cần gọi nhiều và thực hiện tuần tự nếu xử lý cần nhiều bước logic
            # Gọi các model vào xử lý ở đây
            # process có kiểu dữ liệu là AudioProcessor
            logger.info("Calling pipeline.transcribe()")

            try:
                result = self.pipeline.transcribe(audio)

                logger.info("Pipeline finished")

                if result is None:
                    logger.info("Pipeline result      : None")
                else:
                    logger.info(
                        "Pipeline result type : %s",
                        type(result).__name__,
                    )

            except Exception:
                logger.exception("Pipeline raised exception")
                raise

            # Gửi kết quả cho websocket
            if result is not None:

                logger.info("Sending websocket response")

                await self.session.websocket.send_json(result.to_dict())

                logger.info("Response sent")

            self.session.audio_queue.task_done()

            logger.info("Queue task done")
            logger.info(
                "Remaining queue size : %d",
                self.session.audio_queue.qsize(),
            )