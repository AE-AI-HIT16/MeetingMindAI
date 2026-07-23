import numpy as np

from meetasr.pipeline import MeetPipeline
from meetasr.streaming.streaming_processor import StreamingProcessor


class AudioWorker:

    def __init__(
            self,
            session,
            pipeline,
            window_builder,
    ):
        self.session = session
        self.pipeline = pipeline
        self.window_builder = window_builder

        self.processor = StreamingProcessor(
            session=session,
            pipeline=pipeline,
        )

    async def run(self):

        while True:

            # Lấy chunk PCM16 từ queue
            chunk = await self.session.audio_queue.get()

            # PCM16 -> float32 [-1, 1]
            audio = np.frombuffer(
                chunk,
                dtype=np.int16,
            ).astype(np.float32) / 32768.0

            # Lưu chunk vào ring buffer
            self.session.audio_buffer.append(audio)

            # Ghép vào pending buffer
            if self.session.pending_audio.size == 0:
                self.session.pending_audio = audio
            else:
                self.session.pending_audio = np.concatenate(
                    (
                        self.session.pending_audio,
                        audio,
                    )
                )

            # Chạy Streaming VAD
            self.processor.process()

            await self.window_builder.process()

            self.session.audio_queue.task_done()