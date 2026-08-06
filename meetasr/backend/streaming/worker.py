import asyncio

import numpy as np

from meetasr.backend.streaming.streaming_processor import StreamingProcessor


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

        loop = asyncio.get_running_loop()

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

            # Ghép vào partial buffer
            async with self.session.partial_buffer_lock:

                if self.session.partial_buffer.size == 0:
                    self.session.partial_buffer = audio

                else:
                    self.session.partial_buffer = np.concatenate(
                        (
                            self.session.partial_buffer,
                            audio,
                        )
                    )

            # Chạy Streaming VAD off event loop (CPU-bound)
            await loop.run_in_executor(None, self.processor.process)

            await self.window_builder.process()

            self.session.audio_queue.task_done()
