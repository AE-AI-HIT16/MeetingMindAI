import asyncio

SAMPLE_RATE = 16000


class PartialBufferCleaner:

    def __init__(self, session):
        self.session = session

    async def run(self):

        while True:

            cut_absolute_ms = await (
                self.session.partial_cut_queue.get()
            )

            try:

                await self._cut_buffer(
                    cut_absolute_ms
                )

            finally:

                self.session.partial_cut_queue.task_done()

    async def _cut_buffer(
        self,
        cut_absolute_ms: int,
    ):

        async with self.session.partial_buffer_lock:

            start_ms = (
                self.session.partial_buffer_start_ms
            )

            if cut_absolute_ms <= start_ms:
                return

            cut_ms = (
                cut_absolute_ms
                - start_ms
            )

            samples = int(
                cut_ms
                / 1000
                * SAMPLE_RATE
            )

            if samples <= 0:
                return

            if samples >= len(
                self.session.partial_buffer
            ):

                self.session.partial_buffer = (
                    self.session.partial_buffer[:0]
                )

            else:

                self.session.partial_buffer = (
                    self.session.partial_buffer[
                        samples:
                    ].copy()
                )

            self.session.partial_buffer_start_ms = (
                cut_absolute_ms
            )