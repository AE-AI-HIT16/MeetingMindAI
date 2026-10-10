import asyncio
import logging

from meetasr.api.schemas_phase2 import TranscriptSegmentPayload


logger = logging.getLogger("temp_asr")


class TempASRWorker:
    """
    Partial ASR worker.

    Input:
        session.temp_asr_queue

    Output:
        websocket transcript_partial
    """

    def __init__(
        self,
        session,
        asr_service,
    ):
        self.session = session
        self.asr_service = asr_service


    async def run(self):

        try:

            while True:
                await self.session.temp_asr_queue.get()

                async with self.session.partial_buffer_lock:

                    audio = self.session.partial_buffer.copy()

                    start_ms = (
                        self.session.partial_buffer_start_ms
                    )

                duration_ms = int(
                    len(audio)
                    / 16000
                    * 1000
                )

                end_ms = start_ms + duration_ms

                if audio.size == 0:
                    self.session.temp_asr_queue.task_done()
                    continue

                try:
                    results = await self.asr_service.recognize_partial(audio)

                    for result in results:

                        text = result.get(
                            "text",
                            "",
                        ).strip()

                        if not text:
                            continue

                        segment = TranscriptSegmentPayload(
                            start_ms=start_ms,
                            end_ms=end_ms,
                            speaker=None,
                            text=text,
                        )

                        await self.session.websocket.send_json(
                            {
                                "type": "transcript_partial",
                                "segment": segment.model_dump(
                                    mode="json"
                                ),
                            }
                        )


                finally:

                    self.session.temp_asr_queue.task_done()


        except asyncio.CancelledError:
            raise


        except Exception:

            logger.exception(
                "Temp ASR worker crashed"
            )

            raise