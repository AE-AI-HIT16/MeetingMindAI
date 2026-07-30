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
        pipeline,
    ):
        self.session = session
        self.pipeline = pipeline


    async def run(self):

        try:

            while True:
                audio = await self.session.temp_asr_queue.get()
                try:
                    results = await asyncio.to_thread(
                        self.pipeline.asr.recognize,
                        [audio],
                    )

                    for result in results:

                        text = result.get(
                            "text",
                            "",
                        ).strip()

                        if not text:
                            continue

                        segment = TranscriptSegmentPayload(
                            start_ms=0,
                            end_ms=0,
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