import asyncio
import logging


logger = logging.getLogger(
    "asr"
)



class ASRWorker:
    """
    Consumer duy nhất chạy ASR/GPU.
    Input:
        session.asr_queue

    Output:
        websocket transcript_delta
    """


    def __init__(
        self,
        session,
        asr_service,
    ):
        self.session = session
        self.asr_service = asr_service
        self.offset_ms = 0


    async def run(self):

        try:
            while True:
                audio = await (self.session.asr_queue.get())

                try:
                    duration_s = len(audio) / 16000.0
                    print(
                        f"ASR: start transcribe "
                        f"window={duration_s:.2f}s "
                        f"asr_q={self.session.asr_queue.qsize()}"
                    )

                    result = await self._transcribe(audio)

                    text_len = len(result.text)

                    print(
                        f"ASR: done text_len={text_len}"
                    )

                    await self._send_result(result)

                    await self._publish_cut_event(result)

                finally:
                    self.session.asr_queue.task_done()

        except asyncio.CancelledError:
            raise

        except Exception:
            print("ASR worker crashed")
            raise


    async def _transcribe(
        self,
        audio,
    ):
        """
        Gọi MeetPipeline.
        Nếu model sync thì chạy executor.
        """
        result = await self.asr_service.transcribe(
            audio,
            offset_ms=self.offset_ms,
        )
        self.offset_ms += result.duration_ms
        return result


    async def _send_result(
        self,
        result,
    ):
        """
        Stream transcript lên FE.
        """
        print("====================================================================================================")
        print(result)
        print("====================================================================================================")

        for segment in result.segments:
            await self.session.websocket.send_json(
                {
                    "type": "transcript_delta",
                    "segment": segment.model_dump(mode="json"),
                }
            )

    async def _publish_cut_event(
            self,
            result,
    ):
        """
        Thông báo transcript đã được xác nhận
        đến một timeline tuyệt đối.
        """

        absolute_end_ms = (
                self.session.confirmed_end_ms
                + result.duration_ms
        )

        self.session.confirmed_end_ms = (
            absolute_end_ms
        )

        await self.session.partial_cut_queue.put(
            absolute_end_ms
        )
