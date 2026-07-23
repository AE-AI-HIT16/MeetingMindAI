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
        pipeline,
    ):
        self.session = session
        self.pipeline = pipeline


    async def run(self):

        try:
            while True:
                audio = await (self.session.asr_queue.get())

                try:
                    duration_s = len(audio) / 16000.0
                    print(
                        f"ASR: start transcribe window=%.2fs asr_q=%d",
                        duration_s,
                        self.session.asr_queue.qsize(),
                    )

                    result = (await self._transcribe(audio))

                    print(
                        f"ASR: done text_len=%d",
                        len(result.text) if result and result.text else 0,
                    )

                    await self._send_result(result)

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
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None,
            self.pipeline.transcribe,
            audio,
        )
        return result


    async def _send_result(
        self,
        result,
    ):
        """
        Stream transcript lên FE.
        """
        print("====================================================================================================")
        print (result)
        print("====================================================================================================")

        await self.session.websocket.send_json(
            {
                "type": "transcript_delta",
                "text": result.text,
            }
        )
