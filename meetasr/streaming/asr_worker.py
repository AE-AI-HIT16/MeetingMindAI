import logging
import asyncio
import traceback


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
        print("ASR worker started")

        try:
            while True:
                print("Waiting for audio...")

                audio = await self.session.asr_queue.get()

                print(f"Got audio, queue={self.session.asr_queue.qsize()}")

                try:
                    duration_s = len(audio) / 16000.0
                    print(
                        f"ASR: start transcribe "
                        f"window={duration_s:.2f}s "
                        f"asr_q={self.session.asr_queue.qsize()}"
                    )

                    try:
                        result = await self._transcribe(audio)
                        print("DEBUG: _transcribe() OK")
                    except Exception:
                        print("DEBUG: _transcribe() FAILED")
                        traceback.print_exc()
                        raise

                    print(f"ASR: done text_len={len(result.text)}")

                    try:
                        await self._send_result(result)
                        print("DEBUG: _send_result() OK")
                    except Exception:
                        print("DEBUG: _send_result() FAILED")
                        traceback.print_exc()
                        raise

                    try:
                        await self._publish_cut_event(result)
                        print("DEBUG: _publish_cut_event() OK")
                    except Exception:
                        print("DEBUG: _publish_cut_event() FAILED")
                        traceback.print_exc()
                        raise

                finally:
                    try:
                        self.session.asr_queue.task_done()
                        print("DEBUG: task_done() OK")
                    except Exception:
                        print("DEBUG: task_done() FAILED")
                        traceback.print_exc()
                        raise

        except asyncio.CancelledError:
            print("ASR worker cancelled")
            raise

        except Exception as e:
            print("=" * 80)
            print(f"ASR worker crashed: {type(e).__name__}: {e}")
            traceback.print_exc()
            print("=" * 80)
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
