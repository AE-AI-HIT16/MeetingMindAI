import asyncio
import logging

from meetasr.services.realtime_transcript_service import (
    RealtimeTranscriptService,
)
from meetasr.streaming.window_builder import ASRWindow

logger = logging.getLogger(
    "asr"
)



class ASRWorker:
    """
    Consumer duy nhất chạy ASR/GPU.
    Input:
        session.asr_queue

    Output:
        persisted transcript_delta -> recording WebSocket + Job EventBus
    """


    def __init__(
        self,
        session,
        asr_service,
        transcript_service=None,
    ):
        self.session = session
        self.asr_service = asr_service
        self.transcript_service = (
            transcript_service or RealtimeTranscriptService()
        )
    async def run(self):
        logger.info("Confirmed ASR worker started")
        while True:
            window = await self.session.asr_queue.get()
            try:
                result = await self._transcribe(window)
                persisted = await self._send_result(result)
                if not persisted:
                    raise RuntimeError("Confirmed ASR returned no usable text")
                coverage = getattr(self.session, "coverage", None)
                if coverage is not None:
                    coverage.record_confirmed(window.start_ms, window.end_ms)
                await self._publish_cut_event(window)
            except asyncio.CancelledError:
                raise
            except Exception:
                coverage = getattr(self.session, "coverage", None)
                if coverage is not None:
                    coverage.record_asr_failure()
                logger.exception(
                    "Confirmed ASR failed range=%s-%sms; finalizer will fallback",
                    window.start_ms,
                    window.end_ms,
                )
            finally:
                self.session.asr_queue.task_done()


    async def _transcribe(
        self,
        window: ASRWindow,
    ):
        """
        Gọi MeetPipeline.
        Nếu model sync thì chạy executor.
        """
        result = await self.asr_service.transcribe(
            window.audio,
            offset_ms=window.start_ms,
        )
        return result


    async def _send_result(
        self,
        result,
    ):
        """
        Persist and publish confirmed transcript through the canonical service.
        """
        persisted = []
        for segment in result.segments:
            persisted.append(
                await self.transcript_service.persist_and_publish(
                    self.session.job_id,
                    segment,
                    self.session.websocket,
                )
            )
        return persisted

    async def _publish_cut_event(
            self,
            window: ASRWindow,
    ):
        """
        Thông báo transcript đã được xác nhận
        đến một timeline tuyệt đối.
        """

        absolute_end_ms = max(
            self.session.confirmed_end_ms,
            window.end_ms,
        )
        self.session.confirmed_end_ms = absolute_end_ms

        await self.session.partial_cut_queue.put(
            absolute_end_ms
        )
