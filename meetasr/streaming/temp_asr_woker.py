import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from meetasr.api.schemas_phase2 import TranscriptSegmentPayload
from meetasr.services.inference_coordinator import (
    InferenceCoordinator,
    InferenceKind,
    PartialInferenceDropped,
)

logger = logging.getLogger("temp_asr")
SAMPLE_RATE = 16000


@dataclass(frozen=True, slots=True)
class PartialASRRequest:
    """Absolute audio range requested for one disposable preview."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("partial request must have a positive time range")


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
        coordinator: InferenceCoordinator | None = None,
    ):
        self.session = session
        self.pipeline = pipeline
        self.coordinator = coordinator


    async def run(self):

        try:

            while True:
                request = await self.session.temp_asr_queue.get()

                try:
                    await self.process_request(request)
                finally:
                    self.session.temp_asr_queue.task_done()


        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception("Temp ASR worker crashed")
            raise

    async def process_request(self, request: PartialASRRequest) -> None:
        """Transcribe one bounded snapshot unless it was already confirmed."""
        snapshot = await self._snapshot(request)
        if snapshot is None:
            return
        audio, start_ms, end_ms = snapshot
        language = getattr(self.pipeline, "transcription_language", "auto")
        try:
            if self.coordinator is not None:
                results = await self.coordinator.submit(
                    InferenceKind.PARTIAL,
                    self.pipeline.asr.recognize,
                    [audio],
                    partial_key=str(
                        getattr(self.session, "job_id", id(self.session))
                    ),
                    language=language,
                )
            else:
                results = await asyncio.to_thread(
                    self.pipeline.asr.recognize,
                    [audio],
                    language=language,
                )
        except PartialInferenceDropped:
            logger.debug("Partial ASR request replaced by a newer snapshot")
            return
        for result in results:
            text = result.get("text", "").strip()
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
                    "segment": segment.model_dump(mode="json"),
                }
            )

    async def _snapshot(
        self,
        request: PartialASRRequest,
    ) -> tuple[np.ndarray, int, int] | None:
        async with self.session.partial_buffer_lock:
            buffer_start_ms = self.session.partial_buffer_start_ms
            buffer_end_ms = buffer_start_ms + (
                len(self.session.partial_buffer) * 1000 // SAMPLE_RATE
            )
            start_ms = max(request.start_ms, buffer_start_ms)
            end_ms = min(request.end_ms, buffer_end_ms)
            if end_ms <= start_ms:
                return None
            start_sample = (start_ms - buffer_start_ms) * SAMPLE_RATE // 1000
            end_sample = (end_ms - buffer_start_ms) * SAMPLE_RATE // 1000
            audio = self.session.partial_buffer[start_sample:end_sample].copy()
        if audio.size == 0:
            return None
        return audio, start_ms, end_ms
