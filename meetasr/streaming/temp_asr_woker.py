import asyncio
import logging
import time
from dataclasses import dataclass, field

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
    utterance_start_ms: int | None = None
    requested_at: float = field(
        default_factory=time.perf_counter,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("partial request must have a positive time range")
        if (
            self.utterance_start_ms is not None
            and not 0 <= self.utterance_start_ms <= self.start_ms
        ):
            raise ValueError("utterance start must not follow partial window start")


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
        worker_started_at = time.perf_counter()
        snapshot = await self._snapshot(request)
        if snapshot is None:
            return
        audio, start_ms, end_ms = snapshot
        language = getattr(self.pipeline, "transcription_language", "auto")
        inference_started_at = time.perf_counter()
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
        inference_ms = (time.perf_counter() - inference_started_at) * 1000
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
            self._record_emission(
                request,
                worker_started_at=worker_started_at,
                inference_ms=inference_ms,
            )

    def _record_emission(
        self,
        request: PartialASRRequest,
        *,
        worker_started_at: float,
        inference_ms: float,
    ) -> None:
        """Log latency for an emitted preview without changing its contract."""
        emitted_at = time.perf_counter()
        first_audio_at = getattr(
            self.session,
            "first_audio_received_at",
            None,
        )
        previous_emission = getattr(
            self.session,
            "last_partial_utterance_emitted_at",
            None,
        )
        utterance_start_ms = (
            request.utterance_start_ms
            if request.utterance_start_ms is not None
            else request.start_ms
        )
        if (
            getattr(self.session, "partial_utterance_start_ms", None)
            != utterance_start_ms
        ):
            self.session.partial_utterance_start_ms = utterance_start_ms
            self.session.last_partial_utterance_emitted_at = None
            self.session.partial_utterance_emitted_count = 0
            previous_emission = None
        first_audio_latency_ms = (
            (emitted_at - first_audio_at) * 1000
            if first_audio_at is not None
            else -1.0
        )
        update_interval_ms = (
            (emitted_at - previous_emission) * 1000
            if previous_emission is not None
            else -1.0
        )
        if getattr(self.session, "first_partial_emitted_at", None) is None:
            self.session.first_partial_emitted_at = emitted_at
        self.session.last_partial_emitted_at = emitted_at
        self.session.last_partial_utterance_emitted_at = emitted_at
        self.session.partial_emitted_count = (
            getattr(self.session, "partial_emitted_count", 0) + 1
        )
        self.session.partial_utterance_emitted_count = (
            getattr(self.session, "partial_utterance_emitted_count", 0) + 1
        )
        utterance_to_partial_ms = (
            (emitted_at - first_audio_at) * 1000 - utterance_start_ms
            if first_audio_at is not None
            else -1.0
        )
        request_to_emit_ms = (emitted_at - request.requested_at) * 1000
        audio_end_to_emit_ms = (
            first_audio_latency_ms - request.end_ms
            if first_audio_at is not None
            else -1.0
        )
        audio_end_to_request_ms = (
            audio_end_to_emit_ms - request_to_emit_ms
            if first_audio_at is not None
            else -1.0
        )
        logger.info(
            "Partial latency job=%s count=%d utterance_count=%d "
            "utterance_start_ms=%d "
            "window=%d-%d worker_wait_ms=%.1f inference_ms=%.1f "
            "request_to_emit_ms=%.1f first_audio_to_partial_ms=%.1f "
            "utterance_to_partial_ms=%.1f audio_end_to_request_ms=%.1f "
            "audio_end_to_emit_ms=%.1f update_interval_ms=%.1f",
            getattr(self.session, "job_id", "unknown"),
            self.session.partial_emitted_count,
            self.session.partial_utterance_emitted_count,
            utterance_start_ms,
            request.start_ms,
            request.end_ms,
            (worker_started_at - request.requested_at) * 1000,
            inference_ms,
            request_to_emit_ms,
            first_audio_latency_ms,
            utterance_to_partial_ms,
            audio_end_to_request_ms,
            audio_end_to_emit_ms,
            update_interval_ms,
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
