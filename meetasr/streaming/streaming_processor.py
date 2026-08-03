"""Session adapter around the backend-independent streaming VAD state machine."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from meetasr.streaming.streaming_vad import StreamingVAD
from meetasr.streaming.streaming_vad_types import StreamingVADConfig
from meetasr.streaming.temp_asr_woker import PartialASRRequest

logger = logging.getLogger(__name__)


class AlwaysSpeechPredictor:
    """Fallback used only when the configured pipeline intentionally has no VAD."""

    def predict(self, frame: np.ndarray) -> float:
        del frame
        return 1.0

    def reset(self) -> None:
        return None


class StreamingProcessor:
    """Feed session audio through stateful VAD and expose completed utterances."""

    def __init__(self, session: Any, pipeline: Any) -> None:
        self.session = session
        self.pipeline = pipeline
        self.partial_interval_ms = 800
        self.partial_min_audio_ms = 700
        self.partial_max_audio_ms = 5000
        config = self._build_config()
        predictor = self._build_predictor()
        self.detector = StreamingVAD(predictor, config)

    def process(self) -> None:
        """Consume all pending audio once while retaining detector state."""
        if self.session.pending_audio.size == 0:
            return

        audio = self.session.pending_audio.copy()
        self.session.pending_audio = self.session.pending_audio[:0]
        self.session.ready_segments.extend(self.detector.process(audio))
        self.session.vad_state = (
            "speaking" if self.detector.is_speaking else "idle"
        )

    def flush(self) -> None:
        """Finalize the detector's partial frame and in-progress utterance."""
        if self.session.pending_audio.size:
            audio = self.session.pending_audio.copy()
            self.session.pending_audio = self.session.pending_audio[:0]
            self.session.ready_segments.extend(self.detector.process(audio))
        self.session.ready_segments.extend(self.detector.flush())
        self.session.vad_state = "idle"

    async def request_partial_if_due(self) -> None:
        """Request preview ASR from the event-loop thread when due."""
        if not self.detector.is_speaking:
            return
        now_ms = self.detector.processed_ms
        utterance_start_ms = self.detector.current_utterance_start_ms
        if utterance_start_ms is None:
            return
        if self.detector.current_utterance_duration_ms < self.partial_min_audio_ms:
            return
        last_ms = self.session.last_partial_request_ms
        if last_ms and now_ms - last_ms < self.partial_interval_ms:
            return
        request = PartialASRRequest(
            start_ms=max(utterance_start_ms, now_ms - self.partial_max_audio_ms),
            end_ms=now_ms,
            utterance_start_ms=utterance_start_ms,
        )
        try:
            if not self.session.temp_asr_queue.empty():
                self.session.temp_asr_queue.get_nowait()
                self.session.temp_asr_queue.task_done()
            self.session.temp_asr_queue.put_nowait(request)
            self.session.last_partial_request_ms = now_ms
        except asyncio.QueueFull:
            logger.warning("Temp ASR queue full, dropping partial request")

    def _build_predictor(self) -> Any:
        vad = self.pipeline.vad
        if vad is None:
            return AlwaysSpeechPredictor()
        factory = getattr(vad, "create_streaming_predictor", None)
        if not callable(factory):
            raise ValueError(
                f"{type(vad).__name__} does not support streaming probabilities"
            )
        return factory()

    def _build_config(self) -> StreamingVADConfig:
        vad = self.pipeline.vad
        realtime = getattr(self.pipeline, "realtime_config", {}) or {}
        vad_config = realtime.get("vad", {}) or {}
        asr_config = realtime.get("asr", {}) or {}
        self.partial_interval_ms = int(
            asr_config.get("partial_interval_ms", 800)
        )
        self.partial_min_audio_ms = int(
            asr_config.get("partial_min_audio_ms", 700)
        )
        self.partial_max_audio_ms = int(
            asr_config.get("partial_max_audio_ms", 5000)
        )
        if self.partial_interval_ms <= 0:
            raise ValueError("partial_interval_ms must be positive")
        if self.partial_min_audio_ms <= 0:
            raise ValueError("partial_min_audio_ms must be positive")
        if self.partial_max_audio_ms < self.partial_min_audio_ms:
            raise ValueError(
                "partial_max_audio_ms must be at least partial_min_audio_ms"
            )

        start_threshold = float(
            vad_config.get("start_threshold", getattr(vad, "threshold", 0.5))
        )
        configured_end = getattr(vad, "neg_threshold", None)
        default_end = (
            float(configured_end)
            if configured_end is not None
            else max(0.0, start_threshold - 0.15)
        )
        return StreamingVADConfig(
            frame_ms=int(vad_config.get("frame_ms", 32)),
            start_threshold=start_threshold,
            end_threshold=float(
                vad_config.get("end_threshold", default_end)
            ),
            min_speech_ms=int(
                vad_config.get(
                    "min_speech_ms",
                    getattr(vad, "min_speech_duration_ms", 250),
                )
            ),
            min_silence_ms=int(vad_config.get("min_silence_ms", 600)),
            pre_roll_ms=int(vad_config.get("pre_roll_ms", 300)),
            post_roll_ms=int(vad_config.get("post_roll_ms", 100)),
            max_utterance_ms=int(
                asr_config.get("max_utterance_ms", 15000)
            ),
        )
