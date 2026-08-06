"""Silero VAD wrapper for uploaded and offline audio."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from meetasr.runpod.models.abs_models import AbsVAD
from meetasr.runpod.register import tables
from meetasr.runpod.schemas import Segment

SAMPLE_RATE = 16000
_SUPPORTED_BACKENDS = frozenset({"onnx", "torch"})


@tables.register("model_classes", key="silero-vad")
@tables.register("model_classes", key="SileroVAD")
class SileroVAD(AbsVAD):
    """Detect speech in 16 kHz mono audio with the packaged Silero model.

    The wrapper deliberately uses the official ``silero-vad`` Python package
    instead of ``torch.hub``. Model assets therefore come from the pinned
    application dependency and no network access is required at runtime.

    This class implements the full-audio ``AbsVAD.detect`` contract only. The
    WebSocket streaming pipeline is intentionally outside its scope.
    """

    model_name = "silero-vad"

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        backend: str = "onnx",
        threshold: float = 0.5,
        neg_threshold: float | None = 0.35,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 80,
        max_speech_duration_s: float = 30.0,
        opset_version: int = 16,
        **kwargs: Any,
    ) -> None:
        """Store inference settings and lazy-load the model on first use."""
        del model_path  # The official package contains the versioned model assets.

        backend = backend.lower()
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(_SUPPORTED_BACKENDS)}, got {backend!r}"
            )
        if device != "cpu":
            raise ValueError("SileroVAD currently supports device='cpu' only")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if neg_threshold is not None and not 0.0 < neg_threshold < threshold:
            raise ValueError("neg_threshold must be positive and lower than threshold")
        if min_speech_duration_ms <= 0:
            raise ValueError("min_speech_duration_ms must be positive")
        if min_silence_duration_ms < 0:
            raise ValueError("min_silence_duration_ms must be non-negative")
        if speech_pad_ms < 0:
            raise ValueError("speech_pad_ms must be non-negative")
        if max_speech_duration_s <= 0:
            raise ValueError("max_speech_duration_s must be positive")
        if opset_version not in (15, 16):
            raise ValueError("opset_version must be 15 or 16")

        ignored_options = sorted(
            key for key in kwargs if key not in {"hub", "model"}
        )
        if ignored_options:
            logging.warning(
                "Ignoring unsupported Silero VAD config option(s): %s",
                ", ".join(ignored_options),
            )

        self.device = device
        self.backend = backend
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_speech_duration_s = float(max_speech_duration_s)
        self.opset_version = opset_version

        # MeetPipeline reads these generic attributes after detect().
        self.min_segment_ms = min_speech_duration_ms
        self.max_segment_ms = int(round(max_speech_duration_s * 1000))

        self._model = None
        self._get_speech_timestamps = None
        self._load_lock = threading.Lock()
        # get_speech_timestamps resets and mutates model state. Serialize calls
        # when multiple upload jobs share one pipeline instance.
        self._inference_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Lazy-load the official packaged Silero model."""
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return
            try:
                from silero_vad import get_speech_timestamps, load_silero_vad

                self._model = load_silero_vad(
                    onnx=self.backend == "onnx",
                    opset_version=self.opset_version,
                )
                self._get_speech_timestamps = get_speech_timestamps
                logging.info(
                    "SileroVAD loaded with %s backend on cpu",
                    self.backend,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load SileroVAD. Install the pinned 'silero-vad' "
                    f"dependency and its runtime: {exc}"
                ) from exc

    def detect(self, audio: np.ndarray, **kwargs: Any) -> list[Segment]:
        """Return sorted speech segments on the original audio timeline."""
        self._validate_audio(audio)
        if audio.size == 0:
            return []

        threshold = float(kwargs.get("threshold", self.threshold))
        neg_threshold = kwargs.get("neg_threshold", self.neg_threshold)
        if neg_threshold is not None:
            neg_threshold = float(neg_threshold)
        min_speech_ms = int(
            kwargs.get("min_speech_duration_ms", self.min_speech_duration_ms)
        )
        min_silence_ms = int(
            kwargs.get("min_silence_duration_ms", self.min_silence_duration_ms)
        )
        speech_pad_ms = int(kwargs.get("speech_pad_ms", self.speech_pad_ms))
        max_speech_s = float(
            kwargs.get("max_speech_duration_s", self.max_speech_duration_s)
        )

        self._ensure_loaded()
        import torch

        audio_tensor = torch.from_numpy(np.ascontiguousarray(audio))
        with self._inference_lock:
            raw_timestamps = self._get_speech_timestamps(
                audio_tensor,
                self._model,
                threshold=threshold,
                sampling_rate=SAMPLE_RATE,
                min_speech_duration_ms=min_speech_ms,
                max_speech_duration_s=max_speech_s,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=False,
                neg_threshold=neg_threshold,
            )

        return _normalize_timestamps(
            raw_timestamps,
            audio_samples=audio.size,
            min_segment_ms=min_speech_ms,
        )

    @staticmethod
    def _validate_audio(audio: np.ndarray) -> None:
        if not isinstance(audio, np.ndarray):
            raise ValueError(
                f"audio must be a numpy.ndarray, got {type(audio).__name__}"
            )
        if audio.ndim != 1:
            raise ValueError(f"audio must be mono 1-D array, got shape {audio.shape}")
        if audio.dtype != np.float32:
            raise ValueError(f"audio must have dtype np.float32, got {audio.dtype}")


def _normalize_timestamps(
    timestamps: Iterable[Mapping[str, Any]] | None,
    *,
    audio_samples: int,
    min_segment_ms: int,
) -> list[Segment]:
    """Convert Silero sample offsets to clipped millisecond segments."""
    audio_end_ms = math.ceil(audio_samples * 1000 / SAMPLE_RATE)
    segments: list[Segment] = []

    for timestamp in timestamps or []:
        if not isinstance(timestamp, Mapping):
            continue
        try:
            start_sample = max(0, int(timestamp["start"]))
            end_sample = min(audio_samples, int(timestamp["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end_sample <= start_sample:
            continue

        # Floor starts and ceil ends so conversion never cuts detected speech.
        start_ms = start_sample * 1000 // SAMPLE_RATE
        end_ms = min(
            audio_end_ms,
            math.ceil(end_sample * 1000 / SAMPLE_RATE),
        )
        if end_ms - start_ms < min_segment_ms:
            continue
        segments.append(Segment(start_ms=start_ms, end_ms=end_ms))

    return sorted(segments, key=lambda segment: (segment.start_ms, segment.end_ms))
