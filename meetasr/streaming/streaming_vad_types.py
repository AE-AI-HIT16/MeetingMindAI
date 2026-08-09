"""Contracts and value objects shared by the streaming VAD components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

UtteranceReason = Literal["speech_end", "max_window", "session_stop"]


class SpeechProbabilityPredictor(Protocol):
    """Per-session probability predictor used by the endpointing state machine."""

    def predict(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StreamingVADConfig:
    """Configurable timing and hysteresis thresholds for realtime VAD."""

    sample_rate: int = 16000
    frame_ms: int = 32
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    min_speech_ms: int = 250
    min_silence_ms: int = 600
    pre_roll_ms: int = 300
    post_roll_ms: int = 100
    max_utterance_ms: int = 15000

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if not 0.0 < self.start_threshold < 1.0:
            raise ValueError("start_threshold must be between 0 and 1")
        if not 0.0 <= self.end_threshold < self.start_threshold:
            raise ValueError("end_threshold must be below start_threshold")
        if self.min_speech_ms <= 0:
            raise ValueError("min_speech_ms must be positive")
        if self.min_silence_ms < 0:
            raise ValueError("min_silence_ms must be non-negative")
        if self.pre_roll_ms < 0 or self.post_roll_ms < 0:
            raise ValueError("roll durations must be non-negative")
        if self.max_utterance_ms <= 0:
            raise ValueError("max_utterance_ms must be positive")

    @property
    def frame_samples(self) -> int:
        return max(1, round(self.sample_rate * self.frame_ms / 1000))

    def samples_for_ms(self, milliseconds: int) -> int:
        return round(self.sample_rate * milliseconds / 1000)


@dataclass(frozen=True, slots=True)
class SpeechUtterance:
    """One speech window with coordinates on the original session timeline."""

    audio: np.ndarray
    start_sample: int
    end_sample: int
    sample_rate: int
    reason: UtteranceReason

    @property
    def start_ms(self) -> int:
        return self.start_sample * 1000 // self.sample_rate

    @property
    def end_ms(self) -> int:
        return math.ceil(self.end_sample * 1000 / self.sample_rate)
