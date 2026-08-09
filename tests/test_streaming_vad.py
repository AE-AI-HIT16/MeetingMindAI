"""State-machine tests for streaming VAD without loading model weights."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from meetasr.streaming.streaming_vad import (
    StreamingVAD,
    StreamingVADConfig,
)


class FakePredictor:
    def __init__(self, probabilities: Iterable[float]) -> None:
        self.probabilities = iter(probabilities)
        self.frames: list[np.ndarray] = []

    def predict(self, frame: np.ndarray) -> float:
        self.frames.append(frame.copy())
        return next(self.probabilities)

    def reset(self) -> None:
        return None


def make_config(**overrides: object) -> StreamingVADConfig:
    values = {
        "sample_rate": 1000,
        "frame_ms": 100,
        "start_threshold": 0.5,
        "end_threshold": 0.35,
        "min_speech_ms": 200,
        "min_silence_ms": 300,
        "pre_roll_ms": 100,
        "post_roll_ms": 100,
        "max_utterance_ms": 2000,
    }
    values.update(overrides)
    return StreamingVADConfig(**values)


def test_silence_never_creates_an_asr_utterance() -> None:
    detector = StreamingVAD(
        FakePredictor([0.1] * 10),
        make_config(),
    )

    assert detector.process(np.zeros(1000, dtype=np.float32)) == []
    assert detector.flush() == []


def test_speech_end_includes_pre_roll_and_limited_post_roll() -> None:
    detector = StreamingVAD(
        FakePredictor([0.1, 0.8, 0.8, 0.9, 0.9, 0.1, 0.1, 0.1]),
        make_config(),
    )
    audio = np.arange(800, dtype=np.float32)

    utterances = detector.process(audio)

    assert len(utterances) == 1
    utterance = utterances[0]
    assert utterance.reason == "speech_end"
    assert (utterance.start_ms, utterance.end_ms) == (0, 600)
    np.testing.assert_array_equal(utterance.audio, audio[:600])


def test_max_window_emits_without_waiting_for_speech_to_end() -> None:
    detector = StreamingVAD(
        FakePredictor([0.9] * 8),
        make_config(
            min_speech_ms=100,
            pre_roll_ms=0,
            max_utterance_ms=500,
        ),
    )
    audio = np.arange(800, dtype=np.float32)

    utterances = detector.process(audio)
    utterances.extend(detector.flush())

    assert [item.reason for item in utterances] == [
        "max_window",
        "session_stop",
    ]
    assert [
        (item.start_ms, item.end_ms) for item in utterances
    ] == [(0, 500), (500, 800)]
    np.testing.assert_array_equal(
        np.concatenate([item.audio for item in utterances]),
        audio,
    )


def test_session_stop_keeps_a_short_unconfirmed_word() -> None:
    detector = StreamingVAD(
        FakePredictor([0.1, 0.9]),
        make_config(),
    )
    audio = np.arange(200, dtype=np.float32)

    assert detector.process(audio) == []
    utterances = detector.flush()

    assert len(utterances) == 1
    assert utterances[0].reason == "session_stop"
    assert (utterances[0].start_ms, utterances[0].end_ms) == (0, 200)
    np.testing.assert_array_equal(utterances[0].audio, audio)


def test_absolute_timeline_survives_silence_between_utterances() -> None:
    probabilities = [
        0.9,
        0.9,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.9,
        0.9,
        0.1,
        0.1,
        0.1,
    ]
    detector = StreamingVAD(
        FakePredictor(probabilities),
        make_config(
            min_speech_ms=100,
            pre_roll_ms=0,
            post_roll_ms=0,
        ),
    )

    utterances = detector.process(np.zeros(1200, dtype=np.float32))

    assert [
        (item.start_ms, item.end_ms) for item in utterances
    ] == [(0, 200), (700, 900)]


def test_arbitrary_network_chunks_are_buffered_into_exact_vad_frames() -> None:
    predictor = FakePredictor([0.1, 0.1, 0.1])
    detector = StreamingVAD(predictor, make_config())

    detector.process(np.zeros(150, dtype=np.float32))
    detector.process(np.zeros(150, dtype=np.float32))

    assert [len(frame) for frame in predictor.frames] == [100, 100, 100]
