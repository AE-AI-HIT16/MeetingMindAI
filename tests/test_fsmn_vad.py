"""Tests for FSMN-VAD wrapper behavior without loading a real model."""

from __future__ import annotations

import numpy as np
import pytest

from meetasr.models.vad.fsmn_vad import FsmnVAD, _normalize_segments


class FakeVadModel:
    """Small fake model returning deterministic VAD segments."""

    def inference(self, audio, **kwargs):
        return [
            [1200, 1600],
            [0, 100],
            [500, 900],
            [2000, 2000],
            ["bad"],
        ]


def test_normalize_segments_filters_sorts_and_converts():
    """Raw VAD output should become sorted Segment objects."""
    segments = _normalize_segments(
        [
            [1200, 1600],
            [0, 100],
            [500, 900],
            [2000, 2000],
            ["bad"],
        ],
        min_segment_ms=200,
    )

    assert [(segment.start_ms, segment.end_ms) for segment in segments] == [
        (500, 900),
        (1200, 1600),
    ]


def test_detect_empty_audio_returns_empty_without_loading_model():
    """Empty audio should return no speech segments and skip model loading."""
    vad = FsmnVAD()
    audio = np.array([], dtype=np.float32)

    assert vad.detect(audio) == []


def test_detect_rejects_non_mono_audio():
    """VAD expects mono 1-D audio."""
    vad = FsmnVAD()
    stereo_audio = np.zeros((2, 16000), dtype=np.float32)

    with pytest.raises(ValueError, match="mono 1-D"):
        vad.detect(stereo_audio)


def test_detect_rejects_non_float32_audio():
    """VAD expects float32 audio."""
    vad = FsmnVAD()
    audio = np.zeros(16000, dtype=np.float64)

    with pytest.raises(ValueError, match="float32"):
        vad.detect(audio)


def test_detect_uses_loaded_model_and_normalizes_output(monkeypatch):
    """Detect should run the model and normalize raw VAD segments."""
    vad = FsmnVAD()
    vad._model = FakeVadModel()
    monkeypatch.setattr(vad, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        vad,
        "_run_inference",
        lambda audio, max_single_segment_time: vad._model.inference(
            audio,
            max_single_segment_time=max_single_segment_time,
        ),
    )
    audio = np.zeros(16000, dtype=np.float32)

    segments = vad.detect(audio)

    assert [(segment.start_ms, segment.end_ms) for segment in segments] == [
        (500, 900),
        (1200, 1600),
    ]
