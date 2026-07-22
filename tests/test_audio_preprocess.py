"""Tests for lightweight audio preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from meetasr.utils.audio_preprocess import rms_normalize


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def test_rms_normalize_raises_quiet_audio_to_target():
    audio = np.full(16000, 0.02, dtype=np.float32)

    normalized = rms_normalize(audio, target_rms=0.05)

    assert _rms(normalized) == pytest.approx(0.05)
    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous
    assert np.all(audio == np.float32(0.02))


def test_rms_normalize_limits_amplification():
    audio = np.full(16000, 0.01, dtype=np.float32)

    normalized = rms_normalize(audio, target_rms=0.05, max_gain=3.0)

    assert _rms(normalized) == pytest.approx(0.03)


def test_rms_normalize_does_not_amplify_past_peak_limit():
    audio = np.full(100, 0.01, dtype=np.float32)
    audio[0] = 0.8

    normalized = rms_normalize(audio, target_rms=0.2, peak_limit=0.9)

    assert float(np.max(np.abs(normalized))) <= 0.9
    assert normalized[0] == pytest.approx(0.9)


def test_rms_normalize_keeps_loud_audio_unchanged():
    audio = np.full(16000, 0.1, dtype=np.float32)

    normalized = rms_normalize(audio, target_rms=0.05)

    np.testing.assert_array_equal(normalized, audio)
    assert normalized is not audio


def test_rms_normalize_keeps_silence_unchanged():
    audio = np.zeros(16000, dtype=np.float32)

    normalized = rms_normalize(audio)

    np.testing.assert_array_equal(normalized, audio)


def test_rms_normalize_rejects_non_mono_audio():
    audio = np.zeros((2, 16000), dtype=np.float32)

    with pytest.raises(ValueError, match="mono 1-D"):
        rms_normalize(audio)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_rms": 0}, "target_rms"),
        ({"max_gain": 0.5}, "max_gain"),
        ({"peak_limit": 1.1}, "peak_limit"),
        ({"min_rms": -1}, "min_rms"),
    ],
)
def test_rms_normalize_rejects_invalid_parameters(kwargs, message):
    audio = np.zeros(16000, dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        rms_normalize(audio, **kwargs)
