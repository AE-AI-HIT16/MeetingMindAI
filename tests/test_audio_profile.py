"""Tests for audio profiling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from meetasr.utils.audio_profile import profile_waveform


def test_profile_silence_audio():
    audio = np.zeros(16000, dtype=np.float32)

    profile = profile_waveform(audio)

    assert profile.duration == 1.0
    assert profile.peak == 0.0
    assert profile.rms == 0.0
    assert profile.dc_offset == 0.0
    assert profile.clipping_ratio == 0.0
    assert profile.near_silence_ratio == 1.0
    assert profile.noise_rms == 0.0
    assert profile.speech_rms == 0.0
    assert profile.estimated_snr_db == 0.0


def test_profile_detects_dc_offset_and_peak():
    audio = np.full(16000, 0.25, dtype=np.float32)

    profile = profile_waveform(audio)

    assert profile.peak == 0.25
    assert profile.rms == 0.25
    assert profile.dc_offset == 0.25
    assert profile.near_silence_ratio == 0.0
    assert profile.noise_rms == 0.25
    assert profile.speech_rms == 0.25
    assert profile.estimated_snr_db == 0.0


def test_profile_detects_clipping_ratio():
    audio = np.zeros(10, dtype=np.float32)
    audio[:2] = 1.0
    audio[2] = -1.0

    profile = profile_waveform(audio, sample_rate=10)

    assert profile.duration == 1.0
    assert profile.clipping_ratio == 0.3


def test_profile_rejects_non_mono_audio():
    audio = np.zeros((2, 16000), dtype=np.float32)

    with pytest.raises(ValueError, match="mono 1-D"):
        profile_waveform(audio)


def test_profile_empty_audio():
    audio = np.array([], dtype=np.float32)

    profile = profile_waveform(audio)

    assert profile.duration == 0.0
    assert profile.shape == (0,)


def test_profile_estimates_frame_based_snr():
    quiet_frames = np.full(1600, 0.01, dtype=np.float32)
    speech_frames = np.full(1600, 0.1, dtype=np.float32)
    audio = np.concatenate([quiet_frames, speech_frames])

    profile = profile_waveform(audio, frame_duration_ms=20)

    assert profile.noise_rms == pytest.approx(0.01)
    assert profile.speech_rms == pytest.approx(0.1)
    assert profile.estimated_snr_db == pytest.approx(20.0)


def test_profile_caps_snr_for_digitally_silent_frames():
    silence = np.zeros(1600, dtype=np.float32)
    speech = np.full(1600, 0.1, dtype=np.float32)

    profile = profile_waveform(
        np.concatenate([silence, speech]),
        frame_duration_ms=20,
    )

    assert profile.noise_rms == 0.0
    assert profile.speech_rms == pytest.approx(0.1)
    assert profile.estimated_snr_db == 100.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_rate": 0}, "sample_rate"),
        ({"frame_duration_ms": 0}, "frame_duration_ms"),
        ({"noise_percentile": 80, "speech_percentile": 20}, "percentiles"),
    ],
)
def test_profile_rejects_invalid_frame_parameters(kwargs, message):
    audio = np.zeros(16000, dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        profile_waveform(audio, **kwargs)
