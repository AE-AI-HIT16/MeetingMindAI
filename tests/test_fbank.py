"""Tests for WavFrontend and LFR feature stacking."""

from __future__ import annotations

import numpy as np
import pytest

from meetasr.frontends.fbank import WavFrontend, compute_lfr_features


def test_compute_lfr_features_shape_and_dtype():
    """LFR stacking should produce [ceil(T / lfr_n), D * lfr_m] float32 output."""
    fbank = np.random.default_rng(0).normal(size=(100, 80)).astype(np.float32)

    lfr = compute_lfr_features(fbank, lfr_m=7, lfr_n=6)

    assert lfr.shape == (17, 560)
    assert lfr.dtype == np.float32


def test_compute_lfr_features_handles_short_input():
    """Inputs shorter than lfr_m should be padded by clamped border frames."""
    fbank = np.arange(3 * 80, dtype=np.float32).reshape(3, 80)

    lfr = compute_lfr_features(fbank, lfr_m=7, lfr_n=6)

    assert lfr.shape == (1, 560)
    assert np.array_equal(lfr[0, :80], fbank[0])
    assert np.array_equal(lfr[0, -80:], fbank[-1])


def test_compute_lfr_features_known_input_output():
    """LFR stacking should clamp border frames and concatenate in order."""
    fbank = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=np.float32,
    )

    lfr = compute_lfr_features(fbank, lfr_m=3, lfr_n=2)

    expected = np.array(
        [
            [1.0, 10.0, 1.0, 10.0, 2.0, 20.0],
            [2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(lfr, expected)


def test_compute_lfr_features_rejects_invalid_shape():
    """LFR stacking requires a 2-D feature matrix."""
    with pytest.raises(ValueError):
        compute_lfr_features(np.zeros(80, dtype=np.float32))


def test_wav_frontend_default_lfr_output_size():
    """Default WavFrontend config should match FunASR LFR dimensions."""
    frontend = WavFrontend()

    assert frontend.lfr_m == 7
    assert frontend.lfr_n == 6
    assert frontend.output_size() == 560


def test_wav_frontend_forward_extracts_lfr_features_from_audio():
    """WavFrontend.forward should convert waveform input to LFR feature output."""
    frontend = WavFrontend()
    duration_s = 0.25
    time = np.linspace(0.0, duration_s, int(frontend.fs * duration_s), endpoint=False)
    audio = np.sin(2 * np.pi * 440.0 * time).astype(np.float32)

    features = frontend.forward(audio)

    assert features.ndim == 2
    assert features.shape[0] > 0
    assert features.shape[1] == 560
    assert features.dtype == np.float32


