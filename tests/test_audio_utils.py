"""Tests for audio loading utilities."""

import io
import numpy as np
import pytest
from meetasr.utils.audio import load_audio, SAMPLE_RATE


def make_sine(duration_s: float = 1.0, freq: float = 440.0) -> np.ndarray:
    """Generate a sine wave for testing."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t)


class TestLoadAudio:

    def test_load_numpy_passthrough(self):
        """numpy array should be returned as-is (float32)."""
        audio = make_sine()
        result = load_audio(audio)
        assert result.dtype == np.float32
        assert result.ndim == 1

    def test_load_stereo_numpy_converted_to_mono(self):
        """Stereo numpy array should be averaged to mono."""
        mono = make_sine()
        stereo = np.stack([mono, mono * 0.5], axis=0)  # [2, N]
        result = load_audio(stereo)
        assert result.ndim == 1

    def test_load_nonexistent_path_raises(self):
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/audio.wav")

    def test_load_unsupported_type_raises(self):
        """ValueError for unsupported type."""
        with pytest.raises(ValueError):
            load_audio(12345)

    def test_load_wav_file(self, tmp_path):
        """Load a real WAV file from disk."""
        import soundfile as sf
        audio = make_sine(duration_s=2.0)
        wav_path = str(tmp_path / "test.wav")
        sf.write(wav_path, audio, SAMPLE_RATE)

        result = load_audio(wav_path)
        assert result.dtype == np.float32
        assert result.ndim == 1
        assert abs(len(result) - len(audio)) < 100  # allow small resampling diff
