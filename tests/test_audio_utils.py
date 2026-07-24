"""Tests for audio loading utilities."""

import io
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from meetasr.utils import audio as audio_utils
from meetasr.utils.audio import SAMPLE_RATE, load_audio


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

    def test_librosa_fallback_downmixes_channel_first_audio(self, monkeypatch):
        """librosa fallback returns [channels, samples], not [samples, channels]."""
        left = make_sine()
        right = left * 0.5
        channel_first = np.stack([left, right], axis=0)

        def fail_soundfile_read(*args, **kwargs):
            raise RuntimeError("unsupported container")

        fake_soundfile = SimpleNamespace(read=fail_soundfile_read)
        fake_librosa = SimpleNamespace(
            load=lambda *args, **kwargs: (channel_first, SAMPLE_RATE),
        )
        monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
        monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

        result = audio_utils._load_from_file("sample.ogg", SAMPLE_RATE)

        assert result.shape == left.shape
        np.testing.assert_allclose(result, (left + right) / 2)

    def test_load_m4a_uses_ffmpeg(self, tmp_path, monkeypatch):
        """Compressed M4A input should be decoded by the FFmpeg executable."""
        m4a_path = tmp_path / "test.m4a"
        m4a_path.write_bytes(b"fake m4a for mocked ffmpeg")
        expected = np.array([0.25, -0.5, 0.75], dtype="<f4")
        captured = {}

        monkeypatch.setattr("meetasr.utils.audio.shutil.which", lambda name: "/usr/bin/ffmpeg")

        def fake_run(command, **kwargs):
            captured["command"] = command
            return subprocess.CompletedProcess(command, 0, expected.tobytes(), b"")

        monkeypatch.setattr("meetasr.utils.audio.subprocess.run", fake_run)

        result = load_audio(str(m4a_path))

        np.testing.assert_array_equal(result, expected)
        assert captured["command"][0] == "/usr/bin/ffmpeg"
        assert captured["command"][captured["command"].index("-ar") + 1] == str(SAMPLE_RATE)
        assert captured["command"][captured["command"].index("-ac") + 1] == "1"

    def test_load_m4a_reports_ffmpeg_decode_error(self, tmp_path, monkeypatch):
        """Invalid compressed audio should produce a useful decode error."""
        m4a_path = tmp_path / "broken.m4a"
        m4a_path.write_bytes(b"not audio")

        monkeypatch.setattr("meetasr.utils.audio.shutil.which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(
            "meetasr.utils.audio.subprocess.run",
            lambda command, **kwargs: subprocess.CompletedProcess(
                command, 1, b"", b"Invalid data found when processing input"
            ),
        )

        with pytest.raises(ValueError, match="Invalid data found"):
            load_audio(str(m4a_path))
