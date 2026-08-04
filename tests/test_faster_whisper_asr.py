from types import SimpleNamespace

import numpy as np

from meetasr.models.asr import faster_whisper_asr
from meetasr.models.asr.faster_whisper_asr import FasterWhisperASR
from pathlib import PureWindowsPath



class _FakeWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        words = [SimpleNamespace(word=" xin", start=0.0, end=0.2)]
        return iter([SimpleNamespace(words=words)]), SimpleNamespace()


def test_windows_cuda_runtime_adds_pip_nvidia_dll_directories(monkeypatch):
    root = PureWindowsPath(r"C:\Python\Lib\site-packages")

    expected = [
        str(root / "nvidia" / "cublas" / "bin"),
        str(root / "nvidia" / "cudnn" / "bin"),
        str(root / "nvidia" / "cuda_nvrtc" / "bin"),
    ]

    added = []

    monkeypatch.setattr(faster_whisper_asr.os, "name", "nt")
    monkeypatch.setattr(
        faster_whisper_asr.site,
        "getsitepackages",
        lambda: [str(root)],
    )

    monkeypatch.setattr(
        faster_whisper_asr.os.path,
        "isdir",
        lambda path: path in expected,
    )

    monkeypatch.setattr(
        faster_whisper_asr.os,
        "add_dll_directory",
        lambda path: added.append(path),
        raising=False,
    )

    monkeypatch.setattr(
        faster_whisper_asr,
        "_CUDA_DLL_DIRECTORIES",
        [],
    )

    monkeypatch.setattr(
        faster_whisper_asr,
        "_CUDA_DLL_DIRECTORIES_CONFIGURED",
        False,
    )

    monkeypatch.setenv("PATH", "original")

    faster_whisper_asr._configure_windows_cuda_runtime()

    assert added == expected


def test_recognize_forwards_configured_decode_options():
    model = FasterWhisperASR(
        beam_size=5,
        temperature=[0.0, 0.2],
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    fake_model = _FakeWhisperModel()
    model._model = fake_model

    result = model.recognize(np.zeros(16000, dtype=np.float32), language="vi")

    _, call_kwargs = fake_model.calls[0]
    assert call_kwargs == {
        "language": "vi",
        "word_timestamps": True,
        "vad_filter": False,
        "beam_size": 5,
        "temperature": [0.0, 0.2],
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }
    assert result[0]["text"] == "xin"
    assert len(result[0]["timestamp"]) == len(result[0]["text"])


def test_recognize_runtime_decode_options_override_config():
    model = FasterWhisperASR(beam_size=5, condition_on_previous_text=False)
    fake_model = _FakeWhisperModel()
    model._model = fake_model

    model.recognize(
        np.zeros(16000, dtype=np.float32),
        language="auto",
        beam_size=3,
        condition_on_previous_text=True,
    )

    _, call_kwargs = fake_model.calls[0]
    assert call_kwargs["language"] is None
    assert call_kwargs["beam_size"] == 3
    assert call_kwargs["condition_on_previous_text"] is True
    assert call_kwargs["word_timestamps"] is True
    assert call_kwargs["vad_filter"] is False


def test_recognize_long_form_uses_internal_vad_and_global_timestamps():
    model = FasterWhisperASR(
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
    )
    fake_model = _FakeWhisperModel()
    def transcribe(audio, **kwargs):
        fake_model.calls.append((audio, kwargs))
        return iter([
            SimpleNamespace(words=[
                SimpleNamespace(word=" xin", start=1.0, end=1.2),
            ]),
            SimpleNamespace(words=[
                SimpleNamespace(word=" chao.", start=3.0, end=3.4),
            ]),
        ]), SimpleNamespace()

    fake_model.transcribe = transcribe
    model._model = fake_model

    result = model.recognize_long_form(
        np.zeros(5 * 16000, dtype=np.float32), language="vi"
    )

    assert result == [
        {
            "key": "long_form_0",
            "text": "xin",
            "timestamp": [[1000, 1200]] * 3,
        },
        {
            "key": "long_form_1",
            "text": "chao.",
            "timestamp": [[3000, 3400]] * 5,
        },
    ]
    _, call_kwargs = fake_model.calls[0]
    assert call_kwargs["vad_filter"] is True
    assert call_kwargs["vad_parameters"] == {"min_silence_duration_ms": 500}
    assert call_kwargs["hallucination_silence_threshold"] == 2.0
    assert call_kwargs["word_timestamps"] is True


def test_result_preserves_native_segment_quality_fields():
    segment = SimpleNamespace(
        start=1.0,
        end=1.4,
        avg_logprob=-0.27,
        no_speech_prob=0.03,
        compression_ratio=1.12,
        words=[SimpleNamespace(word=" xin", start=1.0, end=1.4)],
    )

    result = FasterWhisperASR._result_from_segments([segment], key="quality")

    assert result["text"] == "xin"
    assert result["segment_quality"] == [{
        "start": 1.0,
        "end": 1.4,
        "avg_logprob": -0.27,
        "no_speech_prob": 0.03,
        "compression_ratio": 1.12,
    }]
