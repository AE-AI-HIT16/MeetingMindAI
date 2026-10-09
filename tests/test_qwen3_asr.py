from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from meetasr.auto.auto_model import AutoModel
from meetasr.models.asr.qwen3_asr import (
    Qwen3ASR,
    _expand_word_timestamps_to_chars,
    _map_language,
)
from meetasr.register import tables


class _FakeQwenModel:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or [
            SimpleNamespace(language="Vietnamese", text="xin chào", time_stamps=None)
        ]

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


def test_qwen3_asr_is_registered_and_buildable_without_downloading():
    assert tables.model_classes["qwen3-asr"] is Qwen3ASR
    assert tables.model_classes["Qwen/Qwen3-ASR-1.7B"] is Qwen3ASR

    model = AutoModel(
        model="qwen3-asr",
        hub="none",
    )

    assert isinstance(model, Qwen3ASR)
    assert model.model_name == "Qwen/Qwen3-ASR-1.7B"


def test_qwen3_asr_loads_resolved_local_path_with_explicit_options(monkeypatch):
    captured = {}

    class _FakeLoader:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            captured["checkpoint"] = checkpoint
            captured["kwargs"] = kwargs
            return object()

    fake_qwen_module = ModuleType("qwen_asr")
    fake_qwen_module.Qwen3ASRModel = _FakeLoader
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_module)

    model = Qwen3ASR(
        model_path="/models/qwen3-asr-1.7b",
        device="cuda",
        dtype="float16",
        forced_aligner="/models/qwen3-forced-aligner",
        max_inference_batch_size=2,
        max_new_tokens=256,
        model_kwargs={"low_cpu_mem_usage": True},
    )
    model._ensure_loaded()

    assert captured["checkpoint"] == "/models/qwen3-asr-1.7b"
    assert captured["kwargs"]["device_map"] == "cuda:0"
    assert captured["kwargs"]["max_inference_batch_size"] == 2
    assert captured["kwargs"]["max_new_tokens"] == 256
    assert captured["kwargs"]["low_cpu_mem_usage"] is True
    assert captured["kwargs"]["forced_aligner"] == "/models/qwen3-forced-aligner"
    assert captured["kwargs"]["forced_aligner_kwargs"]["device_map"] == "cuda:0"


def test_qwen3_asr_lazy_load_is_serialized_across_workers(monkeypatch):
    calls = 0
    loaded_model = object()

    class _SlowFakeLoader:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            nonlocal calls
            calls += 1
            time.sleep(0.05)
            return loaded_model

    fake_qwen_module = ModuleType("qwen_asr")
    fake_qwen_module.Qwen3ASRModel = _SlowFakeLoader
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_module)
    model = Qwen3ASR(device="cuda:0")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(model._ensure_loaded) for _ in range(2)]
        for future in futures:
            future.result()

    assert calls == 1
    assert model._model is loaded_model


def test_qwen3_asr_warm_up_loads_weights_without_running_inference(monkeypatch):
    model = Qwen3ASR()
    loaded_model = _FakeQwenModel()
    calls = 0

    def fake_load() -> None:
        nonlocal calls
        calls += 1
        model._model = loaded_model

    monkeypatch.setattr(model, "_load_model", fake_load)

    model.warm_up()
    model.warm_up()

    assert calls == 1
    assert loaded_model.calls == []


def test_qwen3_asr_inference_is_serialized_across_workers():
    active = 0
    max_active = 0
    guard = threading.Lock()

    class _ConcurrentFakeModel(_FakeQwenModel):
        def transcribe(self, **kwargs):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with guard:
                active -= 1
            return self.results

    model = Qwen3ASR()
    model._model = _ConcurrentFakeModel()
    audio = np.zeros(1600, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(model.recognize, audio) for _ in range(2)]
        for future in futures:
            future.result()

    assert max_active == 1


def test_qwen3_asr_recognize_normalizes_audio_and_language():
    fake_model = _FakeQwenModel()
    model = Qwen3ASR()
    model._model = fake_model

    result = model.recognize(
        np.zeros(1600, dtype=np.float64),
        language="vi",
        context="Họp dự án MeetASR",
        key="meeting",
    )

    assert result == [{
        "key": "meeting",
        "text": "xin chào",
        "timestamp": [],
        "language": "Vietnamese",
    }]
    call = fake_model.calls[0]
    chunk, sample_rate = call["audio"][0]
    assert chunk.dtype == np.float32
    assert chunk.flags["C_CONTIGUOUS"]
    assert sample_rate == 16000
    assert call["language"] == ["Vietnamese"]
    assert call["context"] == "Họp dự án MeetASR"
    assert call["return_time_stamps"] is False


def test_qwen3_asr_skips_unsupported_vietnamese_forced_alignment(caplog):
    fake_model = _FakeQwenModel()
    model = Qwen3ASR(forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B")
    model._model = fake_model

    result = model.recognize(
        np.zeros(1600, dtype=np.float32),
        language="vi",
    )

    assert result[0]["timestamp"] == []
    assert fake_model.calls[0]["return_time_stamps"] is False
    assert "forced alignment skipped" in caplog.text


def test_qwen3_asr_expands_supported_language_alignment_to_each_character():
    alignment = SimpleNamespace(items=[
        SimpleNamespace(text="hello", start_time=0.0, end_time=0.4),
        SimpleNamespace(text="world", start_time=0.5, end_time=0.9),
    ])
    transcription = SimpleNamespace(
        language="English",
        text="hello world.",
        time_stamps=alignment,
    )
    fake_model = _FakeQwenModel([transcription])
    model = Qwen3ASR(forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B")
    model._model = fake_model

    result = model.recognize(
        np.zeros(1600, dtype=np.float32),
        language="en",
    )[0]

    assert len(result["timestamp"]) == len(result["text"])
    assert result["timestamp"][:5] == [[0, 400]] * 5
    assert result["timestamp"][5] == [400, 500]
    assert result["timestamp"][6:11] == [[500, 900]] * 5
    assert result["timestamp"][-1] == [500, 900]
    assert fake_model.calls[0]["return_time_stamps"] is True


def test_timestamp_expansion_accepts_qwen_iterable_result():
    alignment = [
        SimpleNamespace(text="A", start_time=0.1, end_time=0.2),
        SimpleNamespace(text="B", start_time=0.3, end_time=0.4),
    ]

    assert _expand_word_timestamps_to_chars(alignment, "A B") == [
        [100, 200],
        [200, 300],
        [300, 400],
    ]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("auto", None),
        ("vi", "Vietnamese"),
        ("en", "English"),
        ("Vietnamese", "Vietnamese"),
        ("unsupported", None),
    ],
)
def test_qwen_language_mapping(code, expected):
    assert _map_language(code) == expected


def test_qwen3_asr_rejects_invalid_audio_and_dtype():
    with pytest.raises(ValueError, match="Unsupported Qwen3-ASR dtype"):
        Qwen3ASR(dtype="int8")

    model = Qwen3ASR()
    model._model = _FakeQwenModel()

    with pytest.raises(ValueError, match="mono 1-D"):
        model.recognize(np.zeros((2, 800), dtype=np.float32))

    invalid_audio = np.zeros(800, dtype=np.float32)
    invalid_audio[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        model.recognize(invalid_audio)
