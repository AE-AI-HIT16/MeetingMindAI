"""Tests for the offline Silero VAD wrapper without loading real weights."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from meetasr.api import dependencies
from meetasr.api.routes.health import health
from meetasr.auto.auto_model import AutoModel
from meetasr.models.vad.silero_vad import SileroVAD, _normalize_timestamps
from meetasr.register import tables
from meetasr.utils import download


def test_silero_vad_is_registered() -> None:
    assert tables.model_classes["silero-vad"] is SileroVAD


def test_auto_model_builds_packaged_silero_without_hub_download(monkeypatch) -> None:
    def fail_if_downloaded(*args, **kwargs):
        pytest.fail("packaged Silero model must not use a remote model hub")

    monkeypatch.setattr(download, "_resolve_model_path", fail_if_downloaded)

    vad = AutoModel(model="silero-vad", device="cpu")

    assert isinstance(vad, SileroVAD)
    assert vad.backend == "onnx"


@pytest.mark.asyncio
async def test_health_reports_actual_silero_model_name(monkeypatch) -> None:
    class FakePipeline:
        vad = SileroVAD()
        asr = None
        punc = None
        spk = None
        summarizer = None

    monkeypatch.setattr(dependencies, "_pipeline", FakePipeline())

    payload = await health()

    assert "silero-vad" in payload["models_loaded"]


def test_normalize_timestamps_converts_samples_clips_and_sorts() -> None:
    segments = _normalize_timestamps(
        [
            {"start": 8000, "end": 16000},
            {"start": 160, "end": 3360},
            {"start": -100, "end": 100},
            {"start": 1000, "end": 1000},
            {"bad": "entry"},
            "invalid",
        ],
        audio_samples=16000,
        min_segment_ms=200,
    )

    assert [(segment.start_ms, segment.end_ms) for segment in segments] == [
        (10, 210),
        (500, 1000),
    ]


def test_empty_audio_returns_without_loading_model(monkeypatch) -> None:
    vad = SileroVAD()
    monkeypatch.setattr(
        vad,
        "_ensure_loaded",
        lambda: pytest.fail("empty audio must not load Silero"),
    )

    assert vad.detect(np.array([], dtype=np.float32)) == []


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (np.zeros((2, 16000), dtype=np.float32), "mono 1-D"),
        (np.zeros(16000, dtype=np.float64), "float32"),
        ([0.0, 0.0], "numpy.ndarray"),
    ],
)
def test_detect_rejects_invalid_audio(audio, message) -> None:
    with pytest.raises(ValueError, match=message):
        SileroVAD().detect(audio)


def test_detect_forwards_config_and_returns_millisecond_segments() -> None:
    captured = {}
    vad = SileroVAD(
        threshold=0.55,
        neg_threshold=0.38,
        min_speech_duration_ms=200,
        min_silence_duration_ms=450,
        speech_pad_ms=60,
        max_speech_duration_s=20,
    )
    vad._model = object()

    def fake_get_speech_timestamps(audio, model, **kwargs):
        captured["audio"] = audio
        captured["model"] = model
        captured.update(kwargs)
        return [{"start": 1600, "end": 8000}]

    vad._get_speech_timestamps = fake_get_speech_timestamps
    segments = vad.detect(np.zeros(16000, dtype=np.float32))

    assert [(segment.start_ms, segment.end_ms) for segment in segments] == [
        (100, 500),
    ]
    assert captured["audio"].dtype.is_floating_point
    assert captured["model"] is vad._model
    assert captured["sampling_rate"] == 16000
    assert captured["threshold"] == 0.55
    assert captured["neg_threshold"] == 0.38
    assert captured["min_speech_duration_ms"] == 200
    assert captured["min_silence_duration_ms"] == 450
    assert captured["speech_pad_ms"] == 60
    assert captured["max_speech_duration_s"] == 20.0
    assert captured["return_seconds"] is False


def test_model_is_lazy_loaded_once_from_official_package(monkeypatch) -> None:
    calls = []
    fake_model = object()
    fake_module = ModuleType("silero_vad")

    def fake_load_silero_vad(**kwargs):
        calls.append(kwargs)
        return fake_model

    fake_module.load_silero_vad = fake_load_silero_vad
    fake_module.get_speech_timestamps = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "silero_vad", fake_module)

    vad = SileroVAD(backend="onnx", opset_version=16)
    vad._ensure_loaded()
    vad._ensure_loaded()

    assert vad._model is fake_model
    assert calls == [{"onnx": True, "opset_version": 16}]


def test_streaming_predictors_keep_independent_recurrent_state() -> None:
    class FakeStatefulModel:
        def __init__(self) -> None:
            self.reset_states()

        def reset_states(self) -> None:
            self._state = torch.zeros(1)
            self._context = torch.zeros(1)
            self._last_sr = 0
            self._last_batch_size = 0

        def __call__(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
            assert audio.shape == (512,)
            self._state += 0.1
            self._last_sr = sample_rate
            self._last_batch_size = 1
            return self._state.clone()

    vad = SileroVAD()
    vad._model = FakeStatefulModel()
    first = vad.create_streaming_predictor()
    second = vad.create_streaming_predictor()
    frame = np.zeros(512, dtype=np.float32)

    assert first.predict(frame) == pytest.approx(0.1)
    assert first.predict(frame) == pytest.approx(0.2)
    assert second.predict(frame) == pytest.approx(0.1)

    first.reset()
    assert first.predict(frame) == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "invalid"}, "backend"),
        ({"device": "cuda:0"}, "cpu"),
        ({"threshold": 1.0}, "threshold"),
        ({"threshold": 0.5, "neg_threshold": 0.6}, "neg_threshold"),
        ({"min_speech_duration_ms": 0}, "min_speech_duration_ms"),
        ({"min_silence_duration_ms": -1}, "min_silence_duration_ms"),
        ({"speech_pad_ms": -1}, "speech_pad_ms"),
        ({"max_speech_duration_s": 0}, "max_speech_duration_s"),
        ({"opset_version": 14}, "opset_version"),
    ],
)
def test_invalid_config_is_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SileroVAD(**kwargs)
