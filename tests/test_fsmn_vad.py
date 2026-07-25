"""Tests for FSMN-VAD wrapper behavior without loading a real model."""

from __future__ import annotations

from pathlib import Path
import logging
from pathlib import Path
from types import SimpleNamespace

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


class RuntimeOptionVadModel:
    """Fake model that reads the persistent VAD option during inference."""

    def __init__(self, max_single_segment_time=60000):
        self.vad_opts = SimpleNamespace(
            max_single_segment_time=max_single_segment_time,
        )
        self.observed_max_segment_ms = None

    def inference(self, data_in, **kwargs):
        self.observed_max_segment_ms = self.vad_opts.max_single_segment_time
        return []


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


def test_detect_short_audio_returns_empty_without_loading_model(monkeypatch):
    """Audio too short for frontend frames should not reach FunASR."""
    vad = FsmnVAD()
    monkeypatch.setattr(
        vad,
        "_ensure_loaded",
        lambda: pytest.fail("short audio must not load the model"),
    )
    audio = np.zeros(1599, dtype=np.float32)

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


def test_ensure_loaded_applies_only_supported_vad_overrides(
    monkeypatch,
    tmp_path: Path,
    caplog,
):
    """Wrapper config should override YAML without forwarding unknown options."""
    import torch
    import funasr.frontends.wav_frontend as frontend_module
    import funasr.models.fsmn_vad_streaming.model as vad_module

    (tmp_path / "config.yaml").write_text(
        "\n".join([
            "frontend_conf: {}",
            "model_conf:",
            "  max_single_segment_time: 60000",
            "  max_end_silence_time: 800",
            "encoder: FSMN",
            "encoder_conf: {}",
        ]),
        encoding="utf-8",
    )
    captured_model_conf = {}

    class FakeFrontend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeVADXOptions:
        def __init__(
            self,
            max_single_segment_time=60000,
            max_end_silence_time=800,
            **kwargs,
        ):
            self.max_single_segment_time = max_single_segment_time
            self.max_end_silence_time = max_end_silence_time

    class FakeLoadedVad:
        def __init__(self, **kwargs):
            captured_model_conf.update(kwargs)
            self.vad_opts = FakeVADXOptions(**kwargs)

        def load_state_dict(self, state, strict=False):
            return None

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(frontend_module, "WavFrontendOnline", FakeFrontend)
    monkeypatch.setattr(vad_module, "VADXOptions", FakeVADXOptions)
    monkeypatch.setattr(vad_module, "FsmnVADStreaming", FakeLoadedVad)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})

    vad = FsmnVAD(
        model_path=str(tmp_path),
        max_single_segment_time=15000,
        max_end_silence_time=450,
        hub="ms",
        unsupported_vad_option=True,
    )

    with caplog.at_level(logging.WARNING):
        vad._ensure_loaded()

    assert captured_model_conf["max_single_segment_time"] == 15000
    assert captured_model_conf["max_end_silence_time"] == 450
    assert "hub" not in captured_model_conf
    assert "unsupported_vad_option" not in captured_model_conf
    assert "hub" not in caplog.text
    assert "unsupported_vad_option" in caplog.text


def test_detect_updates_runtime_max_segment_option(monkeypatch):
    """Per-call max duration must be visible to the FSMN state machine."""
    vad = FsmnVAD(max_single_segment_time=15000)
    vad._model = RuntimeOptionVadModel()
    vad._frontend = object()
    monkeypatch.setattr(vad, "_ensure_loaded", lambda: None)

    vad.detect(
        np.zeros(16000, dtype=np.float32),
        max_single_segment_time=12000,
    )

    assert vad._model.observed_max_segment_ms == 12000
