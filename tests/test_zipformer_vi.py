"""Tests for Vietnamese Zipformer ASR wrapper."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from meetasr.auto import auto_pipeline
from meetasr.auto.auto_pipeline import AutoPipeline
from meetasr.models.asr.zipformer_vi import (
    DEFAULT_MODEL,
    ZipformerViASR,
    _extract_char_timestamps,
    _find_optional_file,
    _find_required_file,
    _provider_from_device,
    _resolve_model_file,
    _validate_audio_chunk,
)
from meetasr.register import tables


class FakeResult:
    text = "xin chao ca nha"


class FakeStream:
    def __init__(self):
        self.result = FakeResult()
        self.accepted = None

    def accept_waveform(self, sample_rate, samples):
        self.accepted = (sample_rate, samples)


class FakeRecognizer:
    def __init__(self):
        self.streams = []

    def create_stream(self):
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        assert stream in self.streams


def test_zipformer_vi_is_registered():
    assert tables.model_classes["zipformer-vi"] is ZipformerViASR
    assert tables.model_classes[DEFAULT_MODEL] is ZipformerViASR


def test_recognize_returns_meetasr_asr_result(monkeypatch):
    asr = ZipformerViASR()
    fake_recognizer = FakeRecognizer()
    monkeypatch.setattr(asr, "_ensure_loaded", lambda: None)
    asr._recognizer = fake_recognizer
    audio = np.zeros(16000, dtype=np.float32)

    results = asr.recognize(audio)

    assert results == [
        {
            "key": "chunk_0",
            "text": "xin chao ca nha",
            "timestamp": [],
        }
    ]
    sample_rate, samples = fake_recognizer.streams[0].accepted
    assert sample_rate == 16000
    assert samples.dtype == np.float32


def test_recognize_accepts_batch_with_stable_keys(monkeypatch):
    asr = ZipformerViASR()
    monkeypatch.setattr(asr, "_ensure_loaded", lambda: None)
    asr._recognizer = FakeRecognizer()
    audio = [np.zeros(16000, dtype=np.float32), np.ones(8000, dtype=np.float32)]

    results = asr.recognize(audio)

    assert [item["key"] for item in results] == ["chunk_0", "chunk_1"]
    assert [item["text"] for item in results] == ["xin chao ca nha", "xin chao ca nha"]


def test_validate_audio_chunk_rejects_non_mono_audio():
    audio = np.zeros((1, 16000), dtype=np.float32)

    with pytest.raises(ValueError, match="mono 1D audio"):
        _validate_audio_chunk(audio)


def test_recognize_rejects_non_vietnamese_and_non_finite_audio(monkeypatch):
    asr = ZipformerViASR()
    monkeypatch.setattr(asr, "_ensure_loaded", lambda: None)
    asr._recognizer = FakeRecognizer()

    with pytest.raises(ValueError, match="only supports Vietnamese"):
        asr.recognize(np.zeros(16000, dtype=np.float32), language="en")

    invalid = np.zeros(16000, dtype=np.float32)
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        asr.recognize(invalid, language="vi")


def test_find_required_file_accepts_config_json_tokens(tmp_path):
    token_file = tmp_path / "config.json"
    token_file.write_text("0 <blank>\n1 xin\n", encoding="utf-8")

    found = _find_required_file(tmp_path, ("tokens.txt", "config.json"))

    assert found == token_file


def test_find_optional_file_returns_none_when_missing(tmp_path):
    assert _find_optional_file(tmp_path, ("bpe.model",)) is None


def test_model_variant_selects_consistent_fp32_or_int8_graphs(tmp_path):
    for role in ("encoder", "decoder", "joiner"):
        (tmp_path / f"{role}-model.onnx").touch()
        (tmp_path / f"{role}-model.int8.onnx").touch()

    assert _resolve_model_file(
        tmp_path,
        "",
        role="encoder",
        variant="fp32",
    ).name == "encoder-model.onnx"
    assert _resolve_model_file(
        tmp_path,
        "",
        role="encoder",
        variant="int8",
    ).name == "encoder-model.int8.onnx"


def test_build_recognizer_forwards_explicit_snapshot_files(tmp_path, monkeypatch):
    filenames = {
        "encoder": "encoder-fp32.onnx",
        "decoder": "decoder-fp32.onnx",
        "joiner": "joiner-fp32.onnx",
        "tokens": "config.json",
        "bpe_vocab": "bpe.model",
    }
    for filename in filenames.values():
        (tmp_path / filename).touch()

    captured = {}

    class _FakeOfflineRecognizer:
        @classmethod
        def from_transducer(cls, **kwargs):
            captured.update(kwargs)
            return "recognizer"

    fake_sherpa = ModuleType("sherpa_onnx")
    fake_sherpa.OfflineRecognizer = _FakeOfflineRecognizer
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)

    asr = ZipformerViASR(
        model_path=str(tmp_path),
        device="cpu",
        num_threads=3,
        encoder=filenames["encoder"],
        decoder=filenames["decoder"],
        joiner=filenames["joiner"],
        tokens=filenames["tokens"],
        bpe_vocab=filenames["bpe_vocab"],
    )

    assert asr._build_recognizer() == "recognizer"
    assert captured["encoder"] == str(tmp_path / filenames["encoder"])
    assert captured["decoder"] == str(tmp_path / filenames["decoder"])
    assert captured["joiner"] == str(tmp_path / filenames["joiner"])
    assert captured["tokens"] == str(tmp_path / filenames["tokens"])
    assert captured["bpe_vocab"] == str(tmp_path / filenames["bpe_vocab"])
    assert captured["modeling_unit"] == "bpe"
    assert captured["num_threads"] == 3
    assert captured["provider"] == "cpu"


def test_extract_char_timestamps_reconstructs_bpe_text():
    result = SimpleNamespace(
        tokens=["▁XIN", "▁CHÀO"],
        timestamps=[0.1, 0.5],
    )

    text, timestamps = _extract_char_timestamps(result)

    assert text == "XIN CHÀO"
    assert len(timestamps) == len(text)
    assert timestamps[:3] == [[100, 500]] * 3
    assert timestamps[3:] == [[500, 580]] * 5


def test_provider_from_device_maps_project_device_strings():
    assert _provider_from_device("cpu") == "cpu"
    assert _provider_from_device("cuda:0") == "cuda"
    assert _provider_from_device("coreml") == "coreml"


def test_zipformer_config_keeps_common_diarization_first_pipeline(monkeypatch):
    built = []

    def fake_auto_model(**kwargs):
        built.append(kwargs)
        return object()

    monkeypatch.setattr(auto_pipeline, "AutoModel", fake_auto_model)
    pipeline = AutoPipeline.from_yaml("configs/models/zipformer_vi.yaml")

    assert pipeline.diarization_first is True
    assert pipeline.transcription_language == "vi"
    assert pipeline.speaker_turn_max_chunk_ms == 15000
    assert pipeline.speaker_turn_boundary_search_ms == 2000
    assert pipeline.speaker_turn_min_chunk_ms == 1000
    assert built[0]["model"] == "zipformer-vi"
    assert built[0]["model_variant"] == "fp32"
    assert built[0]["device"] == "cpu"
    assert [config["model"] for config in built[1:]] == [
        "silero-vad",
        "vibert-capu",
        "cam++",
    ]
