"""Tests for Vietnamese Zipformer ASR wrapper."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from meetasr.models.asr.zipformer_vi import (
    ZipformerViASR,
    _find_optional_file,
    _find_required_file,
    _provider_from_device,
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
    assert tables.model_classes["hynt/Zipformer-30M-RNNT-6000h"] is ZipformerViASR


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


def test_find_required_file_accepts_config_json_tokens(tmp_path):
    token_file = tmp_path / "config.json"
    token_file.write_text("0 <blank>\n1 xin\n", encoding="utf-8")

    found = _find_required_file(tmp_path, ("tokens.txt", "config.json"))

    assert found == token_file


def test_find_optional_file_returns_none_when_missing(tmp_path):
    assert _find_optional_file(tmp_path, ("bpe.model",)) is None


def test_provider_from_device_maps_project_device_strings():
    assert _provider_from_device("cpu") == "cpu"
    assert _provider_from_device("cuda:0") == "cuda"
    assert _provider_from_device("coreml") == "coreml"
