"""Tests for SenseVoice wrapper output normalization."""

from __future__ import annotations

import numpy as np

from meetasr.models.asr.sense_voice import SenseVoice, _strip_sensevoice_tokens


class FakeSenseVoiceModel:
    """Small fake model returning SenseVoice-style metadata tokens."""

    def inference(self, **kwargs):
        return (
            [
                {
                    "key": "audio",
                    "text": "<|en|><|NEUTRAL|><|Speech|><|withitn|>Hello team.",
                }
            ],
            {"load_data": "0.001"},
        )


def test_strip_sensevoice_tokens_removes_prefix_metadata():
    """SenseVoice metadata tokens should not appear in transcript text."""
    text = "<|vi|><|NEUTRAL|><|Speech|><|withitn|>Xin chao."

    assert _strip_sensevoice_tokens(text) == "Xin chao."


def test_recognize_normalizes_text_and_keeps_raw_text(monkeypatch):
    """Recognize should return MeetASR ASR result dicts."""
    asr = SenseVoice()
    asr._model = FakeSenseVoiceModel()
    asr._inference_kwargs = {"device": "cpu", "frontend": object(), "tokenizer": object()}
    monkeypatch.setattr(asr, "_ensure_loaded", lambda: None)
    audio = np.zeros(16000, dtype=np.float32)

    results = asr.recognize(audio, language="en")

    assert results == [
        {
            "key": "audio",
            "text": "Hello team.",
            "raw_text": "<|en|><|NEUTRAL|><|Speech|><|withitn|>Hello team.",
            "timestamp": [],
        }
    ]
