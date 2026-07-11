"""Tests for ViBERT-CaPu punctuation wrapper."""

from __future__ import annotations

from meetasr.models.punc.vibert_capu import ViBERTCaPuPunc
from meetasr.register import tables


class FakeCaPuModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return self.result


def test_vibert_capu_is_registered_with_aliases():
    assert tables.model_classes["vibert-capu"] is ViBERTCaPuPunc
    assert tables.model_classes["dragonSwing/vibert-capu"] is ViBERTCaPuPunc


def test_restore_returns_input_without_loading_for_blank_text(monkeypatch):
    punc = ViBERTCaPuPunc(model_path="models/vibert-capu")
    monkeypatch.setattr(
        punc,
        "_ensure_loaded",
        lambda: (_ for _ in ()).throw(AssertionError("should not load")),
    )

    assert punc.restore("   ") == "   "


def test_restore_returns_first_model_result_and_strips_whitespace(monkeypatch):
    punc = ViBERTCaPuPunc(model_path="models/vibert-capu")
    fake = FakeCaPuModel([" Alo alo. "])
    punc._inner = fake
    monkeypatch.setattr(punc, "_ensure_loaded", lambda: None)

    result = punc.restore("ALO ALO", batch_size=1)

    assert result == "Alo alo."
    assert fake.calls == [("alo alo", {"batch_size": 1})]


def test_restore_accepts_plain_string_output(monkeypatch):
    punc = ViBERTCaPuPunc(model_path="models/vibert-capu")
    punc._inner = FakeCaPuModel(" Alo alo. ")
    monkeypatch.setattr(punc, "_ensure_loaded", lambda: None)

    assert punc.restore("alo alo") == "Alo alo."


def test_restore_falls_back_to_input_for_unexpected_output(monkeypatch):
    punc = ViBERTCaPuPunc(model_path="models/vibert-capu")
    punc._inner = FakeCaPuModel({"text": "Alo alo."})
    monkeypatch.setattr(punc, "_ensure_loaded", lambda: None)

    assert punc.restore("alo alo") == "alo alo"
