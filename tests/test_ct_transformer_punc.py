"""Tests for CT-Transformer punctuation wrapper."""

from __future__ import annotations

from meetasr.models.punc.ct_transformer import CTTransformerPunc
from meetasr.register import tables


class FakePuncModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_ct_punc_is_registered_with_aliases():
    assert tables.model_classes["ct-punc"] is CTTransformerPunc
    assert (
        tables.model_classes["iic/punc_ct-transformer_cn-en-common-vocab471067-large"]
        is CTTransformerPunc
    )


def test_restore_returns_input_without_loading_for_blank_text(monkeypatch):
    punc = CTTransformerPunc()
    monkeypatch.setattr(
        punc,
        "_ensure_loaded",
        lambda: (_ for _ in ()).throw(AssertionError("should not load")),
    )

    assert punc.restore("   ") == "   "


def test_restore_returns_model_text_and_strips_outer_whitespace(monkeypatch):
    punc = CTTransformerPunc()
    fake = FakePuncModel([{"text": " Hello, team. "}])
    punc._inner = fake
    monkeypatch.setattr(punc, "_ensure_loaded", lambda: None)

    result = punc.restore("hello team", batch_size=1)

    assert result == "Hello, team."
    assert fake.calls == [{"input": "hello team", "batch_size": 1}]


def test_restore_accepts_tuple_output_from_funasr(monkeypatch):
    punc = CTTransformerPunc()
    punc._inner = FakePuncModel(([{"text": "Hello."}], {"meta": "ok"}))
    monkeypatch.setattr(punc, "_ensure_loaded", lambda: None)

    assert punc.restore("hello") == "Hello."


def test_restore_falls_back_to_input_for_unexpected_output(monkeypatch):
    punc = CTTransformerPunc()
    punc._inner = FakePuncModel({"text": "Hello."})
    monkeypatch.setattr(punc, "_ensure_loaded", lambda: None)

    assert punc.restore("hello") == "hello"
