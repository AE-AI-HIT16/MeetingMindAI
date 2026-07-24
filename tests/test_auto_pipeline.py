"""Tests for constructing MeetPipeline from configuration."""

from meetasr.auto import auto_pipeline
from meetasr.auto.auto_pipeline import AutoPipeline


def _patch_asr_model(monkeypatch):
    fake_asr = object()
    monkeypatch.setattr(auto_pipeline, "AutoModel", lambda **kwargs: fake_asr)
    return fake_asr


def test_gap_rescue_is_disabled_by_default(monkeypatch):
    fake_asr = _patch_asr_model(monkeypatch)

    pipeline = AutoPipeline.from_config({"asr": {"model": "fake-asr"}})

    assert pipeline.asr is fake_asr
    assert pipeline.enable_gap_rescue is False


def test_gap_rescue_can_be_enabled_from_pipeline_config(monkeypatch):
    _patch_asr_model(monkeypatch)

    pipeline = AutoPipeline.from_config({
        "asr": {"model": "fake-asr"},
        "pipeline": {"gap_rescue": {"enabled": True}},
    })

    assert pipeline.enable_gap_rescue is True


def test_gap_rescue_can_be_enabled_from_yaml(tmp_path, monkeypatch):
    _patch_asr_model(monkeypatch)
    config_path = tmp_path / "meeting.yaml"
    config_path.write_text(
        "asr:\n"
        "  model: fake-asr\n"
        "pipeline:\n"
        "  gap_rescue:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    pipeline = AutoPipeline.from_yaml(str(config_path))

    assert pipeline.enable_gap_rescue is True
