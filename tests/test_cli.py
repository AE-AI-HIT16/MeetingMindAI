"""Tests for the transcription CLI output contract."""

from __future__ import annotations

import json
import os
from argparse import Namespace

from meetasr.bin.cli import cmd_server, cmd_transcribe
from meetasr.schemas import SentenceInfo, TranscriptResult


class _FakePipeline:
    def transcribe(self, audio_path, language):
        assert audio_path == "meeting.wav"
        assert language == "vi"
        return TranscriptResult(
            key="meeting",
            text="xin chao cuoc hop",
            duration=2.5,
            language="vi",
            sentence_info=[
                SentenceInfo(
                    text="xin chao cuoc hop",
                    start=0.2,
                    end=2.3,
                    speaker=1,
                    char_timestamps=[[200, 300]],
                )
            ],
        )


def _args(output_dir=None):
    return Namespace(
        audio=["meeting.wav"],
        config="meeting.yaml",
        language="vi",
        output_format="json",
        output_dir=output_dir,
    )


def test_transcribe_json_prints_llm_contract(monkeypatch, capsys):
    from meetasr.auto.auto_pipeline import AutoPipeline

    monkeypatch.setattr(AutoPipeline, "from_yaml", lambda path: _FakePipeline())
    monkeypatch.setattr("meetasr.bin.cli.os.path.exists", lambda path: True)

    cmd_transcribe(_args())

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"key", "text", "duration", "sentence_info"}
    assert payload["sentence_info"] == [
        {
            "text": "xin chao cuoc hop",
            "start": 0.2,
            "end": 2.3,
            "speaker": "Speaker 1",
        }
    ]


def test_transcribe_json_writes_output_file(monkeypatch, tmp_path, capsys):
    from meetasr.auto.auto_pipeline import AutoPipeline

    monkeypatch.setattr(AutoPipeline, "from_yaml", lambda path: _FakePipeline())
    monkeypatch.setattr("meetasr.bin.cli.os.path.exists", lambda path: True)

    cmd_transcribe(_args(output_dir=str(tmp_path)))

    payload = json.loads((tmp_path / "meeting.json").read_text(encoding="utf-8"))
    captured = capsys.readouterr()
    assert payload["key"] == "meeting"
    assert "Saved JSON:" in captured.err
    assert captured.out == ""


def test_server_explicit_config_overrides_environment(monkeypatch):
    captured = {}
    monkeypatch.setenv("MEETASR_CONFIG", "old.yaml")
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: captured.update(kwargs))

    cmd_server(
        Namespace(
            config="chosen.yaml",
            host="127.0.0.1",
            port=8123,
            reload=False,
        )
    )

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert os.environ["MEETASR_CONFIG"] == "chosen.yaml"
