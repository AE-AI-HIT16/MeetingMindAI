"""Tests for LLM-facing transcript export."""

from __future__ import annotations

import json

from meetasr.schemas import SentenceInfo, TranscriptResult
from meetasr.utils.llm_export import transcript_to_llm_payload


def test_transcript_to_llm_payload_matches_required_contract():
    transcript = TranscriptResult(
        key="022878",
        text="xin chao hom nay hop",
        duration=9.31,
        sentence_info=[
            SentenceInfo(text="xin chao", start=0.0, end=3.2, speaker=0),
            SentenceInfo(text="hom nay hop", start=3.5, end=9.0, speaker=1),
        ],
    )

    payload = transcript_to_llm_payload(transcript)

    assert payload == {
        "key": "022878",
        "text": "xin chao hom nay hop",
        "duration": 9.31,
        "sentence_info": [
            {
                "text": "xin chao",
                "start": 0.0,
                "end": 3.2,
                "speaker": "Speaker 0",
            },
            {
                "text": "hom nay hop",
                "start": 3.5,
                "end": 9.0,
                "speaker": "Speaker 1",
            },
        ],
    }


def test_transcript_to_llm_payload_uses_default_speaker_when_missing():
    transcript = TranscriptResult(
        key="sample",
        text="alo",
        duration=1.0,
        sentence_info=[SentenceInfo(text="alo", start=0.0, end=1.0)],
    )

    payload = transcript_to_llm_payload(transcript, default_speaker="unknown")

    assert payload["sentence_info"][0]["speaker"] == "unknown"


def test_transcript_to_llm_payload_omits_internal_fields():
    transcript = TranscriptResult(
        key="sample",
        text="alo",
        duration=1.0,
        sentence_info=[
            SentenceInfo(
                text="alo",
                start=0.0,
                end=1.0,
                speaker=0,
                char_timestamps=[[0, 100]],
            )
        ],
    )

    payload = transcript_to_llm_payload(transcript)
    json.dumps(payload, ensure_ascii=False)

    sentence = payload["sentence_info"][0]
    assert "char_timestamps" not in sentence
    assert set(sentence) == {"text", "start", "end", "speaker"}
