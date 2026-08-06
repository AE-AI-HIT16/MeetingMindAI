"""Export helpers for LLM-facing transcript payloads."""

from __future__ import annotations

from meetasr.schemas import SentenceInfo, TranscriptResult


def transcript_to_llm_payload(
    transcript: TranscriptResult,
    default_speaker: str = "Speaker 0",
) -> dict:
    """Convert a TranscriptResult to the JSON contract expected by the LLM side."""
    return {
        "key": transcript.key,
        "text": transcript.text,
        "duration": transcript.duration,
        "sentence_info": [
            _sentence_to_llm_item(sentence, default_speaker)
            for sentence in transcript.sentence_info
        ],
    }


def _sentence_to_llm_item(sentence: SentenceInfo, default_speaker: str) -> dict:
    speaker = default_speaker
    if sentence.speaker is not None:
        speaker = f"Speaker {sentence.speaker}"

    return {
        "text": sentence.text,
        "start": sentence.start,
        "end": sentence.end,
        "speaker": speaker,
    }
