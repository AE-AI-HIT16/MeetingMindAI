"""Tests for MeetPipeline transcription flow."""

from __future__ import annotations

import numpy as np

from meetasr.pipeline import MeetPipeline
from meetasr.schemas import Segment, TranscriptResult


class FakeVAD:
    """Fake VAD returning deterministic speech segments."""

    def __init__(self):
        self.detected_audio = None

    def detect(self, audio):
        self.detected_audio = audio
        return [
            Segment(500, 1500),
            Segment(2200, 3400),
        ]


class FakeASR:
    """Fake ASR that records chunks and returns one result per chunk."""

    def __init__(self):
        self.chunks = None
        self.kwargs = None

    def recognize(self, chunks, **kwargs):
        self.chunks = chunks
        self.kwargs = kwargs
        return [
            {"text": "xin chao", "timestamp": []},
            {"text": "hom nay hop", "timestamp": []},
        ]


class FakePunc:
    """Fake punctuation model returning multiple sentences for the first chunk."""

    def restore(self, text):
        if text == "xin chao":
            return "Xin chào. Mình họp nhé."
        return "Hôm nay họp."


def test_transcribe_builds_transcript_from_vad_and_asr_segments():
    audio = np.zeros(4 * 16000, dtype=np.float32)
    vad = FakeVAD()
    asr = FakeASR()
    pipeline = MeetPipeline(asr_model=asr, vad_model=vad)

    result = pipeline.transcribe(audio, key="sample_vi", language="vi")

    assert isinstance(result, TranscriptResult)
    assert result.key == "sample_vi"
    assert result.duration == 4.0
    assert result.text == "xin chao hom nay hop"
    assert result.language == "vi"
    assert [s.text for s in result.sentence_info] == ["xin chao", "hom nay hop"]
    assert [(s.start, s.end) for s in result.sentence_info] == [(0.5, 1.5), (2.2, 3.4)]
    assert asr.kwargs == {"language": "vi"}
    assert [len(chunk) for chunk in asr.chunks] == [19200, 22400]
    assert vad.detected_audio.shape == audio.shape
    assert vad.detected_audio.dtype == np.float32


def test_transcribe_splits_long_punctuated_segments():
    audio = np.zeros(4 * 16000, dtype=np.float32)
    vad = FakeVAD()
    asr = FakeASR()
    pipeline = MeetPipeline(asr_model=asr, vad_model=vad, punc_model=FakePunc())

    result = pipeline.transcribe(audio, key="sample_vi", language="vi")

    assert [s.text for s in result.sentence_info] == [
        "Xin chào. Mình họp nhé.",
        "Hôm nay họp.",
    ]

    vad.detect = lambda audio: [Segment(0, 6000)]
    asr.recognize = lambda chunks, **kwargs: [
        {"text": "xin chao", "timestamp": []},
    ]
    result = pipeline.transcribe(audio, key="sample_vi", language="vi")

    assert [s.text for s in result.sentence_info] == [
        "Xin chào.",
        "Mình họp nhé.",
    ]
    assert result.sentence_info[0].start == 0.0
    assert result.sentence_info[-1].end == 6.0
