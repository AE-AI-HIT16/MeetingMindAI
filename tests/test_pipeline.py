"""Tests for MeetPipeline transcription flow."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from meetasr.pipeline import MeetPipeline
from meetasr.schemas import Segment, SentenceInfo, TranscriptResult


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


class ShortChunkVAD(FakeVAD):
    """VAD configured to keep chunks at or below 30 seconds."""

    max_segment_ms = 30000

    def detect(self, audio):
        self.detected_audio = audio
        return [Segment(0, 60010)]


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


class TimestampedFakeASR(FakeASR):
    """Fake ASR returning chunk-relative character timestamps."""

    def recognize(self, chunks, **kwargs):
        self.chunks = chunks
        self.kwargs = kwargs
        return [
            {"text": "xin", "timestamp": [[0, 100], [100, 200], [200, 300]]},
            {"text": "hop", "timestamp": [[0, 100], [100, 200], [200, 300]]},
        ]


class LongFormFakeASR:
    """ASR with native VAD and punctuation, like Faster-Whisper."""

    uses_internal_vad = True
    has_native_punctuation = True

    def __init__(self):
        self.audio = None
        self.kwargs = None

    def recognize_long_form(self, audio, **kwargs):
        self.audio = audio
        self.kwargs = kwargs
        return [{
            "text": "xin chao.",
            "timestamp": [[1000, 1100]] * 9,
        }]

    def recognize(self, *args, **kwargs):
        raise AssertionError("Long-form ASR must not receive VAD chunks")


class GapRescueVAD(FakeVAD):
    def detect(self, audio):
        self.detected_audio = audio
        return [Segment(2000, 4500)]


class GapRescueFakeASR(LongFormFakeASR):
    def __init__(self):
        super().__init__()
        self.calls = []

    def recognize_long_form(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if len(self.calls) == 1:
            return [
                {"text": "mot", "timestamp": [[0, 1000]] * 3},
                {"text": "ba", "timestamp": [[5000, 6000]] * 2},
            ]
        return [{
            "text": "giua them",
            "timestamp": [
                [0, 200], [200, 400], [400, 600], [600, 800],
                [4200, 4300], [4300, 4400], [4400, 4500], [4500, 4600],
                [4600, 4800],
            ],
        }]


class RecordingPunc:
    def __init__(self):
        self.calls = []

    def restore(self, text):
        self.calls.append(text)
        return text


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


def test_transcribe_offsets_word_timestamps_from_padded_chunk_start():
    audio = np.zeros(4 * 16000, dtype=np.float32)
    pipeline = MeetPipeline(asr_model=TimestampedFakeASR(), vad_model=FakeVAD())

    result = pipeline.transcribe(audio, key="sample_vi", language="vi")

    # First VAD segment begins at 500 ms but the ASR chunk begins at 400 ms
    # after 100 ms left padding. Whisper timestamps are relative to that chunk.
    assert result.sentence_info[0].char_timestamps == [
        [400, 500],
        [500, 600],
        [600, 700],
    ]
    assert (result.sentence_info[0].start, result.sentence_info[0].end) == (0.4, 0.7)


def test_incremental_segment_keeps_full_file_timestamp_offset():
    audio = np.zeros(4 * 16000, dtype=np.float32)
    asr = TimestampedFakeASR()
    pipeline = MeetPipeline(asr_model=asr, vad_model=FakeVAD())
    prepared_audio, segments, duration_ms = (
        pipeline.prepare_incremental_transcription(audio)
    )

    sentences = pipeline.transcribe_vad_segment(
        prepared_audio,
        segments[1],
        language="vi",
    )

    assert duration_ms == 4000
    # VAD starts at 2200 ms; the 100 ms padded chunk starts at 2100 ms.
    assert sentences[0].char_timestamps == [
        [2100, 2200],
        [2200, 2300],
        [2300, 2400],
    ]
    assert (sentences[0].start, sentences[0].end) == (2.1, 2.4)
    assert len(asr.chunks) == 1


def test_incremental_long_form_model_receives_one_vad_chunk_with_global_offset():
    audio = np.zeros(4 * 16000, dtype=np.float32)
    asr = LongFormFakeASR()
    pipeline = MeetPipeline(asr_model=asr, vad_model=FakeVAD())
    prepared_audio, segments, _ = pipeline.prepare_incremental_transcription(
        audio
    )

    sentences = pipeline.transcribe_vad_segment(
        prepared_audio,
        segments[1],
        language="vi",
    )

    assert len(asr.audio) == int(1.4 * 16000)
    assert sentences[0].char_timestamps == [[3100, 3200]] * 9
    assert (sentences[0].start, sentences[0].end) == (3.1, 3.2)


def test_incremental_finalize_preserves_sentence_count_and_assigns_speakers():
    pipeline = MeetPipeline(
        asr_model=FakeASR(),
        spk_model=object(),
    )
    original = [
        SentenceInfo(text="câu một", start=0.0, end=2.0),
        SentenceInfo(text="câu hai", start=2.0, end=4.0),
    ]

    pipeline._run_spk = lambda audio, sentences, segments: [
        SentenceInfo(text="câu", start=0.0, end=1.0, speaker=0),
        SentenceInfo(text="một", start=1.0, end=2.0, speaker=1),
        SentenceInfo(text="câu hai", start=2.0, end=4.0, speaker=1),
    ]

    finalized = pipeline.finalize_incremental_transcript(
        np.zeros(4 * 16000, dtype=np.float32),
        original,
        [Segment(0, 4000)],
    )

    assert len(finalized) == len(original)
    assert [sentence.speaker for sentence in finalized] == [0, 1]
    assert [sentence.text for sentence in finalized] == ["câu một", "câu hai"]


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


def test_run_vad_respects_model_max_segment_duration():
    vad = ShortChunkVAD()
    pipeline = MeetPipeline(asr_model=FakeASR(), vad_model=vad)
    audio = np.zeros(60 * 16000, dtype=np.float32)

    segments = pipeline._run_vad(audio)

    assert [(segment.start_ms, segment.end_ms) for segment in segments] == [
        (0, 30000),
        (30000, 60010),
    ]


def test_transcribe_uses_whisper_long_form_without_external_punctuation():
    audio = np.zeros(4 * 16000, dtype=np.float32)
    vad = FakeVAD()
    asr = LongFormFakeASR()
    punc = RecordingPunc()
    pipeline = MeetPipeline(asr_model=asr, vad_model=vad, punc_model=punc)

    result = pipeline.transcribe(audio, key="long_form", language="vi")

    assert np.array_equal(asr.audio, audio)
    assert asr.kwargs == {"language": "vi"}
    assert np.array_equal(vad.detected_audio, audio)
    assert punc.calls == []
    assert result.text == "xin chao."
    assert result.sentence_info[0].char_timestamps == [[1000, 1100]] * 9
    assert (result.sentence_info[0].start, result.sentence_info[0].end) == (1.0, 1.1)


def test_transcribe_rescues_only_vad_confirmed_gap_and_clips_context():
    audio = np.zeros(6 * 16000, dtype=np.float32)
    asr = GapRescueFakeASR()
    pipeline = MeetPipeline(
        asr_model=asr,
        vad_model=GapRescueVAD(),
        enable_gap_rescue=True,
    )

    result = pipeline.transcribe(audio, key="gap_rescue", language="vi")

    assert [sentence.text for sentence in result.sentence_info] == [
        "mot", "giua", "ba",
    ]
    assert len(asr.calls) == 2
    assert len(asr.calls[1][0]) == 5 * 16000
    assert asr.calls[1][1]["key"] == "gap_rescue_0"
