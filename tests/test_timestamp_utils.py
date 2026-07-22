"""Tests for timestamp utilities."""

import pytest
from meetasr.utils.timestamp import (
    merge_vad_segments,
    clip_sentence_to_range,
    find_speech_gaps,
    merge_rescued_sentences,
    align_timestamps_to_global,
    build_sentence_info,
    split_punctuated_sentence_info,
)
from meetasr.schemas import Segment, SentenceInfo


class TestMergeVadSegments:

    def test_empty_input(self):
        assert merge_vad_segments([]) == []

    def test_single_segment_unchanged(self):
        segs = [Segment(0, 2000)]
        result = merge_vad_segments(segs)
        assert len(result) == 1
        assert result[0].start_ms == 0
        assert result[0].end_ms == 2000

    def test_merge_close_segments(self):
        """Segments with small gap should merge."""
        segs = [Segment(0, 1000), Segment(1200, 2500)]  # 200ms gap
        result = merge_vad_segments(segs, max_merge_gap_ms=300)
        assert len(result) == 1
        assert result[0].start_ms == 0
        assert result[0].end_ms == 2500

    def test_do_not_merge_distant_segments(self):
        """Segments with large gap should stay separate."""
        segs = [Segment(0, 1000), Segment(5000, 6000)]  # 4s gap
        result = merge_vad_segments(segs, max_merge_gap_ms=300)
        assert len(result) == 2

    def test_do_not_exceed_max_segment_length(self):
        """Even if gap is small, don't create segment > max_segment_ms."""
        segs = [Segment(0, 50000), Segment(50200, 60000)]
        result = merge_vad_segments(segs, max_merge_gap_ms=500, max_segment_ms=60000)
        # 0→60000 = 60000ms → exactly at limit, should NOT merge (would exceed)
        assert len(result) == 2


class TestAlignTimestamps:

    def test_basic_offset(self):
        ts = [[100, 300], [400, 600]]
        result = align_timestamps_to_global(ts, offset_ms=1000)
        assert result == [[1100, 1300], [1400, 1600]]

    def test_zero_offset_unchanged(self):
        ts = [[0, 500]]
        assert align_timestamps_to_global(ts, offset_ms=0) == [[0, 500]]


class TestFindSpeechGaps:

    def test_ignores_asr_gap_when_vad_confirms_silence(self):
        sentences = [
            SentenceInfo(text="Một", start=0.0, end=1.0),
            SentenceInfo(text="Hai", start=5.0, end=6.0),
        ]
        vad_segments = [Segment(0, 1000), Segment(5000, 6000)]

        assert find_speech_gaps(sentences, vad_segments, duration_ms=6000) == []

    def test_returns_asr_gap_when_vad_detects_speech_inside_it(self):
        sentences = [
            SentenceInfo(text="Một", start=0.0, end=1.0),
            SentenceInfo(text="Hai", start=5.0, end=6.0),
        ]
        vad_segments = [Segment(1800, 4500)]

        assert find_speech_gaps(sentences, vad_segments, duration_ms=6000) == [
            Segment(1000, 5000),
        ]

    def test_checks_leading_and_trailing_asr_gaps(self):
        sentences = [SentenceInfo(text="Giữa", start=2.0, end=4.0)]
        vad_segments = [Segment(300, 1500), Segment(4500, 5800)]

        assert find_speech_gaps(sentences, vad_segments, duration_ms=6000) == [
            Segment(0, 2000),
            Segment(4000, 6000),
        ]


class TestRescueSentenceUtilities:

    def test_clip_sentence_to_gap_removes_right_context(self):
        sentence = SentenceInfo(
            text="giua them",
            start=1.0,
            end=5.8,
            char_timestamps=[
                [1000, 1200], [1200, 1400], [1400, 1600], [1600, 1800],
                [5200, 5300], [5300, 5400], [5400, 5500], [5500, 5600],
                [5600, 5800],
            ],
        )

        clipped = clip_sentence_to_range(sentence, start_ms=1000, end_ms=5000)

        assert clipped is not None
        assert clipped.text == "giua"
        assert (clipped.start, clipped.end) == (1.0, 1.8)

    def test_merge_rescued_sentences_skips_overlap_and_duplicate(self):
        baseline = [
            SentenceInfo(text="Mot", start=0.0, end=1.0),
            SentenceInfo(text="Ba", start=5.0, end=6.0),
        ]
        rescued = [
            SentenceInfo(text="Giua", start=1.1, end=4.9),
            SentenceInfo(text="ba", start=5.1, end=5.8),
        ]

        merged = merge_rescued_sentences(baseline, rescued)

        assert [sentence.text for sentence in merged] == ["Mot", "Giua", "Ba"]

    def test_merge_rescued_sentences_trims_repeated_right_boundary_word(self):
        baseline = [SentenceInfo(text="Nha truong", start=5.0, end=6.0)]
        rescued = [
            SentenceInfo(
                text="Qua nha",
                start=1.0,
                end=4.9,
                char_timestamps=[[1000 + 100 * i, 1100 + 100 * i] for i in range(7)],
            ),
        ]

        merged = merge_rescued_sentences(baseline, rescued)

        assert [sentence.text for sentence in merged] == ["Qua", "Nha truong"]
        assert merged[0].end == 1.3


class TestBuildSentenceInfo:

    def test_basic_build(self):
        asr_results = [
            {"text": "Xin chào.", "timestamp": [[0, 200], [300, 500], [600, 900]]},
        ]
        segments = [Segment(1000, 4000)]
        sents = build_sentence_info(asr_results, segments)
        assert len(sents) == 1
        assert sents[0].text == "Xin chào."
        assert sents[0].char_timestamps[0][0] == 1000  # offset applied

    def test_build_uses_padded_chunk_offset_for_asr_timestamps(self):
        asr_results = [
            {"text": "Xin", "timestamp": [[0, 100], [100, 200], [200, 300]]},
        ]
        segments = [Segment(500, 1500)]

        sents = build_sentence_info(
            asr_results,
            segments,
            timestamp_offsets_ms=[400],
        )

        assert sents[0].start == 0.4
        assert sents[0].end == 0.7
        assert sents[0].char_timestamps == [[400, 500], [500, 600], [600, 700]]

    def test_empty_text_skipped(self):
        asr_results = [{"text": "", "timestamp": []}]
        segments = [Segment(0, 2000)]
        sents = build_sentence_info(asr_results, segments)
        assert len(sents) == 0


class TestSplitPunctuatedSentenceInfo:

    def test_split_long_segment_by_sentence_punctuation(self):
        sentence = SentenceInfo(
            text="Alo alo. Hôm nay mình họp. Chốt việc nhé.",
            start=0.0,
            end=10.0,
            speaker=2,
            char_timestamps=[[0, 100]],
        )

        result = split_punctuated_sentence_info([sentence])

        assert [s.text for s in result] == [
            "Alo alo.",
            "Hôm nay mình họp.",
            "Chốt việc nhé.",
        ]
        assert result[0].start == 0.0
        assert result[-1].end == 10.0
        assert all(s.speaker == 2 for s in result)
        assert all(s.char_timestamps == [] for s in result)
        assert all(s.start < s.end for s in result)

    def test_keep_short_segment_unchanged(self):
        sentence = SentenceInfo(
            text="Alo alo. Chốt nhé.",
            start=0.0,
            end=2.5,
        )

        result = split_punctuated_sentence_info([sentence])

        assert result == [sentence]

    def test_keep_single_sentence_unchanged(self):
        sentence = SentenceInfo(
            text="Alo alo chưa có nhiều câu.",
            start=0.0,
            end=10.0,
        )

        result = split_punctuated_sentence_info([sentence])

        assert result == [sentence]
