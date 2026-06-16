"""Tests for timestamp utilities."""

import pytest
from meetasr.utils.timestamp import (
    merge_vad_segments,
    align_timestamps_to_global,
    build_sentence_info,
)
from meetasr.schemas import Segment


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

    def test_empty_text_skipped(self):
        asr_results = [{"text": "", "timestamp": []}]
        segments = [Segment(0, 2000)]
        sents = build_sentence_info(asr_results, segments)
        assert len(sents) == 0
