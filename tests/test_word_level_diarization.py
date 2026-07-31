"""Unit tests for word-level speaker diarization functions.

Tests cover:
    - map_chars_to_speakers: char→speaker mapping via max-overlap
    - _fill_none_gaps: forward+backward fill of None gaps
    - _merge_short_groups: merging groups shorter than min_chars
    - split_at_speaker_turns: splitting sentences at speaker boundaries
"""

import numpy as np
import pytest

from meetasr.schemas import SentenceInfo
from meetasr.utils.diarization import (
    _fill_none_gaps,
    _merge_short_groups,
    build_speaker_turns,
    chunk_segment,
    map_chars_to_speakers,
    split_at_speaker_turns,
)


def test_chunk_segment_keeps_short_backchannel():
    assert chunk_segment(1.0, 1.4) == [[1.0, 1.4]]


def test_build_speaker_turns_splits_long_turn_at_quiet_audio():
    audio = np.ones(36 * 16000, dtype=np.float32)
    audio[int(13.8 * 16000):int(14.2 * 16000)] = 0.0

    turns = build_speaker_turns(
        audio,
        [
            [0.0, 32.0, 7],
            [32.0, 36.0, 9],
        ],
        max_chunk_ms=15000,
        boundary_search_ms=2000,
        min_chunk_ms=1000,
    )

    assert turns[0].speaker == 0
    assert 13700 <= turns[0].end_ms <= 14300
    assert turns[1].start_ms == turns[0].end_ms
    assert turns[-1].speaker == 1
    assert turns[-1].start_ms == 32000
    assert turns[-1].end_ms == 36000


def test_build_speaker_turns_prefers_target_when_energy_is_equal():
    turns = build_speaker_turns(
        np.zeros(20 * 16000, dtype=np.float32),
        [[0.0, 20.0, 4]],
        max_chunk_ms=15000,
        boundary_search_ms=2000,
        min_chunk_ms=1000,
    )

    assert [(turn.start_ms, turn.end_ms) for turn in turns] == [
        (0, 15000),
        (15000, 20000),
    ]


def test_build_speaker_turns_coalesces_small_same_speaker_gap():
    turns = build_speaker_turns(
        np.zeros(4 * 16000, dtype=np.float32),
        [
            [0.0, 1.0, 3],
            [1.2, 3.0, 3],
            [3.0, 4.0, 8],
        ],
    )

    assert [
        (turn.start_ms, turn.end_ms, turn.speaker)
        for turn in turns
    ] == [
        (0, 3000, 0),
        (3000, 4000, 1),
    ]


# ======================================================================
# map_chars_to_speakers
# ======================================================================

class TestMapCharsToSpeakers:
    """Tests for map_chars_to_speakers()."""

    def test_basic_two_speakers(self):
        """2 speakers, clear boundary at 1.0s."""
        char_ts = [[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]]
        diar = [[0.0, 1.0, 0], [1.0, 3.0, 1]]
        result = map_chars_to_speakers(char_ts, diar)
        assert result == [0, 0, 1, 1, 1]

    def test_no_overlap_returns_none(self):
        """Char outside all diar segments → None."""
        char_ts = [[5000, 5500]]
        diar = [[0.0, 1.0, 0]]
        result = map_chars_to_speakers(char_ts, diar)
        assert result == [None]

    def test_single_speaker(self):
        """All chars belong to one speaker."""
        char_ts = [[0, 200], [200, 400], [400, 600]]
        diar = [[0.0, 1.0, 0]]
        result = map_chars_to_speakers(char_ts, diar)
        assert result == [0, 0, 0]

    def test_char_on_boundary(self):
        """Char spans exactly across speaker boundary — max-overlap wins."""
        # Char: 0.8s → 1.2s, Speaker 0: 0→1s, Speaker 1: 1→2s
        # Overlap with spk0 = 0.2s, overlap with spk1 = 0.2s → first match wins (spk0)
        # Actually both equal, but code uses > so first (spk0) stays as best
        char_ts = [[800, 1200]]
        diar = [[0.0, 1.0, 0], [1.0, 2.0, 1]]
        result = map_chars_to_speakers(char_ts, diar)
        # Equal overlap → first speaker wins (no strict > for second)
        assert result[0] in (0, 1)

    def test_char_mostly_in_second_speaker(self):
        """Char spans boundary but mostly in speaker 1."""
        # Char: 0.9s → 1.5s, spk0: 0→1s (overlap=0.1s), spk1: 1→2s (overlap=0.5s)
        char_ts = [[900, 1500]]
        diar = [[0.0, 1.0, 0], [1.0, 2.0, 1]]
        result = map_chars_to_speakers(char_ts, diar)
        assert result == [1]

    def test_three_speakers(self):
        """Three speakers with clear boundaries."""
        char_ts = [
            [0, 300], [300, 600],       # spk 0
            [1000, 1300], [1300, 1600],  # spk 1
            [2000, 2300], [2300, 2600],  # spk 2
        ]
        diar = [[0.0, 0.8, 0], [1.0, 1.8, 1], [2.0, 2.8, 2]]
        result = map_chars_to_speakers(char_ts, diar)
        assert result == [0, 0, 1, 1, 2, 2]

    def test_empty_char_timestamps(self):
        """Empty input → empty output."""
        result = map_chars_to_speakers([], [[0.0, 1.0, 0]])
        assert result == []

    def test_empty_diar_segs(self):
        """No diar segments → all None."""
        char_ts = [[0, 500], [500, 1000]]
        result = map_chars_to_speakers(char_ts, [])
        assert result == [None, None]

    def test_gap_between_diar_segments(self):
        """Char falls in gap between 2 diar segments → None."""
        char_ts = [[1500, 2000]]  # 1.5s-2.0s, gap between segs
        diar = [[0.0, 1.0, 0], [3.0, 4.0, 1]]
        result = map_chars_to_speakers(char_ts, diar)
        assert result == [None]


# ======================================================================
# _fill_none_gaps
# ======================================================================

class TestFillNoneGaps:
    """Tests for _fill_none_gaps()."""

    def test_basic_fill(self):
        """Forward and backward fill."""
        assert _fill_none_gaps([None, None, 0, 0, None, 1, None]) == [0, 0, 0, 0, 0, 1, 1]

    def test_all_none(self):
        """All None → stays all None (no valid speaker to fill from)."""
        assert _fill_none_gaps([None, None, None]) == [None, None, None]

    def test_no_none(self):
        """No gaps → unchanged."""
        assert _fill_none_gaps([0, 1, 0, 1]) == [0, 1, 0, 1]

    def test_leading_none(self):
        """None at start → backward fill from first valid."""
        assert _fill_none_gaps([None, None, 1, 1]) == [1, 1, 1, 1]

    def test_trailing_none(self):
        """None at end → forward fill from last valid."""
        assert _fill_none_gaps([0, 0, None, None]) == [0, 0, 0, 0]

    def test_single_valid(self):
        """Only one valid value fills everything."""
        assert _fill_none_gaps([None, None, 2, None, None]) == [2, 2, 2, 2, 2]

    def test_single_element_valid(self):
        assert _fill_none_gaps([5]) == [5]

    def test_single_element_none(self):
        assert _fill_none_gaps([None]) == [None]

    def test_empty(self):
        assert _fill_none_gaps([]) == []


# ======================================================================
# _merge_short_groups
# ======================================================================

class TestMergeShortGroups:
    """Tests for _merge_short_groups()."""

    def test_no_short_groups(self):
        """All groups >= min_chars → no merging."""
        groups = [(0, 5, 0), (5, 10, 1)]
        assert _merge_short_groups(groups, min_chars=3) == [(0, 5, 0), (5, 10, 1)]

    def test_merge_short_into_previous(self):
        """Short group (2 chars) merged into previous."""
        groups = [(0, 5, 0), (5, 7, 1), (7, 12, 0)]
        result = _merge_short_groups(groups, min_chars=3)
        # (5,7,1) is 2 chars < 3 → merge into (0,5,0) → (0,7,0)
        assert result == [(0, 7, 0), (7, 12, 0)]

    def test_single_group(self):
        """Single group → no merging possible."""
        groups = [(0, 2, 0)]
        assert _merge_short_groups(groups, min_chars=3) == [(0, 2, 0)]

    def test_empty_groups(self):
        assert _merge_short_groups([], min_chars=3) == []

    def test_all_short(self):
        """All groups are short → all merge into first."""
        groups = [(0, 2, 0), (2, 4, 1), (4, 5, 0)]
        result = _merge_short_groups(groups, min_chars=3)
        # (0,2,0) first; (2,4,1) 2<3 → merge → (0,4,0); (4,5,0) 1<3 → merge → (0,5,0)
        assert result == [(0, 5, 0)]

    def test_min_chars_1(self):
        """min_chars=1 → nothing is short (all groups have length>=1)."""
        groups = [(0, 1, 0), (1, 2, 1)]
        result = _merge_short_groups(groups, min_chars=1)
        assert result == [(0, 1, 0), (1, 2, 1)]

    def test_consecutive_short_groups(self):
        """Multiple consecutive short groups merge sequentially."""
        groups = [(0, 5, 0), (5, 6, 1), (6, 7, 0), (7, 12, 1)]
        result = _merge_short_groups(groups, min_chars=3)
        # (5,6,1) 1<3 → merge into (0,5,0) → (0,6,0)
        # (6,7,0) 1<3 → merge into (0,6,0) → (0,7,0)
        # (7,12,1) 5>=3 → keep
        assert result == [(0, 7, 0), (7, 12, 1)]


# ======================================================================
# split_at_speaker_turns
# ======================================================================

class TestSplitAtSpeakerTurns:
    """Tests for split_at_speaker_turns()."""

    def test_basic_two_speakers(self):
        """5 chars, speaker changes at index 2 → 2 sub-sentences."""
        sent = SentenceInfo(
            text="AABBB",
            start=0.0,
            end=2.5,
            char_timestamps=[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
        )
        char_spks = [0, 0, 1, 1, 1]
        subs = split_at_speaker_turns(sent, char_spks)
        assert len(subs) == 2
        assert subs[0].speaker == 0
        assert subs[0].text == "AA"
        assert subs[1].speaker == 1
        assert subs[1].text == "BBB"

    def test_timing_preserved(self):
        """Sub-sentence start/end computed from char_timestamps."""
        sent = SentenceInfo(
            text="AABBB",
            start=0.0,
            end=2.5,
            char_timestamps=[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
        )
        char_spks = [0, 0, 1, 1, 1]
        subs = split_at_speaker_turns(sent, char_spks)
        assert subs[0].start == pytest.approx(0.0)
        assert subs[0].end == pytest.approx(1.0)
        assert subs[1].start == pytest.approx(1.0)
        assert subs[1].end == pytest.approx(2.5)

    def test_char_timestamps_preserved(self):
        """Each sub-sentence gets correct slice of char_timestamps."""
        sent = SentenceInfo(
            text="AABBB",
            start=0.0,
            end=2.5,
            char_timestamps=[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
        )
        char_spks = [0, 0, 1, 1, 1]
        subs = split_at_speaker_turns(sent, char_spks)
        assert subs[0].char_timestamps == [[0, 500], [500, 1000]]
        assert subs[1].char_timestamps == [[1000, 1500], [1500, 2000], [2000, 2500]]

    def test_fallback_no_timestamps(self):
        """No char_timestamps → return original sentence."""
        sent = SentenceInfo(text="hello", start=0.0, end=1.0)
        subs = split_at_speaker_turns(sent, [])
        assert len(subs) == 1
        assert subs[0] is sent

    def test_fallback_empty_char_speakers(self):
        """Empty char_speakers list → return original."""
        sent = SentenceInfo(
            text="hello",
            start=0.0,
            end=1.0,
            char_timestamps=[[0, 200], [200, 400], [400, 600], [600, 800], [800, 1000]],
        )
        subs = split_at_speaker_turns(sent, [])
        assert len(subs) == 1

    def test_all_none_speakers(self):
        """All char_speakers are None → return original."""
        sent = SentenceInfo(
            text="hello",
            start=0.0,
            end=1.0,
            char_timestamps=[[0, 200], [200, 400], [400, 600], [600, 800], [800, 1000]],
        )
        subs = split_at_speaker_turns(sent, [None, None, None, None, None])
        assert len(subs) == 1
        assert subs[0] is sent

    def test_single_speaker_no_split(self):
        """All chars same speaker → 1 sub-sentence."""
        sent = SentenceInfo(
            text="AAAAA",
            start=0.0,
            end=2.5,
            char_timestamps=[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
        )
        char_spks = [0, 0, 0, 0, 0]
        subs = split_at_speaker_turns(sent, char_spks)
        assert len(subs) == 1
        assert subs[0].speaker == 0

    def test_merge_short_group(self):
        """Short group (1 char) at speaker boundary gets merged."""
        sent = SentenceInfo(
            text="AAAABBBBB",
            start=0.0,
            end=4.5,
            char_timestamps=[
                [0, 500], [500, 1000], [1000, 1500], [1500, 2000],     # 4 chars
                [2000, 2500], [2500, 3000], [3000, 3500], [3500, 4000], [4000, 4500],  # 5 chars
            ],
        )
        # 1 noisy char at index 3 classified as spk 1, but only 1 char → merged
        char_spks = [0, 0, 0, 1, 1, 1, 1, 1, 1]
        subs = split_at_speaker_turns(sent, char_spks, min_chars=3)
        # Group (0,3,0), (3,4,1) short→merge → (0,4,0), then (4,9,1)
        assert len(subs) == 2
        assert subs[0].speaker == 0
        assert subs[1].speaker == 1

    def test_none_gaps_filled(self):
        """None gaps between speakers get filled before splitting."""
        sent = SentenceInfo(
            text="AAAAABBBBB",
            start=0.0,
            end=5.0,
            char_timestamps=[
                [0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500],
                [2500, 3000], [3000, 3500], [3500, 4000], [4000, 4500], [4500, 5000],
            ],
        )
        # None in the middle gets filled
        char_spks = [0, 0, 0, None, None, 1, 1, 1, 1, 1]
        subs = split_at_speaker_turns(sent, char_spks)
        # After fill: [0,0,0,0,0,1,1,1,1,1] → 2 groups
        assert len(subs) == 2
        assert subs[0].speaker == 0
        assert subs[1].speaker == 1

    def test_three_speakers(self):
        """Three speakers → 3 sub-sentences."""
        sent = SentenceInfo(
            text="AAABBBBCCC",
            start=0.0,
            end=5.0,
            char_timestamps=[
                [0, 500], [500, 1000], [1000, 1500],
                [1500, 2000], [2000, 2500], [2500, 3000], [3000, 3500],
                [3500, 4000], [4000, 4500], [4500, 5000],
            ],
        )
        char_spks = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        subs = split_at_speaker_turns(sent, char_spks)
        assert len(subs) == 3
        assert [s.speaker for s in subs] == [0, 1, 2]
        assert [s.text for s in subs] == ["AAA", "BBBB", "CCC"]


# ======================================================================
# Integration: map_chars_to_speakers + split_at_speaker_turns
# ======================================================================

class TestIntegration:
    """End-to-end tests combining mapping and splitting."""

    def test_full_pipeline(self):
        """map → split produces correct sub-sentences."""
        sent = SentenceInfo(
            text="Hello World",
            start=0.0,
            end=5.5,
            char_timestamps=[
                [0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500],   # "Hello"
                [2500, 3000],                                                        # " "
                [3000, 3500], [3500, 4000], [4000, 4500], [4500, 5000], [5000, 5500], # "World"
            ],
        )
        diar = [[0.0, 2.8, 0], [2.8, 6.0, 1]]

        char_spks = map_chars_to_speakers(sent.char_timestamps, diar)
        subs = split_at_speaker_turns(sent, char_spks)

        # "Hello" (0-2.5s) → spk 0, " World" (2.5-5.5s) → spk 1
        assert len(subs) == 2
        assert subs[0].speaker == 0
        assert subs[1].speaker == 1

    def test_full_pipeline_single_speaker(self):
        """All in one diar segment → no split."""
        sent = SentenceInfo(
            text="ABCDE",
            start=0.0,
            end=2.5,
            char_timestamps=[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
        )
        diar = [[0.0, 3.0, 0]]

        char_spks = map_chars_to_speakers(sent.char_timestamps, diar)
        subs = split_at_speaker_turns(sent, char_spks)

        assert len(subs) == 1
        assert subs[0].speaker == 0

    def test_full_pipeline_no_diar_data(self):
        """Empty diar → all None → return original."""
        sent = SentenceInfo(
            text="ABCDE",
            start=0.0,
            end=2.5,
            char_timestamps=[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
        )
        char_spks = map_chars_to_speakers(sent.char_timestamps, [])
        subs = split_at_speaker_turns(sent, char_spks)

        assert len(subs) == 1
        assert subs[0] is sent
