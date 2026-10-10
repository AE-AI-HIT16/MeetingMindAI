"""Overlap-aware diarization helpers of the RunPod pipeline."""

from __future__ import annotations

import numpy as np

from meetasr.runpod.schemas import SpeakerTurn
from meetasr.runpod.utils.diarization import annotate_overlaps, build_speaker_turns
from meetasr.runpod.utils.overlap import (
    FRAME_STEP_S,
    LocalSegmentation,
    activity_to_segments,
    frame_of,
    overlap_aware_activity,
    speech_ratio,
)


def _segmentation(local: np.ndarray, count: np.ndarray) -> LocalSegmentation:
    return LocalSegmentation(activity=local[None], chunk_offsets=[0], count=count)


def test_second_voice_is_added_only_where_two_people_talk():
    frames = 600
    local = np.zeros((frames, 3), dtype=np.float32)
    local[:400, 0] = 1  # local speaker 1 talks first
    local[200:300, 1] = 1  # local speaker 2 overlaps
    local[400:, 1] = 1  # then speaker 2 alone
    count = local.sum(axis=1).astype(np.int64)
    clusters = [
        [0.0, 400 * FRAME_STEP_S, 0],
        [400 * FRAME_STEP_S, frames * FRAME_STEP_S, 1],
    ]

    binary = overlap_aware_activity(clusters, _segmentation(local, count), boundary_frames=0)

    assert binary[:200].sum(axis=0).tolist() == [200, 0]
    assert binary[200:300].sum(axis=0).tolist() == [100, 100]  # overlap found
    assert binary[450:].sum(axis=0).tolist() == [0, 150]


def test_no_speaker_where_segmentation_hears_nobody_unless_supplemented():
    local = np.zeros((300, 3), dtype=np.float32)
    local[:100, 0] = 1
    count = local.sum(axis=1).astype(np.int64)
    clusters = [[0.0, 300 * FRAME_STEP_S, 0]]  # VAD/cam++ think it is all speech

    gated = overlap_aware_activity(clusters, _segmentation(local, count), supplement=False)
    filled = overlap_aware_activity(clusters, _segmentation(local, count), supplement=True)

    assert gated[100:].sum() == 0
    assert filled[100:].sum() == 200


def test_activity_to_segments_closes_short_gaps_and_drops_blips():
    binary = np.zeros((300, 2), dtype=np.float32)
    binary[0:100, 0] = 1
    binary[105:200, 0] = 1  # 5-frame gap (~84 ms) is closed
    binary[250:253, 1] = 1  # 3-frame blip is dropped

    segments = activity_to_segments(binary, min_duration_s=0.1, fill_gap_s=0.2)

    assert len(segments) == 1
    start, end, speaker = segments[0]
    assert speaker == 0 and start == 0.0
    assert abs(end - 200 * FRAME_STEP_S) < 1e-3


def test_speech_ratio():
    count = np.zeros(1000, dtype=np.int64)
    count[frame_of(1.0):frame_of(2.0)] = 1
    segmentation = LocalSegmentation(np.zeros((1, 589, 3)), [0], count)
    assert abs(speech_ratio(segmentation, 1.0, 2.0) - 1.0) < 0.02
    assert abs(speech_ratio(segmentation, 0.0, 2.0) - 0.5) < 0.02
    assert speech_ratio(segmentation, 3.0, 4.0) == 0.0


def test_turns_merge_per_speaker_across_an_interjection():
    audio = np.zeros(16000 * 10, dtype=np.float32)
    diar = [
        [0.0, 4.0, 0],
        [3.0, 3.5, 1],  # "dạ" on top of speaker 0
        [4.1, 6.0, 0],  # speaker 0 continues after a 100 ms pause
    ]

    turns = build_speaker_turns(audio, diar)

    assert [(t.start_ms, t.end_ms, t.speaker) for t in turns] == [
        (0, 6000, 0),
        (3000, 3500, 1),
    ]


def test_annotate_overlaps_marks_ranges_shared_with_other_speakers():
    diar = [[0.0, 4.0, 5], [3.0, 3.5, 7], [6.0, 7.0, 7]]
    turns = [SpeakerTurn(0, 4000, 0), SpeakerTurn(3000, 3500, 1), SpeakerTurn(6000, 7000, 1)]

    annotate_overlaps(turns, diar)

    assert turns[0].overlaps == [(3000, 3500)]
    assert turns[1].overlaps == [(3000, 3500)]
    assert turns[2].overlaps == []


def test_overlap_regions_and_assignment_to_turns():
    from meetasr.runpod.utils.diarization import assign_overlap_regions
    from meetasr.runpod.utils.overlap import overlap_regions

    count = np.ones(1000, dtype=np.int64)
    count[frame_of(2.0):frame_of(3.0)] = 2   # 1 s of crosstalk
    count[frame_of(5.0):frame_of(5.1)] = 2   # 100 ms blip is ignored
    segmentation = LocalSegmentation(np.zeros((1, 589, 3)), [0], count)

    regions = overlap_regions(segmentation, min_duration_s=0.3)
    assert len(regions) == 1
    assert abs(regions[0][0] - 2000) < 20 and abs(regions[0][1] - 3000) < 20

    turns = [SpeakerTurn(0, 2500, 0), SpeakerTurn(2500, 4000, 1), SpeakerTurn(4000, 9000, 0)]
    assign_overlap_regions(turns, regions)
    assert turns[0].overlaps == [(regions[0][0], 2500)]
    assert turns[1].overlaps == [(2500, regions[0][1])]
    assert turns[2].overlaps == []
