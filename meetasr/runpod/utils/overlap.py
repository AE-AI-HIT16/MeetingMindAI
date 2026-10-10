"""Overlap-aware speaker activity: segmentation output + cam++ clusters.

The cam++ clustering gives one speaker per 1.5 s window, so it can neither see
two people talking at once nor place a speaker change precisely. Mapping the
segmentation model's local speakers onto the global cam++ clusters yields
frame-level, overlap-aware speaker activity.

Ported and adapted from 3D-Speaker ``infer_diarization.py``
(``--include_overlap``, Apache 2.0).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from meetasr.runpod.utils.segmentation import FRAME_STEP_S, LocalSegmentation, frame_of


def overlap_aware_activity(
    cluster_segments: list[list],
    segmentation: LocalSegmentation,
    *,
    supplement: bool = True,
    min_exclusive_frames: int = 12,
    boundary_frames: int = 45,
) -> np.ndarray:
    """Combine global clusters with local segmentation -> [frames, speakers].

    ``cluster_segments`` are ``[start_s, end_s, speaker]`` from cam++
    clustering (non-overlapping). Each 10 s chunk's local speakers are matched
    to global speakers (Hungarian) by co-activity; per frame the ``count``
    most active speakers are kept, which is how overlap appears.

    Unlike 3D-Speaker, co-activity is counted on frames where the local
    speaker talks alone: cam++ windows inside overlapped speech carry the
    louder voice's label and made whole chunks swap speakers. A local speaker
    heard only on top of someone else (a short "dạ", "vâng") falls back to
    all of its frames.

    Who speaks in single-speaker frames comes from cam++ (it tells voices
    apart better than the 10 s local model); segmentation decides whether
    anyone speaks, moves speaker changes within ``boundary_frames`` of a
    cam++ change to the right frame, and adds the second voice in overlap.

    Args:
        cluster_segments: ``[start_s, end_s, speaker]`` from cam++ clustering.
        segmentation: Segmentation model output for the same audio.
        supplement: Keep cam++ speech where segmentation hears nobody.
        min_exclusive_frames: Solo frames a local speaker needs before its
            mapping uses only those frames.
        boundary_frames: How far from a cam++ change segmentation may move it.

    Returns:
        Binary ``[frames, speakers]`` activity; rows may have two speakers.
    """
    num_speakers = max(int(seg[2]) for seg in cluster_segments) + 1
    num_frames = segmentation.num_frames
    cluster_frames = np.zeros((num_frames, num_speakers), dtype=np.float32)
    for start_s, end_s, speaker in cluster_segments:
        cluster_frames[frame_of(start_s):frame_of(end_s), int(speaker)] = 1.0

    activations = np.zeros((num_frames, num_speakers), dtype=np.float32)
    for offset, local in zip(segmentation.chunk_offsets, segmentation.activity):
        local = local[: max(0, num_frames - offset)]
        if local.size == 0:
            continue
        global_frames = cluster_frames[offset:offset + len(local)]
        exclusive = local * (local.sum(axis=1, keepdims=True) == 1)
        use_exclusive = exclusive.sum(axis=0) >= min_exclusive_frames
        basis = np.where(use_exclusive[None, :], exclusive, local)
        totals = basis.sum(axis=0)
        share = (basis.T @ global_frames) / np.maximum(totals, 1.0)[:, None]
        share[totals == 0] = -1.0
        rows, cols = linear_sum_assignment(-share)
        aligned = np.zeros((len(local), num_speakers), dtype=np.float32)
        for row, col in zip(rows, cols):
            if share[row, col] > 0:
                aligned[:, col] = np.maximum(aligned[:, col], local[:, row])
        activations[offset:offset + len(local)] += aligned

    binary = _select_speakers(
        activations,
        cluster_frames,
        segmentation.count,
        boundary_frames=boundary_frames,
    )

    if supplement:
        # Frames the clustering covered but segmentation left empty.
        empty = (binary.sum(axis=1) == 0) & (cluster_frames.sum(axis=1) > 0)
        binary[empty] = cluster_frames[empty]
    return binary


def _select_speakers(
    activations: np.ndarray,
    cluster_frames: np.ndarray,
    count: np.ndarray,
    *,
    boundary_frames: int,
) -> np.ndarray:
    num_frames, num_speakers = activations.shape
    cluster_label = np.where(
        cluster_frames.sum(axis=1) > 0, np.argmax(cluster_frames, axis=1), -1
    )
    # Distance (frames) to the nearest cam++ speaker change.
    changes = np.flatnonzero(np.diff(cluster_label) != 0) + 1
    near_change = np.zeros(num_frames, dtype=bool)
    for change in changes:
        near_change[max(0, change - boundary_frames):change + boundary_frames] = True

    ranked = np.argsort(-activations, axis=1)
    binary = np.zeros_like(activations)
    for frame in range(num_frames):
        wanted = min(num_speakers, int(count[frame]))
        if wanted == 0:
            continue
        top = ranked[frame, 0]
        primary = cluster_label[frame]
        if primary < 0 or (
            near_change[frame]
            and activations[frame, top] > activations[frame, primary]
        ):
            primary = top if activations[frame, top] > 0 else primary
        if primary < 0:
            continue
        binary[frame, primary] = 1.0
        added = 1
        for speaker in ranked[frame]:
            if added >= wanted:
                break
            if speaker != primary and activations[frame, speaker] > 0:
                binary[frame, speaker] = 1.0
                added += 1
    return binary


def activity_to_segments(
    binary: np.ndarray,
    *,
    min_duration_s: float = 0.0,
    fill_gap_s: float = 0.0,
) -> list[list]:
    """Frame activity -> sorted ``[start_s, end_s, speaker]`` (may overlap).

    Gaps shorter than ``fill_gap_s`` inside one speaker's activity are closed
    first, then pieces shorter than ``min_duration_s`` are dropped.

    Args:
        binary: ``[frames, speakers]`` activity from ``overlap_aware_activity``.
        min_duration_s: Shortest piece kept.
        fill_gap_s: Longest same-speaker gap closed.

    Returns:
        ``[start_s, end_s, speaker]`` lists sorted by start.
    """
    segments = []
    max_gap = int(round(fill_gap_s / FRAME_STEP_S))
    for speaker in range(binary.shape[1]):
        column = binary[:, speaker] > 0.5
        if max_gap > 0:
            column = _close_gaps(column, max_gap)
        start = None
        for frame, value in enumerate(column):
            if value and start is None:
                start = frame
            elif not value and start is not None:
                segments.append([start, frame, speaker])
                start = None
        if start is not None:
            segments.append([start, len(column), speaker])
    result = [
        [round(a * FRAME_STEP_S, 3), round(b * FRAME_STEP_S, 3), int(spk)]
        for a, b, spk in segments
        if (b - a) * FRAME_STEP_S >= min_duration_s
    ]
    return sorted(result, key=lambda item: (item[0], item[1]))


def _close_gaps(column: np.ndarray, max_gap: int) -> np.ndarray:
    column = column.copy()
    active = np.flatnonzero(column)
    if active.size < 2:
        return column
    for left, right in zip(active[:-1], active[1:]):
        if 1 < right - left <= max_gap + 1:
            column[left:right] = True
    return column
