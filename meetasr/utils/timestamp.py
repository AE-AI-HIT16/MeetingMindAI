"""VAD timestamp utilities — merge and align segment timestamps."""

from __future__ import annotations
from meetasr.schemas import Segment, SentenceInfo


def merge_vad_segments(
    segments: list[Segment],
    max_merge_gap_ms: int = 300,
    max_segment_ms: int = 60000,
) -> list[Segment]:
    """Merge short adjacent VAD segments into longer ones.

    Args:
        segments: Input VAD segments sorted by start_ms.
        max_merge_gap_ms: Merge segments with gap ≤ this value.
        max_segment_ms: Don't merge if result would exceed this length.

    Returns:
        Merged segments list.
    """
    if not segments:
        return []

    merged: list[Segment] = [Segment(segments[0].start_ms, segments[0].end_ms)]
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg.start_ms - last.end_ms
        would_be_len = seg.end_ms - last.start_ms
        if gap <= max_merge_gap_ms and would_be_len <= max_segment_ms:
            merged[-1] = Segment(last.start_ms, seg.end_ms)
        else:
            merged.append(Segment(seg.start_ms, seg.end_ms))
    return merged


def align_timestamps_to_global(
    char_timestamps: list[list[int]],
    segment_offset_ms: int,
) -> list[list[int]]:
    """Shift char-level timestamps by VAD segment offset.

    Args:
        char_timestamps: [[start_ms, end_ms], ...] relative to segment start.
        segment_offset_ms: Global offset (segment.start_ms).

    Returns:
        Timestamps adjusted to global timeline.
    """
    return [
        [t[0] + segment_offset_ms, t[1] + segment_offset_ms]
        for t in char_timestamps
    ]


def build_sentence_info(
    asr_results: list[dict],
    vad_segments: list[Segment],
) -> list[SentenceInfo]:
    """Build SentenceInfo list from per-segment ASR results.

    Args:
        asr_results: List of ASR result dicts, one per VAD segment.
            Each has keys: "text", "timestamp" (char-level, relative).
        vad_segments: Corresponding VAD segments (same order).

    Returns:
        List of SentenceInfo with global timestamps.
    """
    sentences: list[SentenceInfo] = []
    for result, seg in zip(asr_results, vad_segments):
        text = result.get("text", "").strip()
        if not text:
            continue
        raw_ts = result.get("timestamp", [])
        global_ts = align_timestamps_to_global(raw_ts, seg.start_ms)

        start_s = seg.start_ms / 1000.0
        end_s = seg.end_ms / 1000.0
        if global_ts:
            start_s = global_ts[0][0] / 1000.0
            end_s = global_ts[-1][1] / 1000.0

        sentences.append(SentenceInfo(
            text=text,
            start=start_s,
            end=end_s,
            char_timestamps=global_ts,
        ))
    return sentences
