"""Conservative speaker attribution for ASR segments without timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SpeakerSegmentKind = Literal["single", "mixed", "uncertain"]


@dataclass(frozen=True, slots=True)
class SpeakerOverlapDecision:
    """Classification and diagnostics for one persisted transcript segment."""

    kind: SpeakerSegmentKind
    speaker: int | None
    dominant_ratio: float = 0.0
    secondary_ratio: float = 0.0
    secondary_turn_ms: int = 0
    coverage_ratio: float = 0.0


def classify_speaker_overlap(
    start_ms: int,
    end_ms: int,
    diar_segments: list[list],
    *,
    dominant_ratio_threshold: float = 0.80,
    secondary_ratio_threshold: float = 0.15,
    meaningful_secondary_ms: int = 400,
    jitter_ms: int = 150,
    minimum_coverage_ratio: float = 0.50,
) -> SpeakerOverlapDecision:
    """Classify one segment without splitting or estimating character timing."""
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("speaker attribution range must be positive")

    overlap_by_speaker: dict[int, int] = {}
    longest_turn_by_speaker: dict[int, int] = {}
    for raw_start_s, raw_end_s, raw_speaker in diar_segments:
        turn_start_ms = round(float(raw_start_s) * 1000)
        turn_end_ms = round(float(raw_end_s) * 1000)
        overlap_ms = min(end_ms, turn_end_ms) - max(start_ms, turn_start_ms)
        if overlap_ms <= 0:
            continue
        speaker = int(raw_speaker)
        overlap_by_speaker[speaker] = (
            overlap_by_speaker.get(speaker, 0) + overlap_ms
        )
        longest_turn_by_speaker[speaker] = max(
            longest_turn_by_speaker.get(speaker, 0),
            overlap_ms,
        )

    duration_ms = end_ms - start_ms
    total_overlap_ms = sum(overlap_by_speaker.values())
    coverage_ratio = min(1.0, total_overlap_ms / duration_ms)
    if not overlap_by_speaker or coverage_ratio < minimum_coverage_ratio:
        return SpeakerOverlapDecision(
            kind="uncertain",
            speaker=None,
            coverage_ratio=coverage_ratio,
        )

    ranked = sorted(
        overlap_by_speaker.items(),
        key=lambda item: (-item[1], item[0]),
    )
    dominant_speaker, dominant_ms = ranked[0]
    dominant_ratio = dominant_ms / total_overlap_ms
    if len(ranked) == 1:
        return SpeakerOverlapDecision(
            kind="single",
            speaker=dominant_speaker,
            dominant_ratio=dominant_ratio,
            coverage_ratio=coverage_ratio,
        )

    secondary_speaker, secondary_ms = ranked[1]
    secondary_ratio = secondary_ms / total_overlap_ms
    secondary_turn_ms = longest_turn_by_speaker[secondary_speaker]
    meaningful_secondary = (
        secondary_turn_ms >= meaningful_secondary_ms
        or (
            secondary_turn_ms >= jitter_ms
            and secondary_ratio >= secondary_ratio_threshold
        )
    )
    if meaningful_secondary:
        kind: SpeakerSegmentKind = "mixed"
        speaker = None
    elif dominant_ratio >= dominant_ratio_threshold:
        kind = "single"
        speaker = dominant_speaker
    else:
        kind = "uncertain"
        speaker = None

    return SpeakerOverlapDecision(
        kind=kind,
        speaker=speaker,
        dominant_ratio=dominant_ratio,
        secondary_ratio=secondary_ratio,
        secondary_turn_ms=secondary_turn_ms,
        coverage_ratio=coverage_ratio,
    )
