"""Plan and resolve safe ASR retries for mixed realtime transcript segments."""

from __future__ import annotations

import copy
import logging
from typing import Callable

from meetasr.schemas import (
    SentenceInfo,
    SpeakerTurn,
    TargetedAction,
    TargetedRetranscriptionResult,
    TargetedRetranscriptionStats,
    TargetedSegmentPlan,
)
from meetasr.utils.speaker_overlap import classify_speaker_overlap

logger = logging.getLogger(__name__)
TurnTranscriber = Callable[[SpeakerTurn], list[SentenceInfo]]


def build_targeted_retranscription_plan(
    sentence_info: list[SentenceInfo],
    diar_segments: list[list],
    *,
    minimum_coverage_ratio: float = 0.50,
    minimum_turn_ms: int = 150,
) -> list[TargetedSegmentPlan]:
    """Choose which realtime segments are safe to keep or retranscribe."""
    if not 0.0 <= minimum_coverage_ratio <= 1.0:
        raise ValueError("minimum_coverage_ratio must be between 0 and 1")
    if minimum_turn_ms <= 0:
        raise ValueError("minimum_turn_ms must be positive")

    plans: list[TargetedSegmentPlan] = []
    for index, sentence in enumerate(sentence_info):
        start_ms = max(0, int(round(sentence.start * 1000)))
        end_ms = max(start_ms, int(round(sentence.end * 1000)))
        if end_ms <= start_ms:
            plans.append(TargetedSegmentPlan(index, "fallback", "uncertain", None))
            continue

        decision = classify_speaker_overlap(start_ms, end_ms, diar_segments)
        if decision.kind == "single":
            plans.append(TargetedSegmentPlan(index, "keep", decision.kind, decision.speaker))
            continue

        turns = _intersect_speaker_turns(
            start_ms,
            end_ms,
            diar_segments,
            minimum_turn_ms=minimum_turn_ms,
        )
        covered_ms = sum(turn.duration_ms for turn in turns)
        coverage_ratio = covered_ms / (end_ms - start_ms)
        action: TargetedAction = (
            "retranscribe" if turns and coverage_ratio >= minimum_coverage_ratio else "fallback"
        )
        plans.append(
            TargetedSegmentPlan(
                index,
                action,
                decision.kind,
                None,
                tuple(turns) if action == "retranscribe" else (),
            )
        )
    return plans


def resolve_targeted_retranscription(
    sentence_info: list[SentenceInfo],
    plans: list[TargetedSegmentPlan],
    transcribe_turn: TurnTranscriber,
) -> TargetedRetranscriptionResult:
    """Apply atomic replacements and retain original text on partial failure."""
    if len(plans) != len(sentence_info):
        raise ValueError("plans must contain one entry per sentence")
    if any(plan.index != index for index, plan in enumerate(plans)):
        raise ValueError("plans must follow sentence order")

    finalized: list[SentenceInfo] = []
    source_indices: list[int] = []
    replaced_indices: list[int] = []
    kept_segments = targeted_segments = replaced_segments = 0
    fallback_segments = asr_audio_ms = 0

    for sentence, plan in zip(sentence_info, plans):
        if plan.action == "keep":
            kept = copy.deepcopy(sentence)
            kept.speaker = plan.speaker
            finalized.append(kept)
            source_indices.append(plan.index)
            kept_segments += 1
            continue

        if plan.action == "fallback":
            finalized.append(_unknown_copy(sentence))
            source_indices.append(plan.index)
            fallback_segments += 1
            continue

        targeted_segments += 1
        replacement: list[SentenceInfo] = []
        try:
            for turn in plan.turns:
                asr_audio_ms += turn.duration_ms
                turn_sentences = [item for item in transcribe_turn(turn) if item.text.strip()]
                if not turn_sentences:
                    raise ValueError("targeted ASR returned no text")
                for item in turn_sentences:
                    resolved = copy.deepcopy(item)
                    resolved.speaker = turn.speaker
                    replacement.append(resolved)
        except Exception as exc:
            logger.warning(
                "Targeted ASR failed for segment %s: %s. Keeping original text.",
                plan.index,
                exc,
            )
            finalized.append(_unknown_copy(sentence))
            source_indices.append(plan.index)
            fallback_segments += 1
            continue

        finalized.extend(replacement)
        source_indices.extend([plan.index] * len(replacement))
        replaced_indices.append(plan.index)
        replaced_segments += 1

    _remap_speakers_by_first_appearance(finalized)
    stats = TargetedRetranscriptionStats(
        kept_segments,
        targeted_segments,
        replaced_segments,
        fallback_segments,
        asr_audio_ms,
    )
    return TargetedRetranscriptionResult(
        finalized,
        source_indices,
        tuple(replaced_indices),
        stats,
    )


def _intersect_speaker_turns(
    start_ms: int,
    end_ms: int,
    diar_segments: list[list],
    *,
    minimum_turn_ms: int,
) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for raw_start_s, raw_end_s, raw_speaker in diar_segments:
        turn_start_ms = max(start_ms, round(float(raw_start_s) * 1000))
        turn_end_ms = min(end_ms, round(float(raw_end_s) * 1000))
        if turn_end_ms - turn_start_ms < minimum_turn_ms:
            continue
        turns.append(SpeakerTurn(turn_start_ms, turn_end_ms, int(raw_speaker)))
    return sorted(turns, key=lambda turn: (turn.start_ms, turn.end_ms))


def _unknown_copy(sentence: SentenceInfo) -> SentenceInfo:
    copied = copy.deepcopy(sentence)
    copied.speaker = None
    return copied


def _remap_speakers_by_first_appearance(sentence_info: list[SentenceInfo]) -> None:
    mapping: dict[int, int] = {}
    for sentence in sentence_info:
        if sentence.speaker is None:
            continue
        sentence.speaker = mapping.setdefault(sentence.speaker, len(mapping))
