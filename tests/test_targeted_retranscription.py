"""Tests for safe targeted ASR retries after a realtime session."""

from __future__ import annotations

from meetasr.schemas import SentenceInfo, SpeakerTurn
from meetasr.utils.targeted_retranscription import (
    build_targeted_retranscription_plan,
    resolve_targeted_retranscription,
)


def test_plan_keeps_single_and_targets_mixed_segment() -> None:
    sentences = [
        SentenceInfo(text="câu sạch", start=0.0, end=2.0),
        SentenceInfo(text="hai người nói", start=2.0, end=6.0),
    ]
    diar_segments = [
        [0.0, 2.0, 7],
        [2.0, 4.0, 7],
        [4.0, 6.0, 3],
    ]

    plans = build_targeted_retranscription_plan(
        sentences,
        diar_segments,
    )

    assert [(plan.action, plan.kind) for plan in plans] == [
        ("keep", "single"),
        ("retranscribe", "mixed"),
    ]
    assert plans[0].speaker == 7
    assert [(turn.start_ms, turn.end_ms, turn.speaker) for turn in plans[1].turns] == [
        (2000, 4000, 7),
        (4000, 6000, 3),
    ]


def test_plan_falls_back_when_diarization_coverage_is_too_low() -> None:
    sentences = [
        SentenceInfo(text="không đủ coverage", start=0.0, end=4.0),
    ]

    plans = build_targeted_retranscription_plan(
        sentences,
        [[0.0, 1.0, 0]],
    )

    assert plans[0].action == "fallback"
    assert plans[0].turns == ()


def test_plan_merges_same_speaker_silence_without_swallowing_backchannel() -> None:
    sentences = [
        SentenceInfo(text="có khoảng lặng và backchannel", start=0.0, end=3.0),
    ]
    diar_segments = [
        [0.0, 1.0, 0],
        [1.45, 2.0, 0],
        [2.0, 2.4, 1],
        [2.4, 3.0, 0],
    ]

    plans = build_targeted_retranscription_plan(
        sentences,
        diar_segments,
    )

    assert plans[0].action == "retranscribe"
    assert [
        (turn.start_ms, turn.end_ms, turn.speaker)
        for turn in plans[0].turns
    ] == [
        (0, 2000, 0),
        (2000, 2400, 1),
        (2400, 3000, 0),
    ]


def test_merged_silence_does_not_inflate_diarization_coverage() -> None:
    plans = build_targeted_retranscription_plan(
        [SentenceInfo(text="coverage thấp", start=0.0, end=4.0)],
        [[0.0, 1.0, 0], [1.59, 2.0, 0]],
    )

    assert plans[0].action == "fallback"
    assert plans[0].turns == ()


def test_resolve_replaces_only_targeted_segment_and_remaps_speakers() -> None:
    sentences = [
        SentenceInfo(text="giữ nguyên", start=0.0, end=2.0),
        SentenceInfo(text="nội dung cũ", start=2.0, end=6.0),
    ]
    plans = build_targeted_retranscription_plan(
        sentences,
        [[0.0, 2.0, 7], [2.0, 4.0, 7], [4.0, 6.0, 3]],
    )
    calls: list[tuple[int, int, int]] = []

    def transcribe_turn(turn: SpeakerTurn) -> list[SentenceInfo]:
        calls.append((turn.start_ms, turn.end_ms, turn.speaker))
        return [
            SentenceInfo(
                text=f"mới {turn.speaker}",
                start=turn.start_ms / 1000,
                end=turn.end_ms / 1000,
            )
        ]

    result = resolve_targeted_retranscription(
        sentences,
        plans,
        transcribe_turn,
    )

    assert calls == [(2000, 4000, 7), (4000, 6000, 3)]
    assert [item.text for item in result.sentence_info] == [
        "giữ nguyên",
        "mới 7",
        "mới 3",
    ]
    assert [item.speaker for item in result.sentence_info] == [0, 0, 1]
    assert result.source_indices == [0, 1, 1]
    assert result.replaced_indices == (1,)
    assert result.stats.kept_segments == 1
    assert result.stats.targeted_segments == 1
    assert result.stats.replaced_segments == 1
    assert result.stats.fallback_segments == 0
    assert result.stats.asr_audio_ms == 4000


def test_resolve_keeps_whole_original_when_one_turn_fails() -> None:
    sentences = [
        SentenceInfo(text="không được mất", start=0.0, end=4.0),
    ]
    plans = build_targeted_retranscription_plan(
        sentences,
        [[0.0, 2.0, 0], [2.0, 4.0, 1]],
    )

    def transcribe_turn(turn: SpeakerTurn) -> list[SentenceInfo]:
        if turn.speaker == 1:
            return []
        return [SentenceInfo(text="một phần", start=0.0, end=2.0)]

    result = resolve_targeted_retranscription(
        sentences,
        plans,
        transcribe_turn,
    )

    assert [item.text for item in result.sentence_info] == ["không được mất"]
    assert result.source_indices == [0]
    assert result.replaced_indices == ()
    assert result.sentence_info[0].speaker is None
    assert result.stats.targeted_segments == 1
    assert result.stats.replaced_segments == 0
    assert result.stats.fallback_segments == 1
    assert result.stats.asr_audio_ms == 4000
