"""Safe speaker attribution without character timestamps."""

from meetasr.utils.speaker_overlap import classify_speaker_overlap


def test_clear_dominant_speaker_is_single() -> None:
    decision = classify_speaker_overlap(
        0,
        5000,
        [[0.0, 4.7, 3], [4.7, 5.0, 3]],
    )

    assert decision.kind == "single"
    assert decision.speaker == 3


def test_very_short_secondary_jitter_does_not_make_segment_mixed() -> None:
    decision = classify_speaker_overlap(
        0,
        1000,
        [[0.0, 0.85, 0], [0.85, 0.95, 1], [0.95, 1.0, 0]],
    )

    assert decision.kind == "single"
    assert decision.speaker == 0


def test_meaningful_short_backchannel_is_mixed_even_below_fifteen_percent() -> None:
    decision = classify_speaker_overlap(
        0,
        10000,
        [[0.0, 4.8, 0], [4.8, 5.2, 1], [5.2, 10.0, 0]],
    )

    assert decision.kind == "mixed"
    assert decision.speaker is None
    assert decision.secondary_turn_ms == 400


def test_missing_diarization_overlap_is_uncertain() -> None:
    decision = classify_speaker_overlap(1000, 2000, [])

    assert decision.kind == "uncertain"
    assert decision.speaker is None
