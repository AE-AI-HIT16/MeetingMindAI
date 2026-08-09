"""Coverage accounting for post-session realtime finalization."""

from types import SimpleNamespace

import pytest

from meetasr.streaming.audio_queue import AudioQueue
from meetasr.streaming.audio_receiver import AudioReceiver
from meetasr.streaming.coverage import RealtimeCoverageTracker


def test_coverage_is_complete_when_every_speech_range_is_confirmed() -> None:
    tracker = RealtimeCoverageTracker()
    tracker.record_speech(500, 1800)
    tracker.record_speech(2400, 3900)
    tracker.record_confirmed(500, 1800)
    tracker.record_confirmed(2400, 3900)
    tracker.mark_flush_completed()

    snapshot = tracker.snapshot()

    assert snapshot.complete is True
    assert snapshot.missing_ranges == ()


def test_coverage_is_incomplete_for_failure_drop_or_missing_range() -> None:
    tracker = RealtimeCoverageTracker()
    tracker.record_speech(0, 1000)
    tracker.record_speech(1500, 2600)
    tracker.record_confirmed(0, 1000)
    tracker.record_asr_failure()
    tracker.record_audio_drop()
    tracker.mark_flush_completed()

    snapshot = tracker.snapshot()

    assert snapshot.complete is False
    assert [(item.start_ms, item.end_ms) for item in snapshot.missing_ranges] == [
        (1500, 2600)
    ]
    assert snapshot.asr_failure_count == 1
    assert snapshot.audio_drop_count == 1


def test_coverage_requires_successful_final_flush() -> None:
    tracker = RealtimeCoverageTracker()

    assert tracker.snapshot().complete is False

    tracker.mark_flush_completed()

    assert tracker.snapshot().complete is True


@pytest.mark.asyncio
async def test_audio_receiver_records_backpressure_drop() -> None:
    session = SimpleNamespace(
        audio_queue=AudioQueue(maxsize=1),
        coverage=RealtimeCoverageTracker(),
    )
    receiver = AudioReceiver(session, chunk_size=2)

    await receiver.receive(b"\x00\x00\x00\x00")

    assert session.coverage.snapshot().audio_drop_count == 1
