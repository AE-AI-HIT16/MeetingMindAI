"""Lossless coverage accounting for confirmed realtime ASR."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, order=True, slots=True)
class TimelineRange:
    """Half-open millisecond interval on the original recording timeline."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("timeline range must be positive")


@dataclass(frozen=True, slots=True)
class RealtimeCoverageSnapshot:
    """Immutable evidence used to choose diarization-only or full fallback."""

    speech_ranges: tuple[TimelineRange, ...] = ()
    confirmed_ranges: tuple[TimelineRange, ...] = ()
    asr_failure_count: int = 0
    audio_drop_count: int = 0
    flush_completed: bool = False

    @property
    def missing_ranges(self) -> tuple[TimelineRange, ...]:
        return tuple(
            speech
            for speech in self.speech_ranges
            if not _is_fully_covered(speech, self.confirmed_ranges)
        )

    @property
    def complete(self) -> bool:
        return (
            self.flush_completed
            and self.asr_failure_count == 0
            and self.audio_drop_count == 0
            and not self.missing_ranges
        )


@dataclass(slots=True)
class RealtimeCoverageTracker:
    """Mutable per-WebSocket coverage state owned by ``StreamSession``."""

    _speech_ranges: set[TimelineRange] = field(default_factory=set)
    _confirmed_ranges: set[TimelineRange] = field(default_factory=set)
    _asr_failure_count: int = 0
    _audio_drop_count: int = 0
    _flush_completed: bool = False

    def record_speech(self, start_ms: int, end_ms: int) -> None:
        self._speech_ranges.add(TimelineRange(start_ms, end_ms))

    def record_confirmed(self, start_ms: int, end_ms: int) -> None:
        self._confirmed_ranges.add(TimelineRange(start_ms, end_ms))

    def record_asr_failure(self) -> None:
        self._asr_failure_count += 1

    def record_audio_drop(self) -> None:
        self._audio_drop_count += 1

    def mark_flush_completed(self) -> None:
        self._flush_completed = True

    def snapshot(self) -> RealtimeCoverageSnapshot:
        return RealtimeCoverageSnapshot(
            speech_ranges=tuple(sorted(self._speech_ranges)),
            confirmed_ranges=tuple(sorted(self._confirmed_ranges)),
            asr_failure_count=self._asr_failure_count,
            audio_drop_count=self._audio_drop_count,
            flush_completed=self._flush_completed,
        )


def _is_fully_covered(
    target: TimelineRange,
    confirmed: tuple[TimelineRange, ...],
) -> bool:
    cursor = target.start_ms
    for item in sorted(confirmed):
        if item.end_ms <= cursor or item.start_ms >= target.end_ms:
            continue
        if item.start_ms > cursor:
            return False
        cursor = max(cursor, item.end_ms)
        if cursor >= target.end_ms:
            return True
    return False
