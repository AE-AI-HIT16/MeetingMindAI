"""Drop transcripts that only echo another speaker's overlapping words.

A turn heard mostly on top of someone else is transcribed from the mixture.
When separation cannot pull the quieter voice out (reverb, noise), Qwen
writes the louder speaker's words again under the wrong name.
"""

from __future__ import annotations

import re
import unicodedata

MIN_OVERLAP_RATIO = 0.6
MIN_ECHO_RATIO = 0.6


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"[^\w\s]", " ", text).split()


def overlap_ratio(start_ms: int, end_ms: int, overlaps: list[tuple[int, int]]) -> float:
    """Share of a time range covered by overlapped speech.

    Args:
        start_ms: Range start.
        end_ms: Range end.
        overlaps: ``(start_ms, end_ms)`` overlapped ranges.

    Returns:
        Fraction in [0, 1]; 0.0 for an empty range.
    """
    if end_ms <= start_ms:
        return 0.0
    covered = sum(
        max(0, min(end_ms, b) - max(start_ms, a)) for a, b in overlaps
    )
    return min(1.0, covered / (end_ms - start_ms))


def is_overlap_echo(text: str, ratio: float, concurrent_texts: list[str]) -> bool:
    """Whether a mostly-overlapped turn only repeats another speaker.

    Args:
        text: Transcript of the overlapped turn.
        ratio: Share of the turn that is overlapped (``overlap_ratio``).
        concurrent_texts: Transcripts of other speakers' turns at that time.

    Returns:
        True if at least ``MIN_OVERLAP_RATIO`` is overlapped and at least
        ``MIN_ECHO_RATIO`` of its words appear in the concurrent turns.
    """
    if ratio < MIN_OVERLAP_RATIO or not concurrent_texts:
        return False
    words = _words(text)
    if not words:
        return False
    other = set(_words(" ".join(concurrent_texts)))
    shared = sum(1 for word in words if word in other)
    return shared / len(words) >= MIN_ECHO_RATIO
