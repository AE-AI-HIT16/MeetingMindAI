"""Chunking and representative-sampling helpers for DocumentPlanner."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

def representative_sample(text: str, max_chars: int) -> str:
    """Return head, middle, and tail excerpts within ``max_chars``.

    Args:
        text: Full transcript text.
        max_chars: Strict output character budget.

    Returns:
        Original text when it fits, otherwise a representative excerpt.

    Raises:
        ValueError: If max_chars is not positive.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if len(text) <= max_chars:
        return text

    separator = "\n...\n"
    if max_chars <= 2 * len(separator):
        return text[:max_chars]
    content_budget = max_chars - 2 * len(separator)
    head_size = content_budget // 3
    middle_size = content_budget // 3
    tail_size = content_budget - head_size - middle_size
    middle_start = max(0, len(text) // 2 - middle_size // 2)
    return (
        text[:head_size]
        + separator
        + text[middle_start : middle_start + middle_size]
        + separator
        + text[-tail_size:]
    )


def format_time_range(start: float, end: float) -> str:
    """Format a transcript range consistently as MM:SS or HH:MM:SS."""
    include_hours = max(start, end) >= 3600

    def _format(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if include_hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    return f"[{_format(start)}–{_format(end)}]"


def chunk_on_lines(text: str, max_chars: int) -> list[str]:
    """Split transcript text on line boundaries within a strict budget.

    Oversized malformed ASR lines are sub-split as a last resort so no LLM
    input can exceed the configured technical limit.

    Args:
        text: Transcript text with one sentence per line.
        max_chars: Maximum characters in each returned chunk.

    Returns:
        Ordered transcript chunks.

    Raises:
        ValueError: If max_chars is not positive.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if not text:
        return [text]

    lines: list[str] = []
    for line in text.split("\n"):
        if len(line) <= max_chars:
            lines.append(line)
        else:
            logger.warning(
                "Transcript line exceeds the LLM budget; splitting it safely."
            )
            lines.extend(_split_monster_line(line, max_chars))

    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = "\n".join([*current, line])
        if current and len(candidate) > max_chars:
            chunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks or [text]


def _split_monster_line(line: str, max_chars: int) -> list[str]:
    """Break a single oversized line into smaller pieces.

    Splits on Vietnamese/general sentence boundaries (. ! ? ;) first,
    then falls back to comma, then hard-cut as last resort.

    Args:
        line: The oversized line.
        max_chars: Target max size per piece.

    Returns:
        List of sub-lines, each no longer than max_chars.

    Raises:
        ValueError: If max_chars is not positive.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if len(line) <= max_chars:
        return [line]

    # Try splitting on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?;])\s+', line)
    if len(parts) == 1:
        # Fallback: split on commas
        parts = re.split(r'(?<=,)\s+', line)
    if len(parts) == 1:
        return [line[i : i + max_chars] for i in range(0, len(line), max_chars)]

    # A malformed ASR sentence may still exceed max_chars even after a
    # punctuation split. Hard-cut those individual parts before merging.
    bounded_parts: list[str] = []
    for part in parts:
        bounded_parts.extend(
            part[i : i + max_chars] for i in range(0, len(part), max_chars)
        )

    # Merge small parts back together up to max_chars
    merged: list[str] = []
    current = ""
    for part in bounded_parts:
        if len(current) + len(part) + 1 > max_chars and current:
            merged.append(current)
            current = part
        else:
            current = f"{current} {part}".strip() if current else part
    if current:
        merged.append(current)

    return merged
