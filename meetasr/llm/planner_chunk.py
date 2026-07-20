"""Chunking utilities for DocumentPlanner.

Handles overlap for pronoun/coreference resolution and oversized ASR lines.

See docs/chunking_analysis.md for design rationale.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

MAX_CHARS_PER_CHUNK = 6000
OVERLAP_LINES = 5


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


def chunk_with_overlap(
    text: str,
    max_chars: int = MAX_CHARS_PER_CHUNK,
    overlap_lines: int = OVERLAP_LINES,
) -> list[str]:
    """Split text on line boundaries with overlap between chunks.

    Handles monster lines by sub-splitting before chunking.

    Args:
        text: Full transcript text (one sentence per line).
        max_chars: Maximum characters per chunk.
        overlap_lines: Lines from previous chunk to prepend for context.

    Returns:
        List of text chunks with overlap for pronoun resolution.

    Raises:
        ValueError: If max_chars or overlap_lines is invalid.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")
    if not text:
        return [text]

    # Pre-process: split monster lines.
    raw_lines = text.split("\n")
    lines: list[str] = []
    for line in raw_lines:
        if len(line) > max_chars:
            logger.warning("Monster line detected (%d chars), sub-splitting.", len(line))
            lines.extend(_split_monster_line(line, max_chars))
        else:
            lines.append(line)

    chunks: list[str] = []
    current: list[str] = []

    for line in lines:
        candidate = "\n".join([*current, line])
        if len(candidate) > max_chars and current:
            chunks.append("\n".join(current))

            # Carry only the newest lines that fit together with the next
            # original line. This guarantees progress and prevents overlap
            # from making the new chunk larger than max_chars.
            overlap: list[str] = []
            if overlap_lines:
                for previous in reversed(current[-overlap_lines:]):
                    with_previous = "\n".join([previous, *overlap, line])
                    if len(with_previous) > max_chars:
                        break
                    overlap.insert(0, previous)
            current = [*overlap, line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join(current))

    if any(len(chunk) > max_chars for chunk in chunks):
        raise AssertionError("chunk_with_overlap produced an oversized chunk")
    return chunks or [text]
