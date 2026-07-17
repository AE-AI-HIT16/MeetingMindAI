"""Chunking utilities for DocumentPlanner.

Handles:
- Overlap for pronoun/coreference resolution
- Monster lines (single line > max_chars)
- Optional token-based measurement via tiktoken

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
        List of sub-lines, each <= max_chars (best effort).
    """
    if len(line) <= max_chars:
        return [line]

    # Try splitting on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?;])\s+', line)
    if len(parts) == 1:
        # Fallback: split on commas
        parts = re.split(r'(?<=,)\s+', line)
    if len(parts) == 1:
        # Last resort: hard-cut at max_chars boundaries
        return [line[i:i + max_chars] for i in range(0, len(line), max_chars)]

    # Merge small parts back together up to max_chars
    merged: list[str] = []
    current = ""
    for part in parts:
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
    """
    # Pre-process: split monster lines
    raw_lines = text.split("\n")
    lines: list[str] = []
    for line in raw_lines:
        if len(line) > max_chars:
            logger.warning("Monster line detected (%d chars), sub-splitting.", len(line))
            lines.extend(_split_monster_line(line, max_chars))
        else:
            lines.append(line)

    if not lines:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for \n

        if size + line_len > max_chars and current:
            chunks.append("\n".join(current))
            # Overlap: carry last N lines into next chunk
            overlap = current[-overlap_lines:] if len(current) > overlap_lines else current[:]
            current = list(overlap)
            size = sum(len(ln) + 1 for ln in current)

        current.append(line)
        size += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks or [text]
