"""Tests for DocumentPlanner chunking helpers."""

import pytest

from meetasr.llm.planner_chunk import (
    _split_monster_line,
    chunk_on_lines,
    format_time_range,
    representative_sample,
)


def test_short_text_is_not_split() -> None:
    text = "line1\nline2\nline3"
    assert chunk_on_lines(text, 100) == [text]


def test_line_boundaries_and_content_are_preserved() -> None:
    lines = [f"line-{index}-" + "x" * 20 for index in range(20)]
    chunks = chunk_on_lines("\n".join(lines), 100)
    flattened = [line for chunk in chunks for line in chunk.split("\n")]

    assert flattened == lines
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_oversized_line_is_bounded() -> None:
    line = "Câu nội dung rất dài. " * 200
    chunks = chunk_on_lines(line, 200)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_hard_cut_preserves_unpunctuated_content() -> None:
    line = "a" * 1000
    pieces = _split_monster_line(line, 200)

    assert "".join(pieces) == line
    assert all(len(piece) <= 200 for piece in pieces)


def test_representative_sample_returns_short_input_unchanged() -> None:
    assert representative_sample("short", 100) == "short"


def test_representative_sample_is_strictly_bounded() -> None:
    text = "A" * 1000 + "B" * 1000 + "C" * 1000
    sample = representative_sample(text, 300)

    assert len(sample) <= 300
    assert all(marker in sample for marker in ("A", "B", "C"))


def test_time_range_uses_minutes_below_one_hour() -> None:
    assert format_time_range(206.0, 321.0) == "[03:26–05:21]"


def test_time_range_uses_hours_after_one_hour() -> None:
    assert format_time_range(3590.0, 3670.0) == "[00:59:50–01:01:10]"


@pytest.mark.parametrize("max_chars", [0, -1])
def test_invalid_character_budget(max_chars: int) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_on_lines("text", max_chars)
    with pytest.raises(ValueError, match="max_chars"):
        representative_sample("text", max_chars)
