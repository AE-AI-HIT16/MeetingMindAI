"""Tests for chunk_with_overlap and _split_monster_line."""

import pytest

from meetasr.llm.planner_chunk import chunk_with_overlap, _split_monster_line


def test_short_text_no_split():
    """Text shorter than max_chars returns single chunk."""
    text = "line1\nline2\nline3"
    result = chunk_with_overlap(text, max_chars=100, overlap_lines=2)
    assert len(result) == 1
    assert result[0] == text


def test_overlap_lines_present():
    """Second chunk starts with overlap lines from first chunk."""
    lines = [f"line_{i}: " + "x" * 50 for i in range(20)]
    text = "\n".join(lines)
    result = chunk_with_overlap(text, max_chars=300, overlap_lines=3)
    assert len(result) >= 2
    chunk1_lines = result[0].split("\n")
    chunk2_lines = result[1].split("\n")
    # Last 3 lines of chunk 1 should be the first 3 lines of chunk 2
    assert chunk1_lines[-3:] == chunk2_lines[:3]


def test_monster_line_gets_split():
    """A single line > max_chars is sub-split, not passed through intact."""
    monster = "Đây là câu rất dài. " * 500  # ~10K chars
    text = "normal line\n" + monster + "\nanother normal"
    result = chunk_with_overlap(text, max_chars=2000, overlap_lines=2)
    # Monster line should be sub-split so no single sub-line exceeds max_chars
    # But chunks can hold multiple sub-lines. Verify sub-splitting happened.
    assert len(result) > 1  # must have split
    assert all(len(chunk) <= 2000 for chunk in result)
    # Each individual sub-line (from _split_monster_line) should be <= max_chars
    from meetasr.llm.planner_chunk import _split_monster_line
    pieces = _split_monster_line(monster, max_chars=2000)
    assert all(len(p) <= 2000 for p in pieces)


def test_monster_line_no_content_loss():
    """Sub-splitting a monster line preserves all content."""
    monster = "Phần A. Phần B. Phần C. Phần D."
    pieces = _split_monster_line(monster, max_chars=15)
    rejoined = " ".join(pieces)
    for word in ["Phần A", "Phần B", "Phần C", "Phần D"]:
        assert word in rejoined


def test_monster_line_hard_cut_fallback():
    """Line with no punctuation falls back to hard-cut."""
    monster = "a" * 10000  # no punctuation at all
    pieces = _split_monster_line(monster, max_chars=3000)
    assert all(len(p) <= 3000 for p in pieces)
    assert "".join(pieces) == monster


def test_empty_text():
    """Empty text returns a single-element list."""
    result = chunk_with_overlap("", max_chars=6000, overlap_lines=5)
    assert len(result) == 1


def test_overlap_preserves_speaker_context():
    """Overlap carries speaker identity across chunk boundary."""
    lines = [
        "[0.0s] Speaker 1: Anh Khương sẽ phụ trách.",
        "[5.0s] Speaker 2: Ok, ghi nhận.",
        "[10.0s] Speaker 1: Anh ấy cần review code.",
        "[15.0s] Speaker 3: Ảnh nói deadline thứ 6.",
        "[20.0s] Speaker 2: Vậy ảnh cập nhật timeline.",
    ]
    text = "\n".join(lines)
    result = chunk_with_overlap(text, max_chars=150, overlap_lines=2)
    if len(result) >= 2:
        # Chunk 2 should have overlap lines + new lines
        assert len(result[1].split("\n")) > 2


def test_single_line_text():
    """Text with no newlines returns single chunk if within limit."""
    text = "Đây là một câu duy nhất."
    result = chunk_with_overlap(text, max_chars=6000, overlap_lines=5)
    assert len(result) == 1
    assert result[0] == text


def test_overlap_zero():
    """overlap_lines=0 produces no overlap between chunks."""
    lines = [f"line_{i}: " + "x" * 50 for i in range(20)]
    text = "\n".join(lines)
    result = chunk_with_overlap(text, max_chars=300, overlap_lines=0)
    assert len(result) >= 2
    chunk1_lines = result[0].split("\n")
    chunk2_lines = result[1].split("\n")
    # With 0 overlap, last line of chunk 1 should NOT be first line of chunk 2
    assert chunk1_lines[-1] != chunk2_lines[0]
    flattened = [line for chunk in result for line in chunk.split("\n")]
    assert flattened == lines


def test_all_content_preserved():
    """All original lines appear in at least one chunk."""
    lines = [f"unique_line_{i}" for i in range(30)]
    text = "\n".join(lines)
    result = chunk_with_overlap(text, max_chars=200, overlap_lines=2)
    all_content = "\n".join(result)
    for line in lines:
        assert line in all_content


def test_chunk_count_increases_with_text_size():
    """More text produces more chunks."""
    small = "\n".join([f"line_{i}: " + "x" * 50 for i in range(5)])
    large = "\n".join([f"line_{i}: " + "x" * 50 for i in range(50)])
    small_chunks = chunk_with_overlap(small, max_chars=300, overlap_lines=2)
    large_chunks = chunk_with_overlap(large, max_chars=300, overlap_lines=2)
    assert len(large_chunks) > len(small_chunks)


def test_monster_line_sentence_boundary_split():
    """Monster line with sentence boundaries splits at those boundaries."""
    # Sentences separated by ". "
    sentences = ["Câu thứ " + str(i) + " rất dài" for i in range(20)]
    monster = ". ".join(sentences) + "."
    pieces = _split_monster_line(monster, max_chars=100)
    # Each piece should be <= 100 chars
    for piece in pieces:
        assert len(piece) <= 100
    # Content should be preserved
    rejoined = " ".join(pieces)
    for i in range(20):
        assert f"Câu thứ {i}" in rejoined


def test_long_sentence_part_is_hard_cut():
    """A punctuation split containing an oversized part must still be bounded."""
    monster = "a" * 500 + ". short sentence."
    pieces = _split_monster_line(monster, max_chars=100)
    assert all(len(piece) <= 100 for piece in pieces)


def test_all_chunks_respect_max_chars():
    """Overlap must never make a produced chunk exceed its configured limit."""
    lines = ["x" * 95 for _ in range(20)]
    chunks = chunk_with_overlap("\n".join(lines), max_chars=200, overlap_lines=5)
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_invalid_chunk_arguments():
    """Invalid size and overlap settings fail with actionable errors."""
    with pytest.raises(ValueError, match="max_chars"):
        chunk_with_overlap("text", max_chars=0)
    with pytest.raises(ValueError, match="overlap_lines"):
        chunk_with_overlap("text", overlap_lines=-1)
