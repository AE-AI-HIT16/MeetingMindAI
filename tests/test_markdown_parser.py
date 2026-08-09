"""Tests for shared Markdown document normalization."""

from meetasr.export.markdown_parser import (
    extract_leading_document_heading,
    resolve_document_title,
    strip_leading_document_heading,
)


def test_extracts_initial_h1_as_document_title():
    markdown = "\n# Báo cáo phân tích trận bán kết\n\n## Tổng quan\n"

    assert extract_leading_document_heading(markdown) == (
        "Báo cáo phân tích trận bán kết"
    )


def test_extracts_h1_with_optional_closing_hashes():
    assert extract_leading_document_heading("# Tiêu đề báo cáo ##\n") == (
        "Tiêu đề báo cáo"
    )


def test_missing_initial_h1_has_no_document_title():
    assert extract_leading_document_heading("## Tổng quan\n") is None


def test_resolves_uppercase_title_from_h1_before_fallback():
    markdown = "# Báo cáo phân tích trận bán kết\n"

    assert resolve_document_title(markdown, "ten_file_md") == (
        "BÁO CÁO PHÂN TÍCH TRẬN BÁN KẾT"
    )


def test_resolves_uppercase_title_from_fallback_without_h1():
    assert resolve_document_title("## Tổng quan\n", "ten_file_md") == (
        "TEN_FILE_MD"
    )


def test_strip_leading_h1_still_preserves_body():
    markdown = "# Tiêu đề\n\n## Tổng quan\n\nNội dung"

    assert strip_leading_document_heading(markdown) == (
        "## Tổng quan\n\nNội dung"
    )
