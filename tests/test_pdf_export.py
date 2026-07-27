"""Tests for HTML/CSS template-based PDF export."""

from __future__ import annotations

from pathlib import Path

import pytest

from meetasr.export.pdf_exporter import _extract_sections, export_pdf


pytest.importorskip("jinja2")
pytest.importorskip("weasyprint")


MARKDOWN = """# Báo cáo cuộc họp

## Tổng quan

Cuộc họp thảo luận về **tiến độ dự án**.

## Công việc cần làm

- Anh Tú hoàn thiện phần export
- Nam kiểm tra frontend

## Phân công

| Thành viên | Công việc |
|---|---|
| Anh Tú | Export DOCX/PDF |
"""
FULL_PAGE_MARKDOWN = (
    Path(__file__).resolve().parent
    / "data"
    / "pdf_blue_modern_demo.md"
).read_text(encoding="utf-8")
MINIMAL_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "meetasr"
    / "export"
    / "templates"
    / "pdf"
    / "minimal.html"
)


def test_pdf_sections_come_from_llm_markdown_headings():
    sections = _extract_sections(MARKDOWN)

    assert [section.heading for section in sections] == [
        "Tổng quan",
        "Công việc cần làm",
        "Phân công",
    ]
    assert "tiến độ dự án" in sections[0].html
    assert "<ul>" in sections[1].html
    assert "<table>" in sections[2].html


def test_blue_modern_template_renders_dynamic_text_only_report():
    artifact = export_pdf(
        MARKDOWN,
        "Báo cáo cuộc họp",
        template="blue_modern",
        generated_at="23/07/2026",
    )

    assert artifact.filename == "Báo cáo cuộc họp.pdf"
    assert artifact.media_type == "application/pdf"
    assert artifact.content.startswith(b"%PDF-")
    assert len(artifact.content) > 5_000


def test_blue_modern_full_page_fixture_contains_real_report_sections():
    sections = _extract_sections(FULL_PAGE_MARKDOWN)
    artifact = export_pdf(
        FULL_PAGE_MARKDOWN,
        "Báo cáo cuộc họp",
        template="blue_modern",
        generated_at="23/07/2026",
    )

    assert [section.heading for section in sections] == [
        "Tổng quan",
        "Mục tiêu và kết quả",
        "Phát hiện và khuyến nghị",
        "Kế hoạch triển khai",
        "Khách hàng mục tiêu",
        "Quyết định và phân công",
    ]
    assert all(section.html for section in sections)
    assert artifact.content.startswith(b"%PDF-")
    assert len(artifact.content) > 15_000


def test_minimal_pdf_remains_the_default_template():
    artifact = export_pdf(MARKDOWN, "Báo cáo cuộc họp")

    assert artifact.content.startswith(b"%PDF-")
    assert len(artifact.content) > 5_000


def test_minimal_template_allows_long_sections_to_fill_remaining_page():
    """Long sections may split; headings should stay with following content."""
    template = MINIMAL_TEMPLATE.read_text(encoding="utf-8")

    assert ".section {" in template
    assert "break-inside: auto;" in template
    assert ".section-title {" in template
    assert "break-after: avoid;" in template
    assert "orphans: 3;" in template
    assert "widows: 3;" in template


def test_unknown_pdf_template_is_rejected():
    with pytest.raises(ValueError, match="Unknown PDF template"):
        export_pdf(MARKDOWN, "Báo cáo", template="unknown")  # type: ignore[arg-type]
