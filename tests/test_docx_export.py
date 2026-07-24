"""Tests for template-based Markdown to DOCX export."""

from __future__ import annotations

import builtins
from io import BytesIO
from zipfile import ZipFile

import pytest

from meetasr.export import create_export_service
from meetasr.export.docx_exporter import (
    _load_docx_template_class,
    _markdown_body,
    export_docx,
)
from meetasr.export.service import MissingExportDependency


docx = pytest.importorskip("docx")
pytest.importorskip("docxtpl")
pytest.importorskip("docxcompose")


MARKDOWN = """# Báo cáo thử nghiệm

## Tổng quan

Nội dung **tiếng Việt đậm**, *in nghiêng*, ~~đã bỏ~~ và [OpenAI](https://openai.com).

- Việc thứ nhất
  - Việc con

3. Bước ba
4. Bước bốn

> Ghi chú quan trọng.

| Người phụ trách | Công việc |
|---|---|
| Anh Tú | Kiểm thử DOCX |

[Không cho file](file:///etc/passwd)

![Không tải ảnh](https://example.com/secret.png)
"""


def _open_artifact(markdown: str = MARKDOWN):
    artifact = export_docx(
        markdown,
        "Báo cáo thử nghiệm",
        generated_at="22/07/2026",
    )
    return artifact, docx.Document(BytesIO(artifact.content))


def test_docx_uses_template_and_preserves_vietnamese_content():
    artifact, document = _open_artifact()

    assert artifact.filename == "Báo cáo thử nghiệm.docx"
    assert artifact.media_type.endswith("wordprocessingml.document")
    assert document.paragraphs[0].text == "BÁO CÁO THỬ NGHIỆM"
    assert document.paragraphs[0].style.name == "Title"
    assert document.paragraphs[1].text == "Ngày tạo: 22/07/2026"

    paragraphs = {paragraph.text: paragraph for paragraph in document.paragraphs}
    assert paragraphs["Tổng quan"].style.name == "Heading 1"
    assert sum(
        paragraph.text == "BÁO CÁO THỬ NGHIỆM"
        for paragraph in document.paragraphs
    ) == 1
    assert paragraphs["Việc thứ nhất"].style.name == "List Bullet"
    assert paragraphs["Việc thứ nhất"].paragraph_format.left_indent is not None
    assert paragraphs["Việc con"].paragraph_format.left_indent > paragraphs[
        "Việc thứ nhất"
    ].paragraph_format.left_indent
    assert paragraphs["Bước ba"].style.name == "List Number"
    assert paragraphs["Bước bốn"].style.name == "List Number"
    assert paragraphs["Ghi chú quan trọng."].style.name == "Quote"


def test_docx_renders_inline_formatting_and_table():
    _, document = _open_artifact()

    content_paragraph = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith("Nội dung")
    )
    assert any(run.text == "tiếng Việt đậm" and run.bold for run in content_paragraph.runs)
    assert any(run.text == "in nghiêng" and run.italic for run in content_paragraph.runs)
    assert any(run.text == "đã bỏ" and run.font.strike for run in content_paragraph.runs)

    assert len(document.tables) == 1
    table_rows = [
        [cell.text for cell in row.cells]
        for row in document.tables[0].rows
    ]
    assert table_rows == [
        ["Người phụ trách", "Công việc"],
        ["Anh Tú", "Kiểm thử DOCX"],
    ]
    assert all(run.bold for run in document.tables[0].rows[0].cells[0].paragraphs[0].runs)


def test_markdown_body_keeps_h2_and_removes_only_leading_h1():
    markdown = "\n# Tiêu đề\n\n## Tổng quan\n\nNội dung"

    assert _markdown_body(markdown) == "## Tổng quan\n\nNội dung"
    assert _markdown_body("## Tổng quan") == "## Tổng quan"


def test_docx_promotes_body_headings_after_template_title():
    artifact = export_docx(
        "## Đề mục lớn\n\n### Đề mục con",
        "Tiêu đề",
        generated_at="23/07/2026",
    )
    document = docx.Document(BytesIO(artifact.content))
    paragraphs = {paragraph.text: paragraph for paragraph in document.paragraphs}

    assert paragraphs["Đề mục lớn"].style.name == "Heading 1"
    assert paragraphs["Đề mục con"].style.name == "Heading 2"


def test_modern_template_renders_optional_context():
    artifact = export_docx(
        MARKDOWN,
        "Báo cáo hiện đại",
        template="modern",
        generated_at="23/07/2026",
        context={"author": "Anh Tú", "department": "AI Team"},
    )
    document = docx.Document(BytesIO(artifact.content))
    with ZipFile(BytesIO(artifact.content)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    assert artifact.filename == "Báo cáo hiện đại.docx"
    assert "BÁO CÁO THỬ NGHIỆM" in document_xml
    assert "Báo cáo hiện đại" not in document_xml
    assert "23/07/2026" in document_xml
    assert "Anh Tú" in document_xml
    assert "{{" not in document_xml
    assert any(
        paragraph.text == "Tổng quan"
        and paragraph.style.name == "Heading 1"
        for paragraph in document.paragraphs
    )


def test_modern_template_requires_author_context():
    with pytest.raises(
        ValueError,
        match="modern.*requires context fields: author",
    ):
        export_docx(
            "## Tổng quan",
            "Báo cáo hiện đại",
            template="modern",
        )


def test_template_context_cannot_replace_system_fields():
    artifact = export_docx(
        "## Tổng quan\n\nNội dung thật",
        "Tiêu đề thật",
        generated_at="23/07/2026",
        context={
            "title": "Tiêu đề giả",
            "generated_at": "01/01/1970",
            "time": "00:00",
            "body": "Nội dung giả",
        },
    )
    document = docx.Document(BytesIO(artifact.content))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)

    assert "TIÊU ĐỀ THẬT" in text
    assert "Tiêu đề giả" not in text
    assert "23/07/2026" in text
    assert "01/01/1970" not in text
    assert "Nội dung thật" in text
    assert "Nội dung giả" not in text


def test_docx_does_not_fetch_or_embed_unsafe_resources():
    artifact, _ = _open_artifact()

    with ZipFile(BytesIO(artifact.content)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
        relationships_xml = archive.read(
            "word/_rels/document.xml.rels"
        ).decode("utf-8")

    assert "{{" not in document_xml
    assert "https://openai.com" in relationships_xml
    assert "file:///etc/passwd" not in relationships_xml
    assert "https://example.com/secret.png" not in relationships_xml
    assert "[Hình: Không tải ảnh]" in document_xml


def test_export_service_sanitizes_unicode_filename():
    artifact = create_export_service().export(
        "Xin chào",
        "docx",
        "Biên bản: Nhóm A/2026",
    )

    assert artifact.filename == "Biên bản Nhóm A2026.docx"
    assert docx.Document(BytesIO(artifact.content)).paragraphs[0].text == (
        "BIÊN BẢN NHÓM A2026"
    )


def test_missing_docxtpl_has_clear_error(monkeypatch):
    real_import = builtins.__import__

    def import_without_docxtpl(name, *args, **kwargs):
        if name == "docxtpl" or name.startswith("docxtpl."):
            raise ModuleNotFoundError("No module named 'docxtpl'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_docxtpl)

    with pytest.raises(MissingExportDependency, match="python-docx and docxtpl"):
        _load_docx_template_class()
