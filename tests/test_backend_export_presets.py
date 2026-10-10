"""Production export path honours the template preset chosen in the UI."""

from __future__ import annotations

import pytest

from meetasr.backend.export import UnsupportedExportPreset, create_export_service


def test_export_service_forwards_preset_to_exporter(monkeypatch):
    calls = []

    import meetasr.backend.export.docx_exporter as docx_exporter
    import meetasr.backend.export.pdf_exporter as pdf_exporter

    monkeypatch.setattr(docx_exporter, "export_docx",
                        lambda md, title, *, template, context: calls.append(("docx", template)))
    monkeypatch.setattr(pdf_exporter, "export_pdf",
                        lambda md, title, *, template, context: calls.append(("pdf", template)))

    service = create_export_service()
    service.export("# x", "docx", preset="modern")
    service.export("# x", "pdf", preset="blue_modern")
    service.export("# x", "pdf")  # default

    assert calls == [("docx", "modern"), ("pdf", "blue_modern"), ("pdf", "minimal")]


def test_unknown_preset_is_rejected():
    with pytest.raises(UnsupportedExportPreset):
        create_export_service().export("# x", "docx", preset="blue_modern")


@pytest.mark.parametrize("preset", ["minimal", "modern"])
def test_both_docx_templates_render(preset):
    pytest.importorskip("docxtpl")
    artifact = create_export_service().export(
        "# Tiêu đề\n\n## Mục\n\nNội dung.", "docx", preset=preset,
        context={"author": "An"},
    )
    assert artifact.content[:2] == b"PK"  # a real .docx


@pytest.mark.parametrize("preset", ["minimal", "blue_modern"])
def test_both_pdf_templates_render(preset):
    pytest.importorskip("weasyprint")
    artifact = create_export_service().export(
        "# Tiêu đề\n\n## Mục\n\nNội dung.", "pdf", preset=preset,
        context={"author": "An"},
    )
    assert artifact.content[:4] == b"%PDF"
