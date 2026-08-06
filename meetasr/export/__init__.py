"""Markdown, DOCX and PDF document export."""

from meetasr.export.markdown_exporter import export_markdown
from meetasr.export.service import (
    ExportArtifact,
    ExportService,
    MissingExportDependency,
    UnsafeExportResource,
    UnsupportedExportFormat,
    sanitize_filename,
)


def _export_docx(markdown: str, title: str) -> ExportArtifact:
    from meetasr.export.docx_exporter import export_docx

    return export_docx(markdown, title)


def _export_pdf(markdown: str, title: str) -> ExportArtifact:
    from meetasr.export.pdf_exporter import export_pdf

    return export_pdf(markdown, title)


def create_export_service() -> ExportService:
    service = ExportService()
    service.register("md", export_markdown)
    service.register("docx", _export_docx)
    service.register("pdf", _export_pdf)
    return service


__all__ = [
    "ExportArtifact",
    "ExportService",
    "MissingExportDependency",
    "UnsupportedExportFormat",
    "UnsafeExportResource",
    "create_export_service",
    "sanitize_filename",
]
