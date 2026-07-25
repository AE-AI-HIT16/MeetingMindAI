"""Markdown, DOCX and PDF document export."""

from meetasr.export.docx_exporter import export_docx
from meetasr.export.markdown_exporter import export_markdown
from meetasr.export.pdf_exporter import export_pdf
from meetasr.export.service import (
    ExportArtifact,
    ExportService,
    MissingExportDependency,
    UnsupportedExportFormat,
    UnsafeExportResource,
    sanitize_filename,
)


def create_export_service() -> ExportService:
    service = ExportService()
    service.register("md", export_markdown)
    service.register("docx", export_docx)
    service.register("pdf", export_pdf)
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