"""Markdown, DOCX and PDF document export."""

from collections.abc import Mapping
from typing import Any, cast

from meetasr.export.markdown_exporter import export_markdown
from meetasr.export.service import (
    ExportArtifact,
    ExportPreset,
    ExportService,
    MissingExportDependency,
    UnsafeExportResource,
    UnsupportedExportFormat,
    UnsupportedExportPreset,
    sanitize_filename,
)


def _export_markdown(
    markdown: str,
    title: str,
    preset: str | None,
    context: Mapping[str, Any] | None,
) -> ExportArtifact:
    return export_markdown(markdown, title)


def _export_docx(
    markdown: str,
    title: str,
    preset: str | None,
    context: Mapping[str, Any] | None,
) -> ExportArtifact:
    from meetasr.export.docx_exporter import export_docx

    return export_docx(
        markdown,
        title,
        template=cast(Any, preset),
        context=context,
    )


def _export_pdf(
    markdown: str,
    title: str,
    preset: str | None,
    context: Mapping[str, Any] | None,
) -> ExportArtifact:
    from meetasr.export.pdf_exporter import export_pdf

    return export_pdf(
        markdown,
        title,
        template=cast(Any, preset),
        context=context,
    )


def create_export_service() -> ExportService:
    service = ExportService()
    service.register("md", _export_markdown)
    service.register("docx", _export_docx)
    service.register("pdf", _export_pdf)
    return service


__all__ = [
    "ExportArtifact",
    "ExportPreset",
    "ExportService",
    "MissingExportDependency",
    "UnsupportedExportFormat",
    "UnsupportedExportPreset",
    "UnsafeExportResource",
    "create_export_service",
    "sanitize_filename",
]
