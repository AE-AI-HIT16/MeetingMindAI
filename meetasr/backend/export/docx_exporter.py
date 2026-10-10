"""Render Markdown into a trusted DOCX template."""

from __future__ import annotations

import io
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from meetasr.backend.export.docx_markdown import render_markdown
from meetasr.backend.export.markdown_parser import (
    resolve_document_title,
    strip_leading_document_heading,
)
from meetasr.backend.export.service import ExportArtifact, MissingExportDependency


DocxTemplateName = Literal["minimal", "modern"]

_DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document"
)
_TEMPLATE_DIRECTORY = Path(__file__).resolve().parent / "templates" / "docx"
_TEMPLATES: dict[str, Path] = {
    "minimal": _TEMPLATE_DIRECTORY / "minimal.docx",
    "modern": _TEMPLATE_DIRECTORY / "modern.docx",
}
_RESERVED_CONTEXT_KEYS = frozenset({"title", "generated_at", "time", "body"})


def _markdown_body(markdown: str) -> str:
    """Compatibility wrapper for the shared template-body normalization."""
    return strip_leading_document_heading(markdown)


def _load_docx_template_class():
    """Load optional DOCX dependencies only when DOCX export is requested."""
    try:
        from docxtpl import DocxTemplate
    except (ImportError, ModuleNotFoundError) as exc:
        raise MissingExportDependency(
            "DOCX export requires python-docx and docxtpl. "
            "Install with: pip install -e '.[export]'"
        ) from exc

    return DocxTemplate


def _resolve_template(template: str) -> Path:
    """Resolve a trusted template name without accepting arbitrary file paths."""
    try:
        path = _TEMPLATES[template]
    except KeyError as exc:
        available = ", ".join(sorted(_TEMPLATES))
        raise ValueError(
            f"Unknown DOCX template '{template}'. Available: {available}"
        ) from exc

    if not path.is_file():
        raise FileNotFoundError(f"DOCX template not found: {path}")
    return path


def _missing_context_fields(
    document: Any,
    context: Mapping[str, Any] | None,
) -> list[str]:
    """Return template-specific variables that have no usable value."""
    supplied = context or {}
    requested = document.get_undeclared_template_variables()
    custom_fields = requested - _RESERVED_CONTEXT_KEYS

    return sorted(
        field
        for field in custom_fields
        if field not in supplied
        or supplied[field] is None
        or (
            isinstance(supplied[field], str)
            and not supplied[field].strip()
        )
    )


def export_docx(
    markdown: str,
    title: str,
    *,
    template: DocxTemplateName = "minimal",
    generated_at: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> ExportArtifact:
    """Export Markdown using a registered DOCX template.

    The exporter owns the system fields ``title``, ``generated_at``, ``time``
    and ``body``. Template-specific fields such as ``author`` or ``department``
    can be supplied through ``context``. Extra fields are harmless when the
    selected template does not reference them.

    A leading Markdown H1 becomes the uppercase display title and is omitted
    from the body. The ``title`` argument remains the output filename fallback.
    """
    DocxTemplate = _load_docx_template_class()
    template_path = _resolve_template(template)
    document_title = resolve_document_title(markdown, title)

    document = DocxTemplate(str(template_path))
    missing_fields = _missing_context_fields(document, context)
    if missing_fields:
        fields = ", ".join(missing_fields)
        raise ValueError(
            f"DOCX template '{template}' requires context fields: {fields}"
        )

    try:
        body = document.new_subdoc()
    except (ImportError, ModuleNotFoundError) as exc:
        raise MissingExportDependency(
            "DOCX template rendering requires docxcompose. "
            "Install with: pip install -e '.[export]'"
        ) from exc
    # The template owns the document title, so body sections start at Word
    # Heading 1 even though canonical Markdown represents them with ##.
    render_markdown(
        body,
        _markdown_body(markdown),
        heading_level_offset=-1,
    )

    now = datetime.now()
    template_context: dict[str, Any] = {
        key: value
        for key, value in (context or {}).items()
        if key not in _RESERVED_CONTEXT_KEYS
    }
    template_context.update(
        {
            "title": document_title,
            "generated_at": generated_at or now.strftime("%d/%m/%Y"),
            "time": now.strftime("%H:%M"),
            "body": body,
        }
    )
    document.render(template_context, autoescape=True)

    output = io.BytesIO()
    document.save(output)
    return ExportArtifact(
        content=output.getvalue(),
        media_type=_DOCX_MEDIA_TYPE,
        filename=f"{title}.docx",
    )
