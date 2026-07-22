"""Render Markdown into a trusted DOCX template."""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from typing import Literal

from meetasr.export.docx_markdown import render_markdown
from meetasr.export.service import ExportArtifact, MissingExportDependency


DocxTemplateName = Literal["minimal"]

_DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document"
)
_TEMPLATE_DIRECTORY = Path(__file__).resolve().parent / "templates" / "docx"
_TEMPLATES: dict[str, Path] = {
    "minimal": _TEMPLATE_DIRECTORY / "minimal.docx",
}


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


def export_docx(
    markdown: str,
    title: str,
    *,
    template: DocxTemplateName = "minimal",
    generated_at: str | None = None,
) -> ExportArtifact:
    """Export Markdown using the placeholders in ``minimal.docx``.

    The template owns page layout and visual styles. This function supplies its
    three placeholders: ``title``, ``generated_at`` and the ``body`` subdocument.
    """
    DocxTemplate = _load_docx_template_class()
    template_path = _resolve_template(template)

    document = DocxTemplate(str(template_path))
    try:
        body = document.new_subdoc()
    except (ImportError, ModuleNotFoundError) as exc:
        raise MissingExportDependency(
            "DOCX template rendering requires docxcompose. "
            "Install with: pip install -e '.[export]'"
        ) from exc
    render_markdown(body, markdown)

    document.render(
        {
            "title": title,
            "generated_at": generated_at or date.today().strftime("%d/%m/%Y"),
            "body": body,
        },
        autoescape=True,
    )

    output = io.BytesIO()
    document.save(output)
    return ExportArtifact(
        content=output.getvalue(),
        media_type=_DOCX_MEDIA_TYPE,
        filename=f"{title}.docx",
    )
