"""Render safe Markdown into registered HTML/CSS PDF templates."""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from markdown_it.token import Token

from meetasr.export.markdown_parser import (
    parse_markdown,
    render_safe_html_tokens,
    resolve_document_title,
    strip_leading_document_heading,
)
from meetasr.export.service import (
    ExportArtifact,
    MissingExportDependency,
    UnsafeExportResource,
)

PdfTemplateName = Literal["minimal", "blue_modern"]

_PDF_MEDIA_TYPE = "application/pdf"
_TEMPLATE_DIRECTORY = Path(__file__).resolve().parent / "templates" / "pdf"
_FONT_DIRECTORY = Path(__file__).resolve().parent / "fonts"
_TEMPLATES: dict[str, Path] = {
    "minimal": _TEMPLATE_DIRECTORY / "minimal.html",
    "blue_modern": _TEMPLATE_DIRECTORY / "blue_modern.html",
}
_SYSTEM_CONTEXT_KEYS = frozenset(
    {
        "title",
        "generated_at",
        "time",
        "brand_name",
        "language",
        "sections",
        "poppins_bold_url",
        "body_font_url",
    }
)


@dataclass(frozen=True)
class PdfSection:
    """One LLM-generated Markdown section rendered for a PDF template."""

    heading: str
    html: str


def _load_pdf_dependencies():
    try:
        from jinja2 import Environment, StrictUndefined, meta, select_autoescape
        from weasyprint import HTML, URLFetcher
    except ImportError as exc:
        raise MissingExportDependency(
            "PDF template export requires Jinja2 and WeasyPrint. "
            "Install with: pip install -e '.[export]'"
        ) from exc

    return (
        HTML,
        URLFetcher,
        Environment,
        StrictUndefined,
        meta,
        select_autoescape,
    )


def _safe_url_fetcher(default_url_fetcher):
    def fetch(url: str):
        parsed = urlparse(url)
        if parsed.scheme != "data":
            raise UnsafeExportResource(
                f"External PDF resource is forbidden: {parsed.scheme or 'relative'}"
            )
        return default_url_fetcher(url)

    return fetch


def _create_url_fetcher(URLFetcher):
    """Create a WeasyPrint 69+ fetcher restricted to embedded data URLs."""

    class EmbeddedDataURLFetcher(URLFetcher):
        def fetch(self, url: str, headers: dict[str, str] | None = None):
            parsed = urlparse(url)
            if parsed.scheme != "data":
                raise UnsafeExportResource(
                    "External PDF resource is forbidden: "
                    f"{parsed.scheme or 'relative'}"
                )
            return super().fetch(url, headers)

    return EmbeddedDataURLFetcher(
        allowed_protocols={"data"},
        allow_redirects=False,
        fail_on_errors=True,
    )


def _resolve_template(template: str) -> Path:
    try:
        path = _TEMPLATES[template]
    except KeyError as exc:
        available = ", ".join(sorted(_TEMPLATES))
        raise ValueError(
            f"Unknown PDF template '{template}'. Available: {available}"
        ) from exc

    if not path.is_file():
        raise FileNotFoundError(f"PDF template not found: {path}")
    return path


def _font_data_url(filename: str) -> str:
    path = _FONT_DIRECTORY / filename
    if not path.is_file():
        raise RuntimeError(f"Bundled font not found: {path}")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:font/ttf;base64,{encoded}"


def _section_heading(tokens: Sequence[Token], heading_index: int) -> str:
    inline_index = heading_index + 1
    if inline_index < len(tokens) and tokens[inline_index].type == "inline":
        return tokens[inline_index].content.strip()
    return ""


def _extract_sections(markdown: str) -> list[PdfSection]:
    """Convert top-level body headings (##) into dynamic template sections."""
    body_markdown = strip_leading_document_heading(markdown)
    tokens = parse_markdown(body_markdown)
    sections: list[PdfSection] = []
    preamble: list[Token] = []
    current_heading = ""
    current_tokens: list[Token] = []
    index = 0

    def append_section(heading: str, content_tokens: list[Token]) -> None:
        html = render_safe_html_tokens(content_tokens).strip()
        if heading or html:
            sections.append(PdfSection(heading=heading, html=html))

    while index < len(tokens):
        token = tokens[index]
        if token.type == "heading_open" and token.tag == "h2":
            if current_heading:
                append_section(current_heading, current_tokens)
            elif preamble:
                append_section("", preamble)
                preamble = []

            current_heading = _section_heading(tokens, index)
            current_tokens = []
            index += 1
            while index < len(tokens) and tokens[index].type != "heading_close":
                index += 1
        elif current_heading:
            current_tokens.append(token)
        else:
            preamble.append(token)
        index += 1

    if current_heading:
        append_section(current_heading, current_tokens)
    elif preamble:
        append_section("", preamble)

    return sections


def _missing_context_fields(
    requested: set[str],
    context: Mapping[str, Any] | None,
) -> list[str]:
    supplied = context or {}
    custom_fields = requested - _SYSTEM_CONTEXT_KEYS
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


def export_pdf(
    markdown: str,
    title: str,
    *,
    template: PdfTemplateName = "minimal",
    generated_at: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> ExportArtifact:
    """Export Markdown through a trusted HTML/CSS template.

    A leading Markdown H1 becomes the uppercase display title and is omitted
    from the body. The ``title`` argument remains the output filename fallback.
    """
    (
        HTML,
        URLFetcher,
        Environment,
        StrictUndefined,
        jinja_meta,
        select_autoescape,
    ) = _load_pdf_dependencies()
    template_path = _resolve_template(template)
    document_title = resolve_document_title(markdown, title)
    template_source = template_path.read_text(encoding="utf-8")
    environment = Environment(
        autoescape=select_autoescape(("html", "xml")),
        undefined=StrictUndefined,
    )
    parsed_template = environment.parse(template_source)
    requested = jinja_meta.find_undeclared_variables(parsed_template)
    missing_fields = _missing_context_fields(requested, context)
    if missing_fields:
        fields = ", ".join(missing_fields)
        raise ValueError(
            f"PDF template '{template}' requires context fields: {fields}"
        )

    now = datetime.now()
    template_context: dict[str, Any] = {
        key: value
        for key, value in (context or {}).items()
        if key not in _SYSTEM_CONTEXT_KEYS
    }
    template_context.update(
        {
            "title": document_title,
            "generated_at": generated_at or now.strftime("%d/%m/%Y"),
            "time": now.strftime("%H:%M"),
            "brand_name": "MeetingMind AI",
            "language": "vi",
            "sections": _extract_sections(markdown),
            "poppins_bold_url": _font_data_url("Poppins-Bold.ttf"),
            "body_font_url": _font_data_url("NotoSans-Regular.ttf"),
        }
    )
    html_document = environment.from_string(template_source).render(
        template_context
    )

    output = io.BytesIO()
    HTML(
        string=html_document,
        url_fetcher=_create_url_fetcher(URLFetcher),
    ).write_pdf(output)
    return ExportArtifact(
        content=output.getvalue(),
        media_type=_PDF_MEDIA_TYPE,
        filename=f"{title}.pdf",
    )
