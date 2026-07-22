"""Secure Markdown to PDF exporter."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from urllib.parse import urlparse

from meetasr.export.markdown_parser import render_safe_html
from meetasr.export.service import (
    ExportArtifact,
    MissingExportDependency,
    UnsafeExportResource,
)


_FONT_PATH = Path(__file__).parent / "fonts" / "NotoSans-Regular.ttf"


def _load_weasyprint():
    try:
        from weasyprint import CSS, HTML, default_url_fetcher
    except ImportError as exc:
        raise MissingExportDependency(
            "PDF export requires WeasyPrint. "
            "Install with: pip install -e '.[export]'"
        ) from exc

    return HTML, CSS, default_url_fetcher

def _safe_url_fetcher(default_url_fetcher):
    def fetch(url: str):
        parsed = urlparse(url)
        if parsed.scheme != "data":
            raise UnsafeExportResource(
                f"External resource is forbidden: {parsed.scheme or 'relative'}"
            )
        return default_url_fetcher(url)
    return fetch

def _font_data_url() -> str:
    if not _FONT_PATH.is_file():
        raise RuntimeError(f"Bundled font not found: {_FONT_PATH}")

    encoded = base64.b64encode(_FONT_PATH.read_bytes()).decode("ascii")
    return f"data:font/ttf;base64,{encoded}"

def export_pdf(markdown: str, title: str) -> ExportArtifact:
    HTML, CSS, default_url_fetcher = _load_weasyprint()

    body = render_safe_html(markdown)
    safe_title = html.escape(title)

    html_document = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>{safe_title}</title>
</head>
<body>{body}</body>
</html>
"""

    css = f"""
    @font-face {{
        font-family: "Noto Sans";
        src: url("{_font_data_url()}") format("truetype");
    }}

    @page {{
        size: A4;
        margin: 20mm;
    }}

    body {{
        font-family: "Noto Sans", sans-serif;
        font-size: 11pt;
        line-height: 1.55;
        color: #222;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
    }}

    th, td {{
        border: 1px solid #aaa;
        padding: 6px;
        vertical-align: top;
    }}

    pre {{
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }}
    """

    output = io.BytesIO()
    HTML(
        string=html_document,
        url_fetcher=_safe_url_fetcher(default_url_fetcher),
    ).write_pdf(output, stylesheets=[CSS(string=css)])

    return ExportArtifact(
        content=output.getvalue(),
        media_type="application/pdf",
        filename=f"{title}.pdf",
    )