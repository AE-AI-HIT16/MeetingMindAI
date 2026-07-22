"""Parse untrusted Markdown for DOCX/PDF export."""

from __future__ import annotations

import bleach
from markdown_it import MarkdownIt
from markdown_it.token import Token

_ALLOWED_HTML_TAGS = {
    "p", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "strong", "em", "s",
    "ul", "ol", "li",
    "blockquote",
    "pre", "code",
    "table", "thead", "tbody", "tr", "th", "td",
    "a",
}
_ALLOWED_ATTRIBUTES: dict[str, list[str]] = {}

def create_markdown_parser() -> MarkdownIt:
    """HTML from the original Markdown is disabled."""
    parser = MarkdownIt("commonmark", {"html": False})
    parser.enable(["table", "strikethrough"])
    return parser

def parse_markdown(markdown: str) -> list[Token]:
    return create_markdown_parser().parse(markdown)

def render_safe_html(markdown: str) -> str:
    """Render Markdown and strip active links/images/raw HTML."""
    rendered = create_markdown_parser().render(markdown)
    return bleach.clean(
        rendered,
        tags=_ALLOWED_HTML_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        protocols=[],
        strip=True
    )
