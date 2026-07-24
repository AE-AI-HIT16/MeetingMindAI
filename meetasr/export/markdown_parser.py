"""Parse untrusted Markdown for DOCX/PDF export."""

from __future__ import annotations

import re
from collections.abc import Sequence

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
_LEADING_H1 = re.compile(
    r"\A(?:\ufeff)?(?:[ \t]*\r?\n)*[ \t]{0,3}#(?!#)[ \t]+[^\r\n]*"
    r"(?:\r?\n|\Z)"
)

def create_markdown_parser() -> MarkdownIt:
    """HTML from the original Markdown is disabled."""
    parser = MarkdownIt("commonmark", {"html": False})
    parser.enable(["table", "strikethrough"])
    return parser

def parse_markdown(markdown: str) -> list[Token]:
    return create_markdown_parser().parse(markdown)


def strip_leading_document_heading(markdown: str) -> str:
    """Remove an initial ATX H1 when a document template owns the title."""
    return _LEADING_H1.sub("", markdown, count=1).lstrip("\r\n")


def _sanitize_html(rendered: str) -> str:
    return bleach.clean(
        rendered,
        tags=_ALLOWED_HTML_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        protocols=[],
        strip=True,
    )


def render_safe_html(markdown: str) -> str:
    """Render Markdown and strip active links/images/raw HTML."""
    rendered = create_markdown_parser().render(markdown)
    return _sanitize_html(rendered)


def render_safe_html_tokens(tokens: Sequence[Token]) -> str:
    """Render an already parsed token slice and sanitize its HTML."""
    parser = create_markdown_parser()
    rendered = parser.renderer.render(
        list(tokens),
        parser.options,
        {},
    )
    return _sanitize_html(rendered)
