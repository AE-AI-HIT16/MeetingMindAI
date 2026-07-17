"""Generic document schema — content-agnostic (Phase 2).

See meet_docs/docs/14_document_planner_design.md §3.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class StructuredChunkResult:
    """Output of structured extraction for one chunk."""

    chunk_index: int
    summary: str = ""
    action_items: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class DocSection:
    """One section of a generated document.

    `kind` is open vocabulary — never validate against a closed enum.
    """

    id: str
    heading: str
    kind: str
    markdown: str = ""
    found: bool = True


@dataclass
class DocumentReport:
    """Full output of DocumentPlanner."""

    content_kind: str
    sections: list[DocSection] = field(default_factory=list)
    language: str = "vi"
    llm_model: str = ""
    processing_time: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to JSON-serializable dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to formatted JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        """Format as a human-readable Markdown document."""
        lines = [f"# {self.content_kind}", ""]
        for sec in self.sections:
            if sec.found and sec.markdown.strip():
                lines += [f"## {sec.heading}", "", sec.markdown, ""]
        return "\n".join(lines).strip()