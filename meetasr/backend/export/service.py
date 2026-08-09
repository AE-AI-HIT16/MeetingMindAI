"""Format-independent document export service."""


from __future__ import annotations


import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Literal


ExportFormat = Literal["md","docx","pdf"]

class UnsupportedExportFormat(ValueError):
    """Requested export format is not registered."""


class MissingExportDependency(RuntimeError):
    """An optional export dependency is not installed."""


class UnsafeExportResource(ValueError):
    """Document attempted to access a forbidden resource."""

@dataclass(frozen=True)
class ExportArtifact:
    """Binary file produced by an exporter."""
    content: bytes
    media_type: str
    filename: str

Exporter = Callable[[str,str], ExportArtifact]
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

def sanitize_filename(title: str, max_length: int =80) -> str:
    """Preserve Unicode while removing unsafe filename characters."""
    value = unicodedata.normalize("NFC",title)
    value = _INVALID_FILENAME.sub("",value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    if not value:
        value = "document"
    if value.upper() in _WINDOWS_RESERVED:
        value = f"_{value}"
    return value[:max_length].rstrip(" .") or "document"

class ExportService:
    """Convert canonical Markdown to downloadable formats."""

    def __init__(self) -> None:
        self._exporters: dict[str, Exporter] = {}
    
    def register(self, format_name: str, exprorter: Exporter) -> None:
        key = format_name.strip().lower()
        if not key:
            raise ValueError("format name must not be empty")
        self._exporters[key] = exprorter
    
    @property
    def supported_formats(self) -> tuple[str, ...]:
        return tuple(sorted(self._exporters))
    
    def export(
        self,
        markdown: str,
        format: ExportFormat,
        title: str = "document",
    ) -> ExportArtifact:
        """Export Markdown without depending on DB, API or LLM."""
        if not isinstance(markdown, str):
            raise TypeError("markdown must be a string")
        key = format.strip().lower()
        exporter = self._exporters.get(key)
        if exporter is None:
            raise UnsupportedExportFormat(
                f"Unsupported format '{format}'. "
                f"Available: {', '.join(self.supported_formats)}"
            )
        return exporter(markdown, sanitize_filename(title))