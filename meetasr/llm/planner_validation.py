"""Validation and normalization helpers for DocumentPlanner LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from meetasr.schemas_doc import DocSection


def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract and validate one JSON object from an LLM response.

    Args:
        raw: Raw response returned by an LLM client.

    Returns:
        Parsed JSON object.

    Raises:
        TypeError: If raw is not a string.
        ValueError: If the decoded JSON is not an object.
        json.JSONDecodeError: If no valid JSON can be decoded.
    """
    if not isinstance(raw, str):
        raise TypeError("LLM response must be a string")

    match = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    text = match.group(1).strip() if match else raw.strip()
    if not match:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    parsed = json.loads(text, strict=False)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must decode to a JSON object")
    return parsed


def normalize_extraction(data: object, chunk_index: int) -> dict[str, Any]:
    """Normalize one structured extraction and flatten its open-ended fields."""
    if not isinstance(data, dict):
        raise ValueError("structured extraction must be a JSON object")

    normalized: dict[str, Any] = {
        "chunk_index": chunk_index,
        "summary": _text(data.get("summary")),
    }
    for key, value in data.items():
        if key in {"chunk_index", "summary", "extra"}:
            continue
        if isinstance(key, str) and key and _has_content(value):
            normalized[key] = value

    extra = data.get("extra")
    if isinstance(extra, dict):
        for key, value in extra.items():
            if (
                isinstance(key, str)
                and key
                and key not in normalized
                and _has_content(value)
            ):
                normalized[key] = value
    return normalized


def normalize_outline(value: object) -> list[dict[str, Any]]:
    """Return a bounded outline with valid, unique section identifiers."""
    if not isinstance(value, list):
        return []

    outline: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for position, raw_section in enumerate(value[:6], start=1):
        if not isinstance(raw_section, dict):
            continue

        section_id = _text(raw_section.get("id"), f"s{position}")
        if section_id in used_ids:
            section_id = f"s{position}"
        while section_id in used_ids:
            section_id = f"{section_id}_{position}"
        used_ids.add(section_id)

        kind = _text(raw_section.get("kind"), "summary")
        heading = _text(raw_section.get("heading"), "Tóm tắt")
        source_keys = _normalize_source_keys(raw_section.get("source_keys"), kind)
        outline.append(
            {
                "id": section_id,
                "heading": heading,
                "kind": kind,
                "source_keys": source_keys,
            }
        )
    return outline


def normalize_written_sections(value: object) -> list[DocSection]:
    """Validate single-pass sections and discard empty generated content."""
    if not isinstance(value, list):
        return []

    sections: list[DocSection] = []
    used_ids: set[str] = set()
    for position, raw_section in enumerate(value[:6], start=1):
        if not isinstance(raw_section, dict):
            continue
        markdown = _text(raw_section.get("markdown"))
        if not markdown:
            continue

        section_id = _text(raw_section.get("id"), f"s{position}")
        if section_id in used_ids:
            section_id = f"s{position}"
        used_ids.add(section_id)
        sections.append(
            DocSection(
                id=section_id,
                heading=_text(raw_section.get("heading"), "Tóm tắt"),
                kind=_text(raw_section.get("kind"), "summary"),
                markdown=markdown,
                found=True,
            )
        )
    return sections


def join_text_parts(parts: list[str], max_chars: int) -> str:
    """Join all non-empty parts while respecting a strict character budget."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    clean_parts = [part.strip() for part in parts if part.strip()]
    joined = "\n\n".join(clean_parts)
    if len(joined) <= max_chars:
        return joined
    if not clean_parts:
        return ""

    separator_chars = 2 * (len(clean_parts) - 1)
    content_budget = max_chars - separator_chars
    if content_budget < len(clean_parts):
        return joined[:max_chars]

    bounded: list[str] = []
    remaining = content_budget
    for index, part in enumerate(clean_parts):
        remaining_parts = len(clean_parts) - index
        quota = max(1, remaining // remaining_parts)
        if len(part) > quota:
            part = "…" if quota == 1 else part[: quota - 1] + "…"
        bounded.append(part)
        remaining -= len(part)
    return "\n\n".join(bounded)[:max_chars]


def _normalize_source_keys(value: object, kind: str) -> list[str]:
    if kind == "summary":
        return ["summary"]
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = [kind]

    keys: list[str] = []
    for candidate in candidates:
        key = _text(candidate)
        if key and key != "summary" and key not in keys:
            keys.append(key)
    if kind not in keys:
        keys.append(kind)
    return keys


def _text(value: object, default: str = "") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _has_content(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return isinstance(value, (int, float, bool))
