"""Validation helpers for DocumentPlanner LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any

MAX_OUTLINE_SECTIONS = 7
MAX_LABEL_CHARS = 200
MAX_TITLE_WORDS = 14
DRAFT_SEPARATOR = "\n\n---\n\n"
MAX_REPAIR_RESPONSE_CHARS = 5000
PLAN_RESPONSE_FORMAT = {"type": "json_object"}
GENERIC_HEADINGS = {
    "phân tích theo chủ đề",
    "các điểm chính",
    "kết luận",
    "trích dẫn nổi bật",
}


def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract one JSON object from a plain or fenced LLM response.

    Args:
        raw: Raw text returned by the LLM.

    Returns:
        The decoded JSON object.

    Raises:
        TypeError: If the response is not text.
        ValueError: If the decoded value is not an object.
        json.JSONDecodeError: If no valid JSON object can be decoded.
    """
    if not isinstance(raw, str):
        raise TypeError("LLM response must be a string")
    match = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    text = match.group(1).strip() if match else raw.strip()
    if not text:
        raise ValueError("LLM returned an empty response")
    if not match:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    parsed = json.loads(text, strict=False)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must decode to a JSON object")
    return parsed


def build_json_repair_prompt(raw: object, error: Exception) -> str:
    """Build a bounded request that fixes JSON syntax without changing content."""
    response = raw if isinstance(raw, str) else repr(raw)
    response = response[:MAX_REPAIR_RESPONSE_CHARS]
    return (
        "JSON kế hoạch dưới đây bị lỗi cú pháp. Hãy sửa dấu phẩy, dấu ngoặc, "
        "dấu ngoặc kép hoặc ký tự escape để nó trở thành một JSON object hợp lệ. "
        "Giữ nguyên nội dung và cấu trúc content_kind/outline; không thêm giải thích "
        "và không dùng markdown code block.\n\n"
        f"Lỗi parser: {error}\n\nJSON cần sửa:\n{response}"
    )


def build_plan_retry_prompt(
    raw: object,
    error: Exception,
    transcript_sample: str,
) -> str:
    """Repair malformed JSON or regenerate a missing plan from its source."""
    if (
        isinstance(raw, str)
        and raw.strip()
        and isinstance(error, json.JSONDecodeError)
    ):
        return build_json_repair_prompt(raw, error)
    return (
        "Lần tạo kế hoạch trước không trả về nội dung. Hãy đọc lại transcript "
        "dưới đây và tạo mới một JSON object ngắn gọn. Bắt buộc có content_kind "
        "và outline; outline bắt đầu bằng Tổng quan/summary, sau đó là 2-6 chủ đề "
        "cụ thể có kind topic. Chỉ trả JSON, không giải thích.\n\n"
        f"TRANSCRIPT:\n{transcript_sample}"
    )


def validate_plan_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Reject syntactically valid plans that cannot produce useful topics."""
    title = data.get("content_kind")
    sections = normalize_outline(data.get("outline"))
    has_summary = any(section["kind"] == "summary" for section in sections)
    has_topic = any(
        section["kind"] != "summary"
        and section["heading"].lower() not in GENERIC_HEADINGS
        for section in sections
    )
    if not isinstance(title, str) or not title.strip():
        raise ValueError("plan is missing content_kind")
    if not has_summary or not has_topic:
        raise ValueError("plan needs a summary and at least one concrete topic")
    return data


def normalize_outline(value: object) -> list[dict[str, str]]:
    """Validate a bounded, open-vocabulary outline returned by the LLM."""
    if not isinstance(value, list):
        return []

    sections: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for position, raw in enumerate(value[:MAX_OUTLINE_SECTIONS], start=1):
        if not isinstance(raw, dict):
            continue

        section_id = safe_text(raw.get("id"), f"s{position}")[:MAX_LABEL_CHARS]
        while section_id in used_ids:
            section_id = f"s{position}_{len(used_ids) + 1}"
        used_ids.add(section_id)
        sections.append(
            {
                "id": section_id,
                "heading": safe_text(
                    raw.get("heading"),
                    "Tóm tắt",
                )[:MAX_LABEL_CHARS],
                "kind": safe_text(
                    raw.get("kind"),
                    "summary",
                )[:MAX_LABEL_CHARS],
            }
        )
    return sections


def normalize_report_outline(value: object) -> list[dict[str, str]]:
    """Return one overview followed by concrete transcript topic sections."""
    normalized = normalize_outline(value)
    summary = {"heading": "Tổng quan", "kind": "summary"}
    topics = []
    for section in normalized:
        heading = section["heading"]
        if section["kind"] == "summary" or heading.lower() in GENERIC_HEADINGS:
            continue
        topics.append({"heading": heading, "kind": "topic"})
        if len(topics) == 6:
            break
    if not topics:
        topics = [{"heading": "Nội dung chính", "kind": "topic"}]

    return [
        {
            "id": f"s{position}",
            "heading": section["heading"],
            "kind": section["kind"],
        }
        for position, section in enumerate([summary, *topics], start=1)
    ]


def fallback_outline() -> list[dict[str, str]]:
    """Return the documented fallback outline."""
    return normalize_report_outline([])


def normalize_title(value: object, default: str = "Tài liệu") -> str:
    """Return a single-line report title with a bounded word count."""
    title = " ".join(safe_text(value, default).split())
    return " ".join(title.split()[:MAX_TITLE_WORDS])


def safe_text(value: object, default: str) -> str:
    """Return stripped text or a safe default."""
    return value.strip() if isinstance(value, str) and value.strip() else default


def bounded_join(
    parts: list[str],
    max_chars: int,
    separator: str = DRAFT_SEPARATOR,
) -> str:
    """Join a representative slice of every draft within a strict budget."""
    if max_chars <= 0:
        return ""

    clean = [part.strip() for part in parts if part.strip()]
    joined = separator.join(clean)
    if len(joined) <= max_chars:
        return joined
    if not clean:
        return ""

    separator_budget = len(separator) * (len(clean) - 1)
    content_budget = max(1, max_chars - separator_budget)
    quota = max(1, content_budget // len(clean))
    bounded = [
        part if len(part) <= quota else f"{part[: max(0, quota - 1)]}…"
        for part in clean
    ]
    return separator.join(bounded)[:max_chars]
