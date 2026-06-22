from typing import TypedDict, List, Optional

# Format input, validate và chuẩn hóa tuần theo notion:
# https://app.notion.com/share/fb445d06d0954d1ebd56e7c56643921b/386b0a500c9281339be600a9fbf5c32c?fbclid=IwY2xjawSlmpFleHRuA2FlbQIxMABicmlkETFFaTl5T3hWU0wyekI4Q0p2c3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHqh5o5C5Ebv6t9AArbjM36knQfehaCAR8RCs5c81kgHAaJdgmeTej1YdGAji_aem_xO9gLT9UmlsWm45_oMrh-Q

class SentenceInfo(TypedDict):
    text: str
    start: float
    end: float
    speaker: Optional[str]


class TranscriptResult(TypedDict):
    key: str
    text: str
    duration: float
    sentence_info: List[SentenceInfo]

# Hàm này để chuyển đầu ra của khối trước đó về format input của mình
def convert_input():
    pass


def validate_transcript(result: TranscriptResult) -> None:
    # ===== key =====
    if not result.get("key") or not isinstance(result["key"], str):
        raise ValueError("key must be a non-empty string")

    # ===== duration =====
    duration = result.get("duration")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError("duration must be a number > 0")

    # ===== text =====
    text = result.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError("text must be a string or None")

    # ===== sentence_info =====
    sentence_info = result.get("sentence_info")
    if not isinstance(sentence_info, list):
        raise ValueError("sentence_info must be a list")

    for s in sentence_info:
        if not isinstance(s, dict):
            raise ValueError("each sentence_info item must be a dict")

        start = s.get("start")
        end = s.get("end")

        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError("start/end must be numbers")

        if start < 0 or end < 0:
            raise ValueError("start/end must be >= 0")

        if start > end:
            raise ValueError(f"start must be <= end: {start} > {end}")

        if start > result["duration"] or end > result["duration"]:
            raise ValueError("start/end must be within duration")


def normalize_sentence_info(result: TranscriptResult) -> TranscriptResult:
    sentence_info = result["sentence_info"]

    cleaned = []

    # ===== 1. clean + speaker normalize =====
    for s in sentence_info:
        cleaned.append({
            "text": s.get("text", ""),
            "start": float(s["start"]),
            "end": float(s["end"]),
            "speaker": s.get("speaker") or "Unknown"
        })

    # ===== 2. sort =====
    cleaned.sort(key=lambda x: x["start"])

    # ===== 3. merge overlaps =====
    merged = []
    for item in cleaned:
        if not merged:
            merged.append(item)
            continue

        last = merged[-1]

        if item["start"] <= last["end"]:
            # overlap → merge
            last["end"] = max(last["end"], item["end"])

            if len(item["text"]) > len(last["text"]):
                last["text"] = item["text"]
        else:
            merged.append(item)

    result["sentence_info"] = merged

    # ===== 4. fallback text =====
    if not result.get("text"):
        result["text"] = " ".join(s["text"] for s in merged).strip()

    return result