from typing import TypedDict, List, Optional, Any


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


def validate_transcript(result: TranscriptResult) -> None:
    # ===== key =====
    if not isinstance(result.get("key"), str) or not result["key"].strip():
        raise ValueError("key must be a non-empty string")

    # ===== duration =====
    duration = result.get("duration")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError("duration must be a number > 0")

    # ===== sentence_info =====
    sentence_info = result.get("sentence_info")
    if not isinstance(sentence_info, list):
        raise ValueError("sentence_info must be a list")

    for idx, s in enumerate(sentence_info):
        if not isinstance(s, dict):
            raise ValueError(f"sentence_info[{idx}] must be dict")

        # text
        text = s.get("text", "")
        if text is not None and not isinstance(text, str):
            raise ValueError(f"sentence_info[{idx}].text must be string")

        # start/end
        start = s.get("start")
        end = s.get("end")

        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"sentence_info[{idx}] start/end must be numbers")

        if start < 0 or end < 0:
            raise ValueError(f"sentence_info[{idx}] start/end must be >= 0")

        if start > end:
            raise ValueError(f"sentence_info[{idx}] start > end")

        if start > duration or end > duration:
            raise ValueError(f"sentence_info[{idx}] out of duration range")

        # speaker
        speaker = s.get("speaker")
        if speaker is not None and not isinstance(speaker, str):
            raise ValueError(f"sentence_info[{idx}].speaker must be string or None")



def normalize_sentence_info(result: TranscriptResult) -> TranscriptResult:
    duration = result["duration"]

    cleaned = []

    # ===== 1. normalize fields =====
    for s in result["sentence_info"]:
        cleaned.append({
            "text": (s.get("text") or "").strip(),
            "start": float(s["start"]),
            "end": float(s["end"]),
            "speaker": (s.get("speaker") or "Unknown").strip() or "Unknown"
        })

    # ===== 2. sort by start =====
    cleaned.sort(key=lambda x: x["start"])

    # ===== 3. merge overlaps (improved) =====
    merged = []
    for cur in cleaned:
        # clamp to duration
        cur["start"] = min(max(cur["start"], 0), duration)
        cur["end"] = min(max(cur["end"], 0), duration)

        if not merged:
            merged.append(cur)
            continue

        last = merged[-1]

        # overlap or touching
        if cur["start"] <= last["end"]:
            last["end"] = max(last["end"], cur["end"])

            # better merge text (concat instead of replace)
            if cur["text"]:
                if last["text"]:
                    last["text"] += " " + cur["text"]
                else:
                    last["text"] = cur["text"]

            # unify speaker if needed
            if last["speaker"] == "Unknown" and cur["speaker"] != "Unknown":
                last["speaker"] = cur["speaker"]

        else:
            merged.append(cur)

    result["sentence_info"] = merged

    # ===== 4. fallback text =====
    if not result.get("text") or not result["text"].strip():
        result["text"] = " ".join(s["text"] for s in merged).strip()

    return result


def process_transcript(result: TranscriptResult) -> TranscriptResult:
    # normalize trước
    result = normalize_sentence_info(result)

    # validate lại sau normalize
    validate_transcript(result)

    return result