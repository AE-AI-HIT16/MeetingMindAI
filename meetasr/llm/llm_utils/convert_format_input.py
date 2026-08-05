def format_sentence_info(sentence_info: list[dict]) -> str:
    lines = []

    for item in sentence_info:
        start = item.get("start", 0)
        speaker = item.get("speaker", "Unknown")
        text = item.get("text", "")

        line = f"[{int(start)}][{speaker}]: {text}"
        lines.append(line)

    return "\n".join(lines)
