"""VAD timestamp utilities — merge and align segment timestamps."""

from __future__ import annotations
import re

from meetasr.schemas import Segment, SentenceInfo


_SENTENCE_END_RE = re.compile(r"[^.!?。！？…]+(?:[.!?。！？…]+|$)")
_BOUNDARY_DEDUP_MS = 250


def merge_vad_segments(
    segments: list[Segment],
    max_merge_gap_ms: int = 300,
    max_segment_ms: int = 60000,
) -> list[Segment]:
    """Merge short adjacent VAD segments into longer ones.

    Args:
        segments: Input VAD segments sorted by start_ms.
        max_merge_gap_ms: Merge segments with gap ≤ this value.
        max_segment_ms: Don't merge if result would exceed this length.

    Returns:
        Merged segments list.
    """
    if not segments:
        return []

    merged: list[Segment] = [Segment(segments[0].start_ms, segments[0].end_ms)]
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg.start_ms - last.end_ms
        would_be_len = seg.end_ms - last.start_ms
        if gap <= max_merge_gap_ms and would_be_len < max_segment_ms:
            merged[-1] = Segment(last.start_ms, seg.end_ms)
        else:
            merged.append(Segment(seg.start_ms, seg.end_ms))
    return merged


def find_speech_gaps(
    sentences: list[SentenceInfo],
    vad_segments: list[Segment],
    duration_ms: int,
    min_gap_ms: int = 1000,
) -> list[Segment]:
    """Find timestamp gaps where external VAD still detects speech.

    A candidate gap means the ASR timeline has no emitted sentence for at
    least ``min_gap_ms``, while a VAD speech segment overlaps that same
    interval. The function only identifies candidates; it does not re-run ASR
    or change transcript text.
    """
    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")
    if min_gap_ms <= 0:
        raise ValueError("min_gap_ms must be greater than zero")

    ordered = []
    for sentence in sentences:
        start_ms = max(0, int(sentence.start * 1000))
        end_ms = min(duration_ms, int(sentence.end * 1000))
        if end_ms > start_ms:
            ordered.append(Segment(start_ms, end_ms))
    ordered.sort(key=lambda segment: (segment.start_ms, segment.end_ms))
    boundaries = [(0, ordered[0].start_ms)] if ordered else [(0, duration_ms)]
    boundaries.extend(
        (left.end_ms, right.start_ms)
        for left, right in zip(ordered, ordered[1:])
    )
    if ordered:
        boundaries.append((ordered[-1].end_ms, duration_ms))

    gaps = []
    for start_ms, end_ms in boundaries:
        start_ms = max(0, start_ms)
        end_ms = min(duration_ms, end_ms)
        if end_ms - start_ms < min_gap_ms:
            continue
        has_speech = any(
            min(end_ms, vad_segment.end_ms) > max(start_ms, vad_segment.start_ms)
            for vad_segment in vad_segments
        )
        if has_speech:
            gaps.append(Segment(start_ms, end_ms))
    return gaps


def clip_sentence_to_range(
    sentence: SentenceInfo,
    start_ms: int,
    end_ms: int,
) -> SentenceInfo | None:
    """Keep only characters whose timestamp midpoint is inside a time range."""
    retained = [
        (char, timestamp)
        for char, timestamp in zip(sentence.text, sentence.char_timestamps)
        if start_ms <= (timestamp[0] + timestamp[1]) / 2 < end_ms
    ]
    while retained and retained[0][0].isspace():
        retained.pop(0)
    while retained and retained[-1][0].isspace():
        retained.pop()
    if not retained:
        return None

    return SentenceInfo(
        text="".join(char for char, _ in retained),
        start=retained[0][1][0] / 1000.0,
        end=retained[-1][1][1] / 1000.0,
        char_timestamps=[timestamp for _, timestamp in retained],
    )


def merge_rescued_sentences(
    sentences: list[SentenceInfo],
    rescued_sentences: list[SentenceInfo],
) -> list[SentenceInfo]:
    """Insert non-overlapping, non-duplicate rescue text into a transcript."""
    merged = sorted(sentences, key=lambda sentence: (sentence.start, sentence.end))
    for rescued in sorted(
        rescued_sentences,
        key=lambda sentence: (sentence.start, sentence.end),
    ):
        following = min(
            (
                sentence
                for sentence in merged
                if sentence.start >= rescued.end
                and (sentence.start - rescued.end) * 1000 <= _BOUNDARY_DEDUP_MS
            ),
            key=lambda sentence: sentence.start,
            default=None,
        )
        if following is not None:
            rescued = _trim_repeated_boundary_word(rescued, following)
            if rescued is None:
                continue
        normalized = " ".join(rescued.text.casefold().split())
        overlaps_existing = any(
            rescued.start < sentence.end and sentence.start < rescued.end
            for sentence in merged
        )
        duplicates_existing = any(
            normalized == " ".join(sentence.text.casefold().split())
            for sentence in merged
        )
        if rescued.text and not overlaps_existing and not duplicates_existing:
            merged.append(rescued)
            merged.sort(key=lambda sentence: (sentence.start, sentence.end))
    return merged


def _trim_repeated_boundary_word(
    rescued: SentenceInfo,
    following: SentenceInfo,
) -> SentenceInfo | None:
    """Remove one duplicated word caused by decoding a rescue right-context."""
    rescued_words = list(re.finditer(r"\S+", rescued.text))
    following_words = list(re.finditer(r"\S+", following.text))
    if not rescued_words or not following_words:
        return rescued

    normalize_word = lambda word: re.sub(r"\W+", "", word.casefold())
    if normalize_word(rescued_words[-1].group()) != normalize_word(
        following_words[0].group()
    ):
        return rescued

    text = rescued.text[:rescued_words[-1].start()].rstrip()
    if not text:
        return None
    timestamps = rescued.char_timestamps[:len(text)]
    end = timestamps[-1][1] / 1000.0 if timestamps else rescued.end
    return SentenceInfo(
        text=text,
        start=rescued.start,
        end=end,
        char_timestamps=timestamps,
    )


def align_timestamps_to_global(
    char_timestamps: list[list[int]],
    offset_ms: int,
) -> list[list[int]]:
    """Shift char-level timestamps by VAD segment offset.

    Args:
        char_timestamps: [[start_ms, end_ms], ...] relative to segment start.
        offset_ms: Global offset in milliseconds.

    Returns:
        Timestamps adjusted to global timeline.
    """
    return [
        [t[0] + offset_ms, t[1] + offset_ms]
        for t in char_timestamps
    ]


def build_sentence_info(
    asr_results: list[dict],
    vad_segments: list[Segment],
    timestamp_offsets_ms: list[int] | None = None,
) -> list[SentenceInfo]:
    """Build SentenceInfo list from per-segment ASR results.

    Args:
        asr_results: List of ASR result dicts, one per VAD segment.
            Each has keys: "text", "timestamp" (char-level, relative).
        vad_segments: Corresponding VAD segments (same order). These remain
            the fallback sentence bounds when an ASR result has no timestamps.
        timestamp_offsets_ms: Global start of each audio chunk passed to ASR.
            When VAD padding is used, this is earlier than the matching VAD
            segment start and must be used to globalize ASR-relative times.

    Returns:
        List of SentenceInfo with global timestamps.
    """
    if timestamp_offsets_ms is None:
        timestamp_offsets_ms = [segment.start_ms for segment in vad_segments]
    if len(timestamp_offsets_ms) != len(vad_segments):
        raise ValueError(
            "timestamp_offsets_ms must have one entry per VAD segment"
        )

    sentences: list[SentenceInfo] = []
    for result, seg, timestamp_offset_ms in zip(
        asr_results, vad_segments, timestamp_offsets_ms
    ):
        text = result.get("text", "").strip()
        if not text:
            continue
        raw_ts = result.get("timestamp", [])
        global_ts = align_timestamps_to_global(raw_ts, timestamp_offset_ms)

        start_s = seg.start_ms / 1000.0
        end_s = seg.end_ms / 1000.0
        if global_ts:
            start_s = global_ts[0][0] / 1000.0
            end_s = global_ts[-1][1] / 1000.0

        sentences.append(SentenceInfo(
            text=text,
            start=start_s,
            end=end_s,
            char_timestamps=global_ts,
        ))
    return sentences


def align_punctuated_timestamps(
    raw_text: str,
    punc_text: str,
    raw_timestamps: list[list[int]],
) -> list[list[int]]:
    """Align char-level timestamps after punctuation model changes text length.

    Punctuation models (like vibert-capu) add commas, periods, and change casing.
    This function uses difflib to map the newly added characters to the timestamps
    of the surrounding original characters, ensuring `len(punc_text) == len(timestamps)`.
    """
    import difflib
    if not raw_timestamps or len(raw_text) != len(raw_timestamps):
        # Fallback if unaligned originally
        if len(raw_timestamps) > 0 and len(punc_text) > 0:
            return raw_timestamps[:len(punc_text)] + [raw_timestamps[-1]] * max(0, len(punc_text) - len(raw_timestamps))
        return []

    matcher = difflib.SequenceMatcher(None, raw_text, punc_text)
    aligned_ts = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or tag == "replace":
            # Map exactly 1-to-1 or replace (e.g., lower to upper case)
            # If lengths differ in 'replace', we stretch/shrink the timestamps
            for k in range(j2 - j1):
                raw_idx = i1 + int(k * (i2 - i1) / (j2 - j1)) if i2 > i1 else i1
                if raw_idx >= len(raw_timestamps):
                    raw_idx = len(raw_timestamps) - 1
                aligned_ts.append(raw_timestamps[raw_idx])
        elif tag == "insert":
            # Punctuation added (e.g. "," or "." or " ")
            # Copy timestamp from the character just before the insertion, or after if at start
            ref_ts = raw_timestamps[i1 - 1] if i1 > 0 else raw_timestamps[i1] if i1 < len(raw_timestamps) else [0, 0]
            for _ in range(j2 - j1):
                aligned_ts.append(ref_ts)
        elif tag == "delete":
            # Text was deleted, do nothing (timestamps are dropped)
            pass

    return aligned_ts


def split_punctuated_sentence_info(
    sentence_info: list[SentenceInfo],
    min_segment_duration_s: float = 3.0,
) -> list[SentenceInfo]:
    """Split punctuated VAD-level sentences into smaller sentence items.

    The ASR wrappers currently do not provide reliable word timestamps, so
    each child sentence receives an estimated duration proportional to its
    character length within the original segment.
    """
    split_sentences: list[SentenceInfo] = []
    for sentence in sentence_info:
        duration = sentence.end - sentence.start
        parts = _split_text_by_sentence_end(sentence.text)
        if duration < min_segment_duration_s or len(parts) <= 1:
            split_sentences.append(sentence)
            continue

        total_weight = sum(_text_timing_weight(part) for part in parts)
        if total_weight <= 0:
            split_sentences.append(sentence)
            continue

        current_start = sentence.start
        for index, part in enumerate(parts):
            if index == len(parts) - 1:
                current_end = sentence.end
            else:
                part_weight = _text_timing_weight(part)
                part_duration = duration * (part_weight / total_weight)
                current_end = current_start + part_duration

            split_sentences.append(SentenceInfo(
                text=part,
                start=round(current_start, 2),
                end=round(current_end, 2),
                speaker=sentence.speaker,
                char_timestamps=[],
            ))
            current_start = current_end

    return split_sentences


def _split_text_by_sentence_end(text: str) -> list[str]:
    """Split text into non-empty sentence-like chunks and keep punctuation."""
    parts = [match.group(0).strip() for match in _SENTENCE_END_RE.finditer(text)]
    return [part for part in parts if part]


def _text_timing_weight(text: str) -> int:
    """Return a simple timing weight based on visible non-space characters."""
    return len(re.sub(r"\s+", "", text))
