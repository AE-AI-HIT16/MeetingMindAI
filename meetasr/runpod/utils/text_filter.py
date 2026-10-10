"""Drop ASR text that is not speech.

Qwen3-ASR names sounds instead of staying silent ("Applause", "haha",
"Đồng hồ.") and can loop ("bố bố bố bố ..."). The diarization gate keeps most
non-speech away from ASR; this catches what still slips through.
"""

from __future__ import annotations

import re
import unicodedata

# Sound tags and laughter Qwen emits for non-speech audio (normalized form).
_SOUND_TAGS = {
    "applause", "applaud", "laughter", "laughing", "laugh", "music", "noise",
    "silence", "cough", "coughing", "sigh", "breathing", "clapping",
    "vỗ tay", "tiếng vỗ tay", "tiếng cười", "nhạc", "tiếng nhạc", "tiếng ồn",
}
_LAUGHTER = re.compile(r"^(h[aeiô]){2,}h?$")
# Vietnamese is ~4-6 syllables per second; 9+ over a second of audio is a
# runaway decode, not speech.
MAX_WORDS_PER_SECOND = 9.0
MAX_REPEATS = 3


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _collapse_loops(words: list[str]) -> list[str]:
    """Cut n-gram loops (n <= 4) repeated more than ``MAX_REPEATS`` times."""
    for size in range(1, 5):
        output: list[str] = []
        index = 0
        while index < len(words):
            gram = words[index:index + size]
            repeats = 1
            while words[index + repeats * size:index + (repeats + 1) * size] == gram:
                repeats += 1
            if len(gram) == size and repeats > MAX_REPEATS:
                output.extend(gram)
                index += repeats * size
            else:
                output.append(words[index])
                index += 1
        words = output
    return words


def clean_transcript_text(text: str, duration_s: float) -> str:
    """Return ``text`` cleaned of non-speech output, or "" to drop it."""
    stripped = text.strip()
    normalized = _normalize(stripped)
    if not normalized:
        return ""
    if normalized in _SOUND_TAGS or _LAUGHTER.match(normalized.replace(" ", "")):
        return ""

    words = stripped.split()
    collapsed = _collapse_loops(words)
    if len(collapsed) < len(words):
        stripped = " ".join(collapsed)
    if duration_s >= 1.0 and len(collapsed) / duration_s > MAX_WORDS_PER_SECOND:
        return ""
    return stripped
