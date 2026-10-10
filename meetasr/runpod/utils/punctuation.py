"""Punctuate and capitalize transcripts per speaker run.

Qwen3-ASR punctuates clean read speech, but about half of conversational
Vietnamese segments come back lowercase with no punctuation. Segments are cut
at pauses near the 15 s ASR limit, often mid-sentence, so punctuating each
segment alone puts a full stop at every cut. Consecutive segments of one
speaker are therefore punctuated as one text and split back, keeping the
number of segments (they are already stored and shown).
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Callable

from meetasr.runpod.schemas import SentenceInfo

logger = logging.getLogger(__name__)

_TERMINAL = re.compile(r"[.?!…]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
MAX_WORDS_PER_CALL = 150


def needs_punctuation(text: str) -> bool:
    """No sentence-final mark, or starts lowercase."""
    stripped = text.strip()
    if not stripped:
        return False
    return not _TERMINAL.search(stripped) or stripped[0].islower()


def _plain(token: str) -> str:
    return _PUNCT.sub("", token).lower()


def punctuate_speaker_runs(
    sentences: list[SentenceInfo],
    restore: Callable[[str], str],
    *,
    max_words: int = MAX_WORDS_PER_CALL,
) -> list[SentenceInfo]:
    """Restore punctuation for runs of same-speaker segments that need it.

    Words are never changed: if the model's output does not contain exactly
    the input words, that piece keeps its original text. Words Qwen already
    capitalized mid-segment (names, often from the user's keywords) keep
    their capitals.
    """
    result = copy.deepcopy(sentences)
    index = 0
    while index < len(result):
        end = index + 1
        while end < len(result) and result[end].speaker == result[index].speaker:
            end += 1
        run = result[index:end]
        if any(needs_punctuation(s.text) for s in run):
            _punctuate_run(run, restore, max_words)
        index = end
    return result


def _punctuate_run(
    run: list[SentenceInfo],
    restore: Callable[[str], str],
    max_words: int,
) -> None:
    # Pieces of whole segments, at most ``max_words`` words each. Overlapping
    # windows were tried: ViBERT then left long colloquial runs unpunctuated.
    piece: list[SentenceInfo] = []
    count = 0
    for sentence in run + [None]:
        words = len(sentence.text.split()) if sentence is not None else 0
        if sentence is None or (piece and count + words > max_words):
            if piece:
                _punctuate_piece(piece, restore)
            piece, count = [], 0
        if sentence is not None:
            piece.append(sentence)
            count += words


def _punctuate_piece(piece: list[SentenceInfo], restore: Callable[[str], str]) -> None:
    kept = [[token for token in s.text.split() if _plain(token)] for s in piece]
    flat = [_plain(token) for tokens in kept for token in tokens]
    if not flat:
        return
    try:
        restored = restore(" ".join(flat)).split()
    except Exception as exc:
        logger.warning("Punctuation failed for %d segment(s): %s", len(piece), exc)
        return
    if [_plain(token) for token in restored] != flat:
        logger.warning("Punctuation changed words; keeping %d segment(s) as is.", len(piece))
        return

    position = 0
    for sentence, tokens in zip(piece, kept):
        out = restored[position:position + len(tokens)]
        position += len(tokens)
        for offset, (token, new) in enumerate(zip(tokens, out)):
            # Keep Qwen's mid-segment capitals (proper names).
            if offset > 0 and any(ch.isupper() for ch in _PUNCT.sub("", token)):
                out[offset] = _recase(new, _PUNCT.sub("", token))
        if out:
            sentence.text = " ".join(out)


def _recase(token: str, cased_word: str) -> str:
    """Apply ``cased_word``'s letters to ``token``, keeping its punctuation."""
    letters = iter(cased_word)
    return "".join(next(letters, ch) if not _PUNCT.match(ch) else ch for ch in token)
