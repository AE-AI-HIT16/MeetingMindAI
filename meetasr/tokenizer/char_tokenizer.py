"""Char-level tokenizer for Chinese/Vietnamese ASR."""

from __future__ import annotations

import os

from meetasr.register import tables


@tables.register("tokenizer_classes", key="CharTokenizer")
class CharTokenizer:
    """Character-level tokenizer.

    Reads token list from a text file (one token per line).
    Used by Paraformer and SenseVoice models.

    Args:
        token_list: Path to token list file or list of token strings.
    """

    def __init__(self, token_list: str | list[str] = "", **kwargs):
        if isinstance(token_list, (list, tuple)):
            self.tokens = list(token_list)
        elif isinstance(token_list, str) and os.path.exists(token_list):
            with open(token_list, encoding="utf-8") as f:
                self.tokens = [line.strip() for line in f if line.strip()]
        else:
            self.tokens = []

        self._token2id = {t: i for i, t in enumerate(self.tokens)}

    @property
    def token_list(self) -> list[str]:
        return self.tokens

    def get_vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return [self._token2id.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return "".join(
            self.tokens[i] for i in ids
            if 0 <= i < len(self.tokens)
        )


@tables.register("tokenizer_classes", key="SentencePieceTokenizer")
class SentencePieceTokenizer:
    """SentencePiece subword tokenizer.

    Args:
        model_path: Path to .model file.
    """

    def __init__(self, model_path: str = "", **kwargs):
        self._sp = None
        if model_path and os.path.exists(model_path):
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(model_path)

    def get_vocab_size(self) -> int:
        return self._sp.GetPieceSize() if self._sp else 0

    def encode(self, text: str) -> list[int]:
        return self._sp.EncodeAsIds(text) if self._sp else []

    def decode(self, ids: list[int]) -> str:
        return self._sp.DecodeIds(ids) if self._sp else ""
