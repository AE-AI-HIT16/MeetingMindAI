"""CT-Transformer Punctuation Restoration wrapper."""

from __future__ import annotations

import logging

from meetasr.register import tables
from meetasr.models.abs_models import AbsPunc


@tables.register("model_classes", key="ct-punc")
class CTTransformerPunc(AbsPunc):
    """CT-Transformer punctuation restoration model.

    Adds punctuation to raw ASR text (Chinese/English).
    Compatible with:
      ms: damo/punc_ct-transformer_cn-en-common-vocab471067-large
      hf: funasr/ct-punc
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize CT-Transformer Punc model.

        Args:
            model_path: Local path to downloaded model directory.
            device: Torch device string.
            **kwargs: Additional model config.
        """
        self.model_path = model_path
        self.device = device
        self._inner = None
        self._kwargs = kwargs

    def _ensure_loaded(self):
        """Lazy-load via FunASR AutoModel."""
        if self._inner is not None:
            return
        try:
            from funasr import AutoModel as FunASRAutoModel
            self._inner = FunASRAutoModel(
                model=self.model_path,
                device=self.device,
                disable_update=True,
                disable_log=True,
            )
            logging.info(f"CT-Punc loaded from {self.model_path} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CT-Transformer Punc: {e}") from e

    def restore(self, text: str, **kwargs) -> str:
        """Restore punctuation in raw ASR text.

        Args:
            text: Raw text string without punctuation.
            **kwargs: Additional inference parameters.

        Returns:
            Punctuated text string.
        """
        if not text.strip():
            return text

        self._ensure_loaded()
        results = self._inner.generate(input=text, **kwargs)
        if results and isinstance(results, list):
            return results[0].get("text", text)
        return text
