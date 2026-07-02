"""ViBERT capitalization and punctuation wrapper."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from meetasr.models.abs_models import AbsPunc
from meetasr.register import tables


@tables.register("model_classes", key="vibert-capu")
@tables.register("model_classes", key="dragonSwing/vibert-capu")
class ViBERTCaPuPunc(AbsPunc):
    """Vietnamese capitalization and punctuation restoration model."""

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        **kwargs,
    ):
        self.model_path = model_path
        self.device = device
        self._inner = None
        self._kwargs = kwargs

    def _ensure_loaded(self):
        """Lazy-load the downloaded ViBERT-CaPu model."""
        if self._inner is not None:
            return
        if not self.model_path:
            raise RuntimeError("ViBERT-CaPu requires a local model_path")

        model_dir = Path(self.model_path)
        vocab_dir = model_dir / "vocabulary"
        if not model_dir.exists() or not vocab_dir.exists():
            raise RuntimeError(
                f"ViBERT-CaPu model files not found under {self.model_path}"
            )

        try:
            self._patch_transformers_resize()
            sys.path.insert(0, str(model_dir))
            import modeling_seq2labels
            from gec_model import GecBERTModel

            modeling_seq2labels.Seq2LabelsOutput = dataclass(
                modeling_seq2labels.Seq2LabelsOutput
            )
            self._inner = GecBERTModel(
                vocab_path=str(vocab_dir),
                model_paths=str(model_dir),
                device=self.device,
                split_chunk=True,
                **self._kwargs,
            )
            logging.info("ViBERT-CaPu loaded from %s on %s", self.model_path, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load ViBERT-CaPu: {e}") from e

    @staticmethod
    def _patch_transformers_resize():
        """Keep old custom model code compatible with newer transformers."""
        import transformers

        if getattr(transformers.PreTrainedModel, "_meetasr_resize_patch", False):
            return

        original = transformers.PreTrainedModel.resize_token_embeddings

        def resize_token_embeddings(
            self,
            new_num_tokens=None,
            pad_to_multiple_of=None,
            mean_resizing=False,
        ):
            return original(self, new_num_tokens, pad_to_multiple_of, False)

        transformers.PreTrainedModel.resize_token_embeddings = resize_token_embeddings
        transformers.PreTrainedModel._meetasr_resize_patch = True

    def restore(self, text: str, **kwargs) -> str:
        """Restore Vietnamese capitalization and punctuation."""
        if not text.strip():
            return text

        self._ensure_loaded()
        result = self._inner(text.lower(), **kwargs)
        if isinstance(result, list) and result and isinstance(result[0], str):
            return result[0].strip()
        if isinstance(result, str):
            return result.strip()
        return text
