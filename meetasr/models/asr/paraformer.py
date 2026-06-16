"""Paraformer ASR model wrapper."""

from __future__ import annotations

import logging
import numpy as np

from meetasr.register import tables
from meetasr.models.abs_models import AbsASR


@tables.register("model_classes", key="paraformer-zh")
class Paraformer(AbsASR):
    """Paraformer ASR model — 170x realtime, Chinese/English.

    Compatible with model weights from:
      ms: damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
      hf: funasr/paraformer-zh
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        batch_size: int = 1,
        **kwargs,
    ):
        """Initialize Paraformer.

        Args:
            model_path: Local path to downloaded model directory.
            device: Torch device string.
            batch_size: Inference batch size.
            **kwargs: Additional model config.
        """
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._kwargs = kwargs

    def _ensure_loaded(self):
        """Lazy-load the Paraformer model via FunASR internals."""
        if self._model is not None:
            return
        try:
            from funasr import AutoModel as FunASRAutoModel
            self._inner = FunASRAutoModel(
                model=self.model_path,
                device=self.device,
                disable_update=True,
                disable_log=True,
            )
            logging.info(f"Paraformer loaded from {self.model_path} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Paraformer: {e}") from e

    def recognize(
        self,
        audio: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> list[dict]:
        """Recognize speech using Paraformer.

        Args:
            audio: Single float32 mono array or list at 16kHz.
            **kwargs: hotword, batch_size, etc.

        Returns:
            List of dicts with "text" and "timestamp".
        """
        self._ensure_loaded()
        if isinstance(audio, np.ndarray):
            audio = [audio]

        results = self._inner.generate(
            input=audio,
            batch_size=kwargs.get("batch_size", self.batch_size),
            **kwargs,
        )
        return results
