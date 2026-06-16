"""SenseVoice ASR model wrapper."""

from __future__ import annotations

import logging
import numpy as np

from meetasr.register import tables
from meetasr.models.abs_models import AbsASR


@tables.register("model_classes", key="sensevoice-small")
@tables.register("model_classes", key="iic/SenseVoiceSmall")
class SenseVoice(AbsASR):
    """SenseVoice ASR model — supports zh/en/ja/ko/yue + emotion detection.

    Compatible with model weights from:
      ms: iic/SenseVoiceSmall
      hf: FunAudioLLM/SenseVoiceSmall
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize SenseVoice.

        Args:
            model_path: Local path to downloaded model directory.
            device: Torch device string (e.g. "cpu", "cuda:0").
            **kwargs: Additional model config.
        """
        self.model_path = model_path
        self.device = device
        self._model = None
        self._kwargs = kwargs

    def _ensure_loaded(self):
        """Lazy-load the SenseVoice model."""
        if self._model is not None:
            return
        try:
            from funasr.models.sense_voice.model import SenseVoiceSmall
            import torch
            import os

            self._model = SenseVoiceSmall.from_pretrained(
                model=self.model_path,
                device=self.device,
            )
            self._model.eval()
            logging.info(f"SenseVoice loaded from {self.model_path} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SenseVoice: {e}") from e

    def recognize(
        self,
        audio: np.ndarray | list[np.ndarray],
        language: str = "auto",
        use_itn: bool = True,
        **kwargs,
    ) -> list[dict]:
        """Recognize speech in audio.

        Args:
            audio: Single float32 mono array or list of arrays at 16kHz.
            language: "auto", "zh", "en", "vi", "ja", "ko", "yue".
            use_itn: Apply inverse text normalization.
            **kwargs: Additional inference parameters.

        Returns:
            List of dicts with "text" and "timestamp" (char-level ms).
        """
        self._ensure_loaded()
        if isinstance(audio, np.ndarray):
            audio = [audio]

        import torch
        results = []
        with torch.no_grad():
            for chunk in audio:
                res = self._model.inference(
                    data_in=chunk,
                    language=language,
                    use_itn=use_itn,
                    **kwargs,
                )
                # res is list of dicts
                if isinstance(res, (list, tuple)) and len(res) > 0:
                    results.extend(res[0] if isinstance(res[0], list) else res)
        return results
