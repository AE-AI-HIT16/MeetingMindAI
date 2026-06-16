"""FSMN-VAD model wrapper."""

from __future__ import annotations

import logging
import numpy as np

from meetasr.register import tables
from meetasr.schemas import Segment
from meetasr.models.abs_models import AbsVAD


@tables.register("model_classes", key="fsmn-vad")
class FsmnVAD(AbsVAD):
    """FSMN Voice Activity Detection.

    Wraps FunASR's FSMN-VAD model for speech segment detection.
    Compatible with model weights from:
      ms: damo/speech_fsmn_vad_zh-cn-16k-common-pytorch
      hf: funasr/fsmn-vad
    """

    def __init__(
        self,
        model_path: str = "",
        max_single_segment_time: int = 60000,
        **kwargs,
    ):
        """Initialize FSMN-VAD.

        Args:
            model_path: Local path to downloaded model directory.
            max_single_segment_time: Max segment duration in ms (default 60s).
            **kwargs: Additional model config (passed to underlying model).
        """
        self.model_path = model_path
        self.max_segment_ms = max_single_segment_time
        self._model = None  # lazy load
        self._kwargs = kwargs

    def _ensure_loaded(self):
        """Lazy-load the underlying funasr model."""
        if self._model is not None:
            return
        # Reuse FunASR model internals via its registered model class
        # This lets us use FunASR weights without reimplementing the model
        try:
            from funasr.models.fsmn_vad_streaming.model import FsmnVAD as _FsmnVAD
            import torch
            from omegaconf import OmegaConf
            import os

            config_path = os.path.join(self.model_path, "config.yaml")
            cfg = OmegaConf.load(config_path)
            cfg = OmegaConf.to_container(cfg, resolve=True)

            self._model = _FsmnVAD(**cfg.get("model_conf", {}))
            weight_path = os.path.join(self.model_path, "model.pt")
            state = torch.load(weight_path, map_location="cpu")
            self._model.load_state_dict(state, strict=False)
            self._model.eval()
            logging.info(f"FsmnVAD loaded from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load FsmnVAD: {e}") from e

    def detect(self, audio: np.ndarray, **kwargs) -> list[Segment]:
        """Detect speech segments in audio.

        Args:
            audio: Float32 mono audio at 16kHz.
            **kwargs: Overrides (max_single_segment_time, etc.)

        Returns:
            List of Segment(start_ms, end_ms), sorted by start_ms.
        """
        self._ensure_loaded()
        max_ms = kwargs.get("max_single_segment_time", self.max_segment_ms)

        import torch
        with torch.no_grad():
            # Model expects [1, T] tensor
            x = torch.from_numpy(audio).unsqueeze(0)
            segments_raw = self._model.inference(x, **{
                "max_single_segment_time": max_ms,
            })

        segments = []
        for seg in segments_raw:
            # seg is typically [start_ms, end_ms]
            if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                segments.append(Segment(int(seg[0]), int(seg[1])))

        return segments
