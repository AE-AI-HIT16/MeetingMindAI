"""CAM++ Speaker Diarization model wrapper."""

from __future__ import annotations

import logging
import numpy as np

from meetasr.register import tables
from meetasr.models.abs_models import AbsSpk


@tables.register("model_classes", key="cam++")
class CAMPlusPlus(AbsSpk):
    """CAM++ speaker embedding + clustering for speaker diarization.

    Lightweight (7.2M params) speaker embedding model.
    Compatible with:
      ms: iic/speech_campplus_sv_zh-cn_16k-common
      hf: funasr/campplus
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        cluster_type: str = "spectral",
        cluster_line: int = 40,
        mer_cos: float = 0.8,
        min_cluster_size: int = 4,
        **kwargs,
    ):
        """Initialize CAM++.

        Args:
            model_path: Local path to downloaded model directory.
            device: Torch device string.
            **kwargs: Additional model config.
        """
        self.model_path = model_path
        self.device = device
        self._inner = None
        self._kwargs = kwargs
        self._cluster_config = {
            "cluster_type": cluster_type,
            "cluster_line": cluster_line,
            "mer_cos": mer_cos,
            "min_cluster_size": min_cluster_size,
        }

    def _ensure_loaded(self) -> None:
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
            logging.info(f"CAM++ loaded from {self.model_path} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CAM++: {e}") from e

    def embed(self, audio: np.ndarray, **kwargs) -> "torch.Tensor":
        """Extract speaker embedding for an audio chunk.

        Args:
            audio: Float32 mono audio at 16kHz.
            **kwargs: Additional inference parameters.

        Returns:
            Speaker embedding tensor of shape [1, 192]. Always torch.Tensor.
        """
        import torch
        self._ensure_loaded()
        results = self._inner.generate(input=audio, **kwargs)
        if results and "spk_embedding" in results[0]:
            emb = results[0]["spk_embedding"]
            # FunASR may return np.ndarray depending on version — enforce contract.
            if not isinstance(emb, torch.Tensor):
                emb = torch.from_numpy(np.array(emb, dtype=np.float32))
            return emb
        return torch.zeros(1, 192)

    def cluster(
        self,
        embeddings: "torch.Tensor",
        oracle_num: int | None = None,
    ) -> list[int]:
        """Cluster embeddings into speaker labels.

        Uses CommonClustering (Spectral + AHC fallback) ported from 3D-Speaker.

        Args:
            embeddings: Stacked embeddings tensor [N, D].
            oracle_num: Known number of speakers. If None, auto-detect.

        Returns:
            List of speaker labels (int) of length N.
        """
        try:
            from meetasr.models.spk.cluster import CommonClustering

            X = embeddings.numpy() if hasattr(embeddings, "numpy") else np.array(embeddings)
            cc = CommonClustering(**self._cluster_config)
            kwargs = {}
            if oracle_num is not None:
                kwargs["speaker_num"] = oracle_num
            labels = cc(X, **kwargs)
            return labels.tolist()
        except Exception as e:
            logging.warning(f"Speaker clustering failed: {e}. Assigning all to Speaker 0.")
            n = embeddings.shape[0] if hasattr(embeddings, "shape") else 1
            return [0] * n
