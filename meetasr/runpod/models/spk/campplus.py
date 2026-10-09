"""CAM++ Speaker Diarization model wrapper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from meetasr.runpod.models.abs_models import AbsSpk
from meetasr.runpod.register import tables

if TYPE_CHECKING:
    import torch


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
        pval: float = 0.012,
        max_num_spks: int = 15,
        min_num_spks: int = 1,
        **kwargs,
    ):
        """Initialize CAM++.

        Args:
            model_path: Local path to downloaded model directory.
            device: Torch device string.
            cluster_type: "spectral" (long audio) or "AHC".
            cluster_line: segment-count threshold to switch spectral/AHC.
            mer_cos: cosine ≥ this merges two speakers into one (lower = merge
                more aggressively → fewer speakers; higher = split more).
            min_cluster_size: clusters with ≤ this many chunks are absorbed into
                the nearest bigger one (lower = keep speakers who talk little).
            pval: spectral affinity pruning (lower = keep fewer edges → split
                into more speakers; higher = merge).
            max_num_spks / min_num_spks: bounds on auto-detected speaker count.
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
            "pval": pval,
            "max_num_spks": max_num_spks,
            "min_num_spks": min_num_spks,
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
                disable_pbar=True,
                log_level="ERROR",
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
        kwargs.setdefault("disable_pbar", True)
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
            from meetasr.runpod.models.spk.cluster import CommonClustering

            if hasattr(embeddings, "detach"):
                # Move off GPU before numpy: .numpy() raises on CUDA tensors,
                # which the broad except below would silently turn into 1 speaker.
                X = embeddings.detach().cpu().numpy()
            elif hasattr(embeddings, "numpy"):
                X = embeddings.numpy()
            else:
                X = np.array(embeddings)
            cc = CommonClustering(**self._cluster_config)
            kwargs = {}
            if oracle_num is not None:
                kwargs["speaker_num"] = oracle_num
            labels = cc(X, **kwargs)
            labels_list = labels.tolist()
            uniq = sorted(set(labels_list))
            logging.info(
                "SPK clustering: %d embeddings -> %d speaker(s) %s (config=%s)",
                X.shape[0], len(uniq), uniq, self._cluster_config,
            )
            return labels_list
        except Exception as e:
            logging.exception(
                "Speaker clustering FAILED (%s). Assigning all to Speaker 0.", e
            )
            n = embeddings.shape[0] if hasattr(embeddings, "shape") else 1
            return [0] * n
