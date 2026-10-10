"""Abstract base classes for all model types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np
from meetasr.runpod.schemas import Segment

if TYPE_CHECKING:
    from meetasr.runpod.utils.segmentation import LocalSegmentation


class AbsVAD(ABC):
    """Abstract Voice Activity Detector."""

    @abstractmethod
    def detect(self, audio: np.ndarray, **kwargs) -> list[Segment]:
        """Detect speech segments.

        Args:
            audio: Float32 mono audio at 16kHz.
            **kwargs: Model-specific parameters.

        Returns:
            List of Segment(start_ms, end_ms).
        """
        ...


class AbsASR(ABC):
    """Abstract Automatic Speech Recognizer."""

    @abstractmethod
    def recognize(
        self,
        audio: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> list[dict]:
        """Recognize speech in audio.

        Args:
            audio: Single audio array or batch list. All at 16kHz float32.
            **kwargs: language, hotword, etc.

        Returns:
            List of dicts with keys: "text", "timestamp" (char-level ms).
        """
        ...


class AbsPunc(ABC):
    """Abstract Punctuation Restorer."""

    @abstractmethod
    def restore(self, text: str, **kwargs) -> str:
        """Add punctuation to raw ASR text.

        Args:
            text: Raw text without punctuation.
            **kwargs: Model-specific parameters.

        Returns:
            Text with punctuation restored.
        """
        ...


class AbsSpk(ABC):
    """Abstract Speaker Diarization model."""

    @abstractmethod
    def embed(self, audio: np.ndarray, **kwargs) -> "torch.Tensor":
        """Extract speaker embedding for an audio chunk.

        Args:
            audio: Float32 mono audio at 16kHz.
            **kwargs: Model-specific parameters.

        Returns:
            Speaker embedding tensor of shape [1, D].
        """
        ...

    @abstractmethod
    def cluster(self, embeddings: "torch.Tensor", **kwargs) -> list[int]:
        """Cluster embeddings into speaker labels.

        Args:
            embeddings: Stacked embeddings tensor [N, D].
            **kwargs: e.g. oracle_num for known speaker count.

        Returns:
            List of integer speaker labels of length N.
        """
        ...


class AbsSegmenter(ABC):
    """Abstract frame-level speaker segmentation (who-speaks-when, overlaps)."""

    @abstractmethod
    def __call__(self, audio: np.ndarray) -> "LocalSegmentation":
        """Predict local speaker activity over the whole audio.

        Args:
            audio: Float32 mono audio at 16kHz.

        Returns:
            LocalSegmentation with per-window activity and per-frame counts.
        """
        ...

    @abstractmethod
    def refine(
        self,
        audio: np.ndarray,
        cluster_segments: list[list],
        segmentation: "LocalSegmentation | None" = None,
    ) -> tuple[list[list], float]:
        """Turn clustering segments into overlap-aware speaker segments.

        Args:
            audio: Float32 mono audio at 16kHz.
            cluster_segments: ``[start_s, end_s, speaker]`` from clustering.
            segmentation: Precomputed output of ``__call__`` (optional).

        Returns:
            ``([start_s, end_s, speaker], ...)`` (may overlap) and the number
            of overlapped seconds.
        """
        ...


class AbsSeparator(ABC):
    """Abstract speech separation for overlapped speech."""

    @abstractmethod
    def separate(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split mixed speech into one stream per voice.

        Args:
            audio: Float32 mono audio at 16kHz.

        Returns:
            Streams with the input's length and loudness.
        """
        ...

    @abstractmethod
    def extract(
        self,
        chunk: np.ndarray,
        regions_ms: list[tuple[int, int]],
        target_embedding: np.ndarray,
        embed_batch: Callable[[list[np.ndarray]], object],
    ) -> np.ndarray:
        """Replace overlapped regions of ``chunk`` with the target voice.

        Args:
            chunk: Float32 mono audio at 16kHz.
            regions_ms: Overlapped ``(start_ms, end_ms)`` ranges in ``chunk``.
            target_embedding: Voice embedding of the speaker to keep.
            embed_batch: Speaker embedder used to pick the target stream.

        Returns:
            Audio of the same length with the target voice in those regions.
        """
        ...
