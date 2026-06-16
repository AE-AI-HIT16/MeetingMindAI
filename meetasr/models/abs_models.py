"""Abstract base classes for all model types."""

from abc import ABC, abstractmethod
import numpy as np
from meetasr.schemas import Segment


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
