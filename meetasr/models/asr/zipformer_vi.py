"""Vietnamese Zipformer ASR wrapper using sherpa-onnx."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from meetasr.models.abs_models import AbsASR
from meetasr.register import tables


@tables.register("model_classes", key="zipformer-vi")
@tables.register("model_classes", key="hynt/Zipformer-30M-RNNT-6000h")
class ZipformerViASR(AbsASR):
    """Vietnamese Zipformer RNNT ASR.

    Compatible with hynt/Zipformer-30M-RNNT-6000h when exported for
    sherpa-onnx offline transducer inference.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        sample_rate: int = 16000,
        feature_dim: int = 80,
        num_threads: int = 1,
        decoding_method: str = "greedy_search",
        **kwargs,
    ):
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self._recognizer = None
        self._kwargs = kwargs

    def _ensure_loaded(self):
        """Lazy-load the sherpa-onnx offline recognizer."""
        if self._recognizer is not None:
            return
        try:
            self._recognizer = self._build_recognizer()
            logging.info("ZipformerViASR loaded from %s on %s", self.model_path, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load ZipformerViASR: {e}") from e

    def _build_recognizer(self):
        try:
            import sherpa_onnx
        except ImportError as e:
            raise RuntimeError(
                "sherpa_onnx is required for ZipformerViASR. "
                "Install it with: pip install sherpa-onnx"
            ) from e

        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"model_path must be a directory: {model_dir}")

        encoder = _find_required_file(model_dir, ("encoder*.onnx", "*encoder*.onnx"))
        decoder = _find_required_file(model_dir, ("decoder*.onnx", "*decoder*.onnx"))
        joiner = _find_required_file(model_dir, ("joiner*.onnx", "*joiner*.onnx"))
        tokens = _find_required_file(model_dir, ("tokens.txt", "config.json", "*tokens*.txt"))
        bpe_vocab = _find_optional_file(model_dir, ("bpe.model", "*.model"))

        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=self.num_threads,
            sample_rate=self.sample_rate,
            feature_dim=self.feature_dim,
            decoding_method=self.decoding_method,
            modeling_unit="bpe" if bpe_vocab else "cjkchar",
            bpe_vocab=str(bpe_vocab) if bpe_vocab else "",
            provider=_provider_from_device(self.device),
        )

    def recognize(
        self,
        audio: np.ndarray | list[np.ndarray],
        **kwargs,
    ) -> list[dict]:
        """Recognize Vietnamese speech from 16 kHz mono float32 audio."""
        self._ensure_loaded()
        chunks = [audio] if isinstance(audio, np.ndarray) else audio

        results = []
        for index, chunk in enumerate(chunks):
            samples = _validate_audio_chunk(chunk)
            stream = self._recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, samples)
            self._recognizer.decode_stream(stream)
            result = stream.result
            text = getattr(result, "text", str(result)).strip()
            results.append(
                {
                    "key": kwargs.get("key", f"chunk_{index}"),
                    "text": text,
                    "timestamp": [],
                }
            )
        return results


def _find_required_file(model_dir: Path, patterns: tuple[str, ...]) -> Path:
    """Find a required model file using the first matching glob pattern."""
    for pattern in patterns:
        matches = sorted(model_dir.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Missing required model file in {model_dir}: {patterns}")


def _find_optional_file(model_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    """Find an optional model file using the first matching glob pattern."""
    for pattern in patterns:
        matches = sorted(model_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _provider_from_device(device: str) -> str:
    """Map project-style device strings to sherpa-onnx provider names."""
    normalized = device.lower()
    if normalized.startswith("cuda"):
        return "cuda"
    if normalized.startswith("coreml"):
        return "coreml"
    return "cpu"


def _validate_audio_chunk(chunk: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 mono waveform."""
    if not isinstance(chunk, np.ndarray):
        raise TypeError("audio chunks must be numpy arrays")
    if chunk.ndim != 1:
        raise ValueError("ZipformerViASR expects mono 1D audio")
    return np.ascontiguousarray(chunk, dtype=np.float32)
