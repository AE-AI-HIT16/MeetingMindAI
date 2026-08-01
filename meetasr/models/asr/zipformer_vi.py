"""Vietnamese Zipformer ASR wrapper using sherpa-onnx."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from meetasr.models.abs_models import AbsASR
from meetasr.register import tables

DEFAULT_MODEL = "hynt/Zipformer-30M-RNNT-6000h"
_MODEL_VARIANTS = {"fp32", "int8"}


@tables.register("model_classes", key="zipformer-vi")
@tables.register("model_classes", key=DEFAULT_MODEL)
class ZipformerViASR(AbsASR):
    """Vietnamese Zipformer RNNT ASR.

    Compatible with hynt/Zipformer-30M-RNNT-6000h when exported for
    sherpa-onnx offline transducer inference.
    """

    uses_internal_vad = False
    has_native_punctuation = False

    def __init__(
        self,
        model_path: str = "",
        model_size: str = DEFAULT_MODEL,
        device: str = "cpu",
        sample_rate: int = 16000,
        feature_dim: int = 80,
        num_threads: int = 4,
        decoding_method: str = "greedy_search",
        model_variant: str = "fp32",
        encoder: str = "",
        decoder: str = "",
        joiner: str = "",
        tokens: str = "",
        bpe_vocab: str = "",
        **kwargs,
    ):
        if model_variant not in _MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported Zipformer model_variant '{model_variant}'. "
                f"Expected one of: {', '.join(sorted(_MODEL_VARIANTS))}"
            )
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")

        self.model_path = model_path
        self.model_size = model_size
        self.model_name = model_path if Path(model_path).is_dir() else model_size
        self.device = device
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self.model_variant = model_variant
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        self.tokens = tokens
        self.bpe_vocab = bpe_vocab
        self._recognizer = None
        self._kwargs = kwargs

    def _ensure_loaded(self) -> None:
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

        encoder = _resolve_model_file(
            model_dir,
            self.encoder,
            role="encoder",
            variant=self.model_variant,
        )
        decoder = _resolve_model_file(
            model_dir,
            self.decoder,
            role="decoder",
            variant=self.model_variant,
        )
        joiner = _resolve_model_file(
            model_dir,
            self.joiner,
            role="joiner",
            variant=self.model_variant,
        )
        tokens = _resolve_auxiliary_file(
            model_dir,
            self.tokens,
            patterns=("tokens.txt", "config.json", "*tokens*.txt"),
            required=True,
        )
        bpe_vocab = _resolve_auxiliary_file(
            model_dir,
            self.bpe_vocab,
            patterns=("bpe.model", "*.model"),
            required=False,
        )

        logging.info(
            "Zipformer files: variant=%s encoder=%s decoder=%s joiner=%s",
            self.model_variant,
            encoder.name,
            decoder.name,
            joiner.name,
        )

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
        language: str = "vi",
        **kwargs,
    ) -> list[dict]:
        """Recognize Vietnamese speech from 16 kHz mono float32 audio.

        Args:
            audio: Single 1D numpy array or list of arrays containing 16kHz audio samples.
            **kwargs: Additional parameters (e.g., key).

        Returns:
            List of dictionaries containing text and character-level timestamps.
        """
        _validate_language(language)
        self._ensure_loaded()
        chunks = [audio] if isinstance(audio, np.ndarray) else audio

        results = []
        for index, chunk in enumerate(chunks):
            samples = _validate_audio_chunk(chunk)
            key = kwargs.get("key", f"chunk_{index}")
            if samples.size == 0:
                results.append({"key": key, "text": "", "timestamp": []})
                continue

            stream = self._recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, samples)
            self._recognizer.decode_stream(stream)
            result = stream.result

            # Try to extract per-character timestamps from token data
            text_from_tokens, timestamps = _extract_char_timestamps(result)
            if text_from_tokens is not None:
                text = text_from_tokens
            else:
                text = getattr(result, "text", str(result)).strip()
                timestamps = []

            results.append(
                {
                    "key": key,
                    "text": text,
                    "timestamp": timestamps,
                }
            )
        return results


def _extract_char_timestamps(result) -> tuple[str | None, list[list[int]]]:
    """Convert sherpa-onnx token timestamps to per-character timestamps.

    sherpa-onnx transducer models provide:
        result.timestamps — list[float] of start times (seconds) per token
        result.tokens     — list[str] of BPE tokens

    The pipeline expects len(text) == len(timestamps) (one entry per character).
    This function expands each token's timestamp to cover every character in
    that token, and reconstructs text from tokens to guarantee alignment.

    Returns:
        (text, char_timestamps) if tokens available, else (None, []).
    """
    tokens = getattr(result, "tokens", None)
    raw_timestamps = getattr(result, "timestamps", None)

    if not tokens or not raw_timestamps:
        return None, []
    if len(tokens) != len(raw_timestamps):
        return None, []

    text_chars = []
    char_timestamps = []
    for i, (token, ts) in enumerate(zip(tokens, raw_timestamps)):
        start_ms = int(ts * 1000)
        # Estimate end_ms from next token's start, or +80ms for last
        if i + 1 < len(raw_timestamps):
            end_ms = int(raw_timestamps[i + 1] * 1000)
        else:
            end_ms = start_ms + 80

        # Convert BPE token to text characters: ▁ = space
        token_text = token.replace("▁", " ")

        for c in token_text:
            text_chars.append(c)
            char_timestamps.append([start_ms, end_ms])

    # Strip whitespace and trim timestamps to match
    raw_text = "".join(text_chars)
    text = raw_text.strip()
    start_trim = len(raw_text) - len(raw_text.lstrip())
    end_trim = len(raw_text) - len(raw_text.rstrip())
    if end_trim > 0:
        char_timestamps = char_timestamps[start_trim:-end_trim]
    elif start_trim > 0:
        char_timestamps = char_timestamps[start_trim:]

    return text, char_timestamps


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


def _resolve_model_file(
    model_dir: Path,
    configured_path: str,
    *,
    role: str,
    variant: str,
) -> Path:
    """Resolve one ONNX graph without mixing FP32 and INT8 components."""
    if configured_path:
        return _resolve_configured_file(model_dir, configured_path)

    matches = sorted(model_dir.rglob(f"*{role}*.onnx"))
    if variant == "int8":
        matches = [path for path in matches if ".int8." in path.name]
    else:
        matches = [path for path in matches if ".int8." not in path.name]
    if not matches:
        raise FileNotFoundError(
            f"Missing {variant} {role} ONNX file in {model_dir}"
        )
    return matches[0]


def _resolve_auxiliary_file(
    model_dir: Path,
    configured_path: str,
    *,
    patterns: tuple[str, ...],
    required: bool,
) -> Path | None:
    if configured_path:
        return _resolve_configured_file(model_dir, configured_path)
    if required:
        return _find_required_file(model_dir, patterns)
    return _find_optional_file(model_dir, patterns)


def _resolve_configured_file(model_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if not path.is_absolute():
        path = model_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"Configured model file not found: {path}")
    return path


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
    if not np.isfinite(chunk).all():
        raise ValueError("ZipformerViASR audio contains non-finite samples")
    return np.ascontiguousarray(chunk, dtype=np.float32)


def _validate_language(language: str) -> None:
    normalized = (language or "auto").strip().lower().replace("_", "-")
    if normalized not in {"auto", "vi", "vi-vn", "vietnamese"}:
        raise ValueError(
            f"ZipformerViASR only supports Vietnamese, got language '{language}'"
        )
