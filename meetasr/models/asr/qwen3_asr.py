"""Qwen3-ASR model wrapper — multilingual offline transcription.

Supports 52+ languages including Vietnamese via the Qwen3-ASR family.
Uses the ``qwen-asr`` library for inference with HuggingFace Transformers backend.

Config example:
    asr:
      model: qwen3-asr
      model_size: Qwen/Qwen3-ASR-0.6B
      device: cuda:0
      dtype: bfloat16
      # Optional for supported aligner languages (Vietnamese is not supported):
      # forced_aligner: Qwen/Qwen3-ForcedAligner-0.6B
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from meetasr.models.abs_models import AbsASR
from meetasr.register import tables

SAMPLE_RATE = 16000
_REGISTRY_MODEL_PATHS = {"", "qwen3-asr"}
_DTYPES = {"bfloat16", "float16", "float32"}
_ALIGNER_LANGUAGES = {
    "Cantonese",
    "Chinese",
    "English",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
}

logger = logging.getLogger(__name__)


@tables.register("model_classes", key="qwen3-asr")
@tables.register("model_classes", key="Qwen/Qwen3-ASR-0.6B")
class Qwen3ASR(AbsASR):
    """Qwen3-ASR — multilingual speech recognition (52+ languages).

    Compatible with Qwen/Qwen3-ASR-0.6B and Qwen/Qwen3-ASR-1.7B.
    Uses ``qwen-asr`` library with Transformers backend.

    Notable attributes consumed by MeetPipeline:
        uses_internal_vad = False   — relies on external VAD
        has_native_punctuation = True — Qwen3-ASR produces punctuated text
    """

    uses_internal_vad = False
    has_native_punctuation = True

    def __init__(
        self,
        model_path: str = "",
        model_size: str = "Qwen/Qwen3-ASR-0.6B",
        device: str = "cpu",
        forced_aligner: str = "",
        dtype: str = "bfloat16",
        max_inference_batch_size: int = 1,
        max_new_tokens: int = 512,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize Qwen3ASR.

        Args:
            model_path: Downloaded local directory from AutoModel. When ``hub:
                none`` leaves this at the registry key, ``model_size`` is used.
            model_size: HuggingFace model ID (e.g. "Qwen/Qwen3-ASR-0.6B").
            device: Torch device string (e.g. "cpu", "cuda", "cuda:0").
            forced_aligner: Optional HuggingFace model ID for forced aligner
                timestamps. The official aligner does not support Vietnamese.
            dtype: Torch dtype string ("bfloat16", "float16", "float32").
            max_inference_batch_size: Internal qwen-asr inference batch limit.
            max_new_tokens: Maximum output tokens for each internal audio chunk.
            model_kwargs: Extra kwargs forwarded only to ``from_pretrained``.
            **kwargs: Additional config.
        """
        if dtype not in _DTYPES:
            raise ValueError(
                f"Unsupported Qwen3-ASR dtype '{dtype}'. Expected one of: "
                f"{', '.join(sorted(_DTYPES))}"
            )

        self.model_path = model_path
        self.model_size = model_size
        self.model_name = (
            model_path if model_path not in _REGISTRY_MODEL_PATHS else model_size
        )
        self.device = device
        self.forced_aligner = forced_aligner
        self.dtype = dtype
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self.model_kwargs = dict(model_kwargs or {})
        self._model = None
        self._kwargs = kwargs

    def _ensure_loaded(self) -> None:
        """Lazy-load the Qwen3-ASR model."""
        if self._model is not None:
            return
        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map[self.dtype]

            if self.device.startswith("cuda"):
                device_map = "cuda:0" if self.device == "cuda" else self.device
            else:
                device_map = "cpu"

            load_kwargs: dict[str, Any] = {
                "dtype": torch_dtype,
                "device_map": device_map,
                "max_inference_batch_size": self.max_inference_batch_size,
                "max_new_tokens": self.max_new_tokens,
            }
            load_kwargs.update(self.model_kwargs)

            if self.forced_aligner:
                load_kwargs["forced_aligner"] = self.forced_aligner
                load_kwargs["forced_aligner_kwargs"] = {
                    "dtype": torch_dtype,
                    "device_map": device_map,
                }

            self._model = Qwen3ASRModel.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
            logger.info(
                "Qwen3ASR loaded: %s on %s (%s)%s",
                self.model_name,
                device_map,
                self.dtype,
                f" + aligner {self.forced_aligner}" if self.forced_aligner else "",
            )
        except ImportError as e:
            raise RuntimeError(
                "qwen-asr is required for Qwen3ASR. "
                "Install it with: pip install qwen-asr"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen3ASR: {e}") from e

    def recognize(
        self,
        audio: np.ndarray | list[np.ndarray],
        language: str = "auto",
        **kwargs: Any,
    ) -> list[dict]:
        """Recognize speech and return MeetASR-compatible results.

        Args:
            audio: Single float32 mono array or list of arrays at 16kHz.
            language: Language hint ("auto", "vi", "en", "zh", etc.).
                Mapped to Qwen3-ASR language names.
            **kwargs: Additional parameters.

        Returns:
            List of dicts with "text" and "timestamp" (char-level ms pairs).
        """
        self._ensure_loaded()
        if isinstance(audio, np.ndarray):
            audio = [audio]

        qwen_language = _map_language(language)
        use_timestamps = bool(
            self.forced_aligner and qwen_language in _ALIGNER_LANGUAGES
        )
        if self.forced_aligner and not use_timestamps:
            logger.warning(
                "Qwen3 forced alignment skipped for language %r. "
                "Use one of the officially supported languages: %s",
                qwen_language or language,
                ", ".join(sorted(_ALIGNER_LANGUAGES)),
            )

        results = []
        for idx, chunk in enumerate(audio):
            chunk = _validate_audio_chunk(chunk)
            if chunk.size == 0:
                results.append(
                    {"key": kwargs.get("key", f"chunk_{idx}"), "text": "", "timestamp": []}
                )
                continue

            transcription = self._model.transcribe(
                audio=[(chunk, SAMPLE_RATE)],
                context=kwargs.get("context", ""),
                language=[qwen_language] if qwen_language else None,
                return_time_stamps=use_timestamps,
            )

            if not transcription:
                results.append(
                    {"key": kwargs.get("key", f"chunk_{idx}"), "text": "", "timestamp": []}
                )
                continue

            result_obj = transcription[0]
            text = getattr(result_obj, "text", "").strip()

            char_timestamps = []
            if use_timestamps and hasattr(result_obj, "time_stamps") and result_obj.time_stamps:
                char_timestamps = _expand_word_timestamps_to_chars(
                    result_obj.time_stamps, text
                )

            results.append(
                {
                    "key": kwargs.get("key", f"chunk_{idx}"),
                    "text": text,
                    "timestamp": char_timestamps,
                    "language": getattr(result_obj, "language", ""),
                }
            )

        return results


# ------------------------------------------------------------------
# Language mapping
# ------------------------------------------------------------------

_LANGUAGE_MAP = {
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "th": "Thai",
    "auto": None,
}


def _map_language(code: str) -> str | None:
    """Map a short ISO code to the Qwen3-ASR full language name.

    Returns None for "auto" or unrecognized codes (let the model detect).
    """
    if not code or code == "auto":
        return None
    mapped = _LANGUAGE_MAP.get(code.lower())
    if mapped is not None:
        return mapped
    # If it looks like a full language name already, pass through
    if code[0].isupper() and len(code) > 3:
        return code
    return None


# ------------------------------------------------------------------
# Audio validation and timestamp expansion
# ------------------------------------------------------------------


def _validate_audio_chunk(chunk: np.ndarray) -> np.ndarray:
    """Return a contiguous 16 kHz mono float32 waveform."""
    if not isinstance(chunk, np.ndarray):
        raise TypeError("Qwen3ASR audio chunks must be numpy arrays")
    if chunk.ndim != 1:
        raise ValueError(
            f"Qwen3ASR expects mono 1-D audio, got shape {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise ValueError("Qwen3ASR audio contains non-finite samples")
    return np.ascontiguousarray(chunk, dtype=np.float32)


def _expand_word_timestamps_to_chars(
    time_stamps: Any,
    text: str,
) -> list[list[int]]:
    """Convert Qwen3 forced-aligner word/char timestamps to per-character pairs.

    The MeetASR pipeline expects len(text) == len(timestamps), where each
    entry is [start_ms, end_ms] for one character.

    Qwen returns a ``ForcedAlignResult`` iterable. Alignment items can be words
    or characters depending on the language. Spaces and punctuation omitted by
    the aligner are filled using the interval between adjacent aligned items.
    """
    if not time_stamps or not text:
        return []

    stamps = _alignment_items(time_stamps)
    char_timestamps: list[list[int] | None] = [None] * len(text)
    cursor = 0
    previous_timestamp: list[int] | None = None

    for stamp in stamps:
        start_s = getattr(stamp, "start_time", None)
        end_s = getattr(stamp, "end_time", None)
        aligned_text = str(getattr(stamp, "text", ""))

        if start_s is None or end_s is None or not aligned_text:
            continue

        start_ms = int(float(start_s) * 1000)
        end_ms = int(float(end_s) * 1000)
        timestamp = [start_ms, max(start_ms, end_ms)]

        match_start = text.find(aligned_text, cursor)
        if match_start < 0:
            match_start = cursor
        match_end = min(len(text), match_start + len(aligned_text))

        gap_timestamp = _gap_timestamp(previous_timestamp, timestamp)
        for char_index in range(cursor, match_start):
            char_timestamps[char_index] = list(gap_timestamp)
        for char_index in range(match_start, match_end):
            char_timestamps[char_index] = list(timestamp)

        cursor = match_end
        previous_timestamp = timestamp
        if cursor >= len(text):
            break

    fallback_timestamp = previous_timestamp or [0, 0]
    for char_index, timestamp in enumerate(char_timestamps):
        if timestamp is None:
            char_timestamps[char_index] = list(fallback_timestamp)

    return [timestamp for timestamp in char_timestamps if timestamp is not None]


def _alignment_items(time_stamps: Any) -> list[Any]:
    """Normalize Qwen's ForcedAlignResult and legacy nested lists."""
    items = getattr(time_stamps, "items", None)
    if isinstance(items, list):
        return items
    if isinstance(time_stamps, (list, tuple)):
        if (
            len(time_stamps) == 1
            and not hasattr(time_stamps[0], "start_time")
            and hasattr(time_stamps[0], "__iter__")
        ):
            return list(time_stamps[0])
        return list(time_stamps)
    try:
        return list(time_stamps)
    except TypeError:
        return []


def _gap_timestamp(
    previous_timestamp: list[int] | None,
    current_timestamp: list[int],
) -> list[int]:
    """Timestamp unaligned spaces/punctuation between two aligned units."""
    if previous_timestamp is None:
        return list(current_timestamp)
    start_ms = previous_timestamp[1]
    end_ms = max(start_ms, current_timestamp[0])
    return [start_ms, end_ms]
