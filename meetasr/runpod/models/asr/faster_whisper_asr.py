"""Faster-Whisper ASR model wrapper — multilingual with word-level timestamps.

Supports Vietnamese (and 90+ languages) via CTranslate2 backend.
Provides native word-level timestamps for word-level speaker diarization.

Config example:
    asr:
      model: faster-whisper
      model_size: large-v3
      device: cuda
"""

from __future__ import annotations

import logging
import os
import site
from pathlib import Path, PurePath, PureWindowsPath

import numpy as np

from meetasr.runpod.register import tables
from meetasr.runpod.models.abs_models import AbsASR


_DECODE_OPTIONS = (
    "beam_size",
    "temperature",
    "condition_on_previous_text",
    "compression_ratio_threshold",
    "log_prob_threshold",
    "no_speech_threshold",
)
_LONG_FORM_OPTIONS = (
    "vad_filter",
    "vad_parameters",
    "hallucination_silence_threshold",
)
_CUDA_DLL_DIRECTORIES = []
_CUDA_DLL_DIRECTORIES_CONFIGURED = False

def _configure_windows_cuda_runtime() -> None:
    """Expose pip-installed NVIDIA DLLs before CTranslate2 is imported."""
    global _CUDA_DLL_DIRECTORIES_CONFIGURED

    if os.name != "nt" or _CUDA_DLL_DIRECTORIES_CONFIGURED:
        return

    _CUDA_DLL_DIRECTORIES_CONFIGURED = True
    dll_paths = []

    for site_package in site.getsitepackages():
        if os.name == "nt":
            root = str(PureWindowsPath(site_package) / "nvidia")
        else:
            root = str(Path(site_package) / "nvidia")

        for package in ("cublas", "cudnn", "cuda_nvrtc"):
            if os.name == "nt":
                dll_directory = str(
                    PureWindowsPath(site_package)
                    / "nvidia"
                    / package
                    / "bin"
                )
            else:
                dll_directory = str(
                    Path(site_package)
                    / "nvidia"
                    / package
                    / "bin"
                )

            if os.path.isdir(dll_directory):
                dll_paths.append(dll_directory)

                if hasattr(os, "add_dll_directory"):
                    _CUDA_DLL_DIRECTORIES.append(
                        os.add_dll_directory(dll_directory)
                    )

    if dll_paths:
        os.environ["PATH"] = os.pathsep.join(
            [*dll_paths, os.environ.get("PATH", "")]
        )


@tables.register("model_classes", key="faster-whisper")
class FasterWhisperASR(AbsASR):
    """Faster-Whisper ASR — multilingual with word-level timestamps.

    Compatible with OpenAI Whisper model sizes (tiny → large-v3).
    Uses CTranslate2 for fast inference with native word timestamps.
    """

    uses_internal_vad = True
    has_native_punctuation = True

    def __init__(
        self,
        model_path: str = "",
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "auto",
        num_workers: int = 1,
        **kwargs,
    ):
        """Initialize FasterWhisperASR.

        Args:
            model_path: Local path or HuggingFace model ID. If empty, uses model_size.
            model_size: Whisper model size (tiny/base/small/medium/large-v3).
            device: Torch device string (e.g. "cpu", "cuda:0").
            compute_type: CTranslate2 compute type ("auto", "float16", "int8").
            num_workers: Number of workers for batch processing.
            **kwargs: Additional config.
        """
        self.model_path = model_path
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.num_workers = num_workers
        self._model = None
        self._kwargs = kwargs

    def _ensure_loaded(self) -> None:
        """Lazy-load the faster-whisper model."""
        if self._model is not None:
            return
        try:
            _configure_windows_cuda_runtime()
            from faster_whisper import WhisperModel

            # Ignore model_path if it's just the registry key from AutoModel
            if not self.model_path or self.model_path == "faster-whisper":
                model_id = self.model_size
            else:
                model_id = self.model_path

            device = "cuda" if self.device.startswith("cuda") else "cpu"
            compute = self.compute_type
            if compute == "auto":
                compute = "float16" if device == "cuda" else "int8"

            self._model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute,
                num_workers=self.num_workers,
            )
            logging.info(
                f"FasterWhisper loaded: {model_id} on {device} ({compute})"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FasterWhisper: {e}") from e

    def recognize(
        self,
        audio: np.ndarray | list[np.ndarray],
        language: str = "vi",
        **kwargs,
    ) -> list[dict]:
        """Recognize speech with char-level timestamps.

        Expands word-level timestamps from faster-whisper to per-character
        so that len(text) == len(timestamp), which is required by the
        downstream split_at_speaker_turns function.

        Args:
            audio: Single float32 mono array or list of arrays at 16kHz.
            language: Language code ("vi", "en", "auto", etc.).
            **kwargs: Additional parameters (beam_size, etc.).

        Returns:
            List of dicts with "text" and "timestamp" [[start_ms, end_ms], ...].
        """
        self._ensure_loaded()
        if isinstance(audio, np.ndarray):
            audio = [audio]

        decode_options = self._decode_options(kwargs)

        results = []
        for idx, chunk in enumerate(audio):
            segments_gen, _ = self._model.transcribe(
                chunk,
                language=language if language != "auto" else None,
                word_timestamps=True,
                vad_filter=False,  # MeetASR has its own VAD
                **decode_options,
            )

            results.append(self._result_from_segments(
                segments_gen,
                key=kwargs.get("key", f"chunk_{idx}"),
            ))
        return results

    def recognize_long_form(
        self,
        audio: np.ndarray,
        language: str = "vi",
        **kwargs,
    ) -> list[dict]:
        """Transcribe one complete recording with Faster-Whisper's VAD.

        Faster-Whisper restores timestamps to the original audio timeline after
        its internal VAD removes silence. This avoids feeding arbitrary fixed
        chunks to Whisper, which can otherwise drop words at chunk boundaries.
        """
        self._ensure_loaded()
        if not isinstance(audio, np.ndarray):
            raise TypeError("recognize_long_form expects one numpy audio array")

        long_form_options = {
            name: self._kwargs[name]
            for name in _LONG_FORM_OPTIONS
            if name in self._kwargs
        }
        long_form_options.update({
            name: kwargs[name]
            for name in _LONG_FORM_OPTIONS
            if name in kwargs
        })
        long_form_options.setdefault("vad_filter", True)
        long_form_options.setdefault(
            "vad_parameters", {"min_silence_duration_ms": 500}
        )
        long_form_options.setdefault("hallucination_silence_threshold", 2.0)

        segments_gen, _ = self._model.transcribe(
            audio,
            language=language if language != "auto" else None,
            word_timestamps=True,
            **long_form_options,
            **self._decode_options(kwargs),
        )
        key = kwargs.get("key", "long_form")
        results = []
        for index, segment in enumerate(segments_gen):
            result = self._result_from_segments([segment], key=f"{key}_{index}")
            if result["text"]:
                results.append(result)
        return results

    def _decode_options(self, runtime_kwargs: dict) -> dict:
        """Merge configured decode controls with per-call overrides."""
        options = {
            name: self._kwargs[name]
            for name in _DECODE_OPTIONS
            if name in self._kwargs
        }
        options.update({
            name: runtime_kwargs[name]
            for name in _DECODE_OPTIONS
            if name in runtime_kwargs
        })
        return options

    @staticmethod
    def _result_from_segments(segments_gen, key: str) -> dict:
        """Expand word timestamps and preserve native per-segment quality data."""
        text_chars = []
        char_timestamps = []
        segment_quality = []

        for segment in segments_gen:
            if not segment.words:
                continue
            quality = FasterWhisperASR._quality_from_segment(segment)
            if quality:
                segment_quality.append(quality)
            for word in segment.words:
                timestamp = [int(word.start * 1000), int(word.end * 1000)]
                for char in word.word:
                    text_chars.append(char)
                    char_timestamps.append(timestamp)

        raw_text = "".join(text_chars)
        text = raw_text.strip()
        start_trim = len(raw_text) - len(raw_text.lstrip())
        end_trim = len(raw_text) - len(raw_text.rstrip())
        if end_trim > 0:
            char_timestamps = char_timestamps[start_trim:-end_trim]
        elif start_trim > 0:
            char_timestamps = char_timestamps[start_trim:]

        result = {"key": key, "text": text, "timestamp": char_timestamps}
        if segment_quality:
            result["segment_quality"] = segment_quality
        return result

    @staticmethod
    def _quality_from_segment(segment) -> dict:
        """Return JSON-safe quality fields exposed by Faster-Whisper, if present."""
        quality = {}
        for name in (
            "start",
            "end",
            "avg_logprob",
            "no_speech_prob",
            "compression_ratio",
        ):
            value = getattr(segment, name, None)
            if value is None:
                continue
            try:
                quality[name] = float(value)
            except (TypeError, ValueError):
                continue
        return quality
