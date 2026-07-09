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

import numpy as np

from meetasr.register import tables
from meetasr.models.abs_models import AbsASR


@tables.register("model_classes", key="faster-whisper")
class FasterWhisperASR(AbsASR):
    """Faster-Whisper ASR — multilingual with word-level timestamps.

    Compatible with OpenAI Whisper model sizes (tiny → large-v3).
    Uses CTranslate2 for fast inference with native word timestamps.
    """

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

        results = []
        for idx, chunk in enumerate(audio):
            segments_gen, info = self._model.transcribe(
                chunk,
                language=language if language != "auto" else None,
                beam_size=kwargs.get("beam_size", 5),
                word_timestamps=True,
                vad_filter=False,  # MeetASR has its own VAD
            )

            # Expand word timestamps → per-character timestamps
            # so that len(text) == len(char_timestamps)
            text_chars = []
            char_timestamps = []

            for segment in segments_gen:
                if segment.words:
                    for word in segment.words:
                        ts = [int(word.start * 1000), int(word.end * 1000)]
                        for c in word.word:
                            text_chars.append(c)
                            char_timestamps.append(ts)

            # Strip whitespace and trim timestamps to match
            raw_text = "".join(text_chars)
            text = raw_text.strip()
            start_trim = len(raw_text) - len(raw_text.lstrip())
            end_trim = len(raw_text) - len(raw_text.rstrip())
            if end_trim > 0:
                char_timestamps = char_timestamps[start_trim:-end_trim]
            elif start_trim > 0:
                char_timestamps = char_timestamps[start_trim:]

            results.append({
                "key": kwargs.get("key", f"chunk_{idx}"),
                "text": text,
                "timestamp": char_timestamps,
            })
        return results
