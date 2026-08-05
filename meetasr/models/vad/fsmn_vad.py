"""FSMN-VAD model wrapper."""

from __future__ import annotations

import inspect
import logging

import numpy as np

from meetasr.models.abs_models import AbsVAD
from meetasr.register import tables
from meetasr.schemas import Segment

_MODEL_METADATA_OPTIONS = frozenset({
    "encoder",
    "encoder_conf",
    "frontend",
    "frontend_conf",
    "hub",
    "model",
    "model_conf",
})


@tables.register("model_classes", key="fsmn-vad")
@tables.register("model_classes", key="FsmnVADStreaming")
class FsmnVAD(AbsVAD):
    """FSMN Voice Activity Detection.

    Wraps FunASR's FSMN-VAD model for speech segment detection.
    Compatible with model weights from:
      ms: damo/speech_fsmn_vad_zh-cn-16k-common-pytorch
      hf: funasr/fsmn-vad
    """

    model_name = "fsmn-vad"

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        max_single_segment_time: int = 60000,
        min_segment_ms: int = 200,
        **kwargs,
    ):
        """Initialize FSMN-VAD.

        Args:
            model_path: Local path to downloaded model directory.
            device: Torch device string.
            max_single_segment_time: Max segment duration in ms (default 60s).
            min_segment_ms: Drop detected segments shorter than this duration.
            **kwargs: Additional model config (passed to underlying model).
        """
        self.model_path = model_path
        self.device = device
        self.max_segment_ms = max_single_segment_time
        self.min_segment_ms = min_segment_ms
        self._model = None  # lazy load
        self._frontend = None
        self._kwargs = kwargs

    def _ensure_loaded(self) -> None:
        """Lazy-load the underlying funasr model."""
        if self._model is not None:
            return
        try:
            import os

            import funasr.models.fsmn_vad_streaming.encoder  # noqa: F401
            import torch
            from funasr.frontends.wav_frontend import WavFrontendOnline
            from funasr.models.fsmn_vad_streaming.model import (
                FsmnVADStreaming as _FsmnVAD,
            )
            from funasr.models.fsmn_vad_streaming.model import (
                VADXOptions,
            )
            from omegaconf import OmegaConf

            config_path = os.path.join(self.model_path, "config.yaml")
            cfg = OmegaConf.load(config_path)
            cfg = OmegaConf.to_container(cfg, resolve=True)

            frontend_conf = dict(cfg.get("frontend_conf", {}))
            self._frontend = WavFrontendOnline(**frontend_conf)

            model_conf = dict(cfg.get("model_conf", {}))
            model_conf.update(
                {
                    "encoder": cfg.get("encoder"),
                    "encoder_conf": cfg.get("encoder_conf", {}),
                }
            )
            supported_options = _supported_vad_options(VADXOptions)
            unsupported_options = sorted(
                set(self._kwargs).difference(
                    supported_options,
                    _MODEL_METADATA_OPTIONS,
                )
            )
            if unsupported_options:
                logging.warning(
                    "Ignoring unsupported FSMN-VAD config option(s): %s",
                    ", ".join(unsupported_options),
                )
            model_conf.update({
                name: value
                for name, value in self._kwargs.items()
                if name in supported_options
            })
            model_conf["max_single_segment_time"] = self.max_segment_ms

            self._model = _FsmnVAD(**model_conf)
            weight_path = os.path.join(self.model_path, "model.pt")
            state = torch.load(weight_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self._model.load_state_dict(state, strict=False)
            self._model.to(self.device)
            self._model.eval()
            logging.info(f"FsmnVAD loaded from {self.model_path} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load FsmnVAD: {e}") from e

    def detect(self, audio: np.ndarray, **kwargs) -> list[Segment]:
        """Detect speech segments in audio.

        Args:
            audio: Float32 mono audio at 16kHz.
            **kwargs: Overrides (max_single_segment_time, min_segment_ms,
                cache, is_final). ``cache`` is mutated in place for streaming
                sessions that reuse ``session.vad_state``.

        Returns:
            List of Segment(start_ms, end_ms), sorted by start_ms.
        """
        self._validate_audio(audio)
        # funasr's WavFrontendOnline requires at least enough samples for lfr_m frames
        # Usually ~400-800 samples minimum at 16kHz. We use 1600 (100ms) to be safe.
        if audio.size < 1600:
            return []

        self._ensure_loaded()
        max_ms = kwargs.get("max_single_segment_time", self.max_segment_ms)
        min_ms = kwargs.get("min_segment_ms", self.min_segment_ms)
        cache = kwargs.get("cache")
        if cache is None:
            cache = {}
        is_final = kwargs.get("is_final", True)

        segments_raw = self._run_inference(
            audio,
            max_single_segment_time=max_ms,
            cache=cache,
            is_final=is_final,
        )
        return _normalize_segments(segments_raw, min_segment_ms=min_ms)

    def _run_inference(
        self,
        audio: np.ndarray,
        max_single_segment_time: int,
        cache: dict | None = None,
        is_final: bool = True,
    ):
        """Run the underlying model and return raw VAD segments.

        ``cache`` is passed through to FunASR and mutated in place so callers
        (realtime streaming) can persist state via ``session.vad_state``.
        """
        if cache is None:
            cache = {}

        import torch

        runtime_options = getattr(self._model, "vad_opts", None)
        if runtime_options is not None:
            runtime_options.max_single_segment_time = max_single_segment_time

        with torch.no_grad():
            results = self._model.inference(
                data_in=[audio],
                key=["audio"],
                frontend=self._frontend,
                cache=cache,
                device=self.device,
                is_final=is_final,
                max_single_segment_time=max_single_segment_time,
            )

        if isinstance(results, tuple):
            results = results[0]
        if isinstance(results, list) and results and isinstance(results[0], dict):
            return results[0].get("value", [])
        return results

    @staticmethod
    def _validate_audio(audio: np.ndarray) -> None:
        """Validate VAD audio input before model inference."""
        if not isinstance(audio, np.ndarray):
            raise ValueError(f"audio must be a numpy.ndarray, got {type(audio).__name__}")
        if audio.ndim != 1:
            raise ValueError(f"audio must be mono 1-D array, got shape {audio.shape}")
        if audio.dtype != np.float32:
            raise ValueError(f"audio must have dtype np.float32, got {audio.dtype}")


def _supported_vad_options(vad_options_class) -> set[str]:
    """Return VADXOptions parameters that are safe to pass by keyword."""
    return {
        name
        for name, parameter in inspect.signature(
            vad_options_class.__init__
        ).parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _normalize_segments(segments_raw, min_segment_ms: int = 200) -> list[Segment]:
    """Convert raw VAD output to sorted Segment objects."""
    segments = []
    for seg in segments_raw or []:
        if not isinstance(seg, (list, tuple)) or len(seg) < 2:
            continue

        start_ms = int(seg[0])
        end_ms = int(seg[1])
        if end_ms <= start_ms:
            continue
        if end_ms - start_ms < min_segment_ms:
            continue

        segments.append(Segment(start_ms, end_ms))

    return sorted(segments, key=lambda item: item.start_ms)
