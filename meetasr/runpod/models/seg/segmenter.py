"""pyannote segmentation-3.0 (ONNX) runner.

The ONNX export is ``onnx-community/pyannote-segmentation-3.0`` (MIT, not
gated), so neither ``pyannote.audio`` nor a Hugging Face token is required.
Besides finding overlapped speech, the per-frame speaker count gates VAD
segments without a voice (noise, laughter) before diarization and ASR.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from meetasr.runpod.models.abs_models import AbsSegmenter
from meetasr.runpod.utils.overlap import activity_to_segments, overlap_aware_activity
from meetasr.runpod.utils.segmentation import (
    FRAME_STEP_S,
    FRAME_STEP_SAMPLES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    LocalSegmentation,
)

logger = logging.getLogger(__name__)

# Powerset classes -> active local speakers (none, 1, 2, 3, 1+2, 1+3, 2+3).
_POWERSET = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
    dtype=np.float32,
)


class SpeakerSegmenter(AbsSegmenter):
    """Sliding-window runner for the segmentation-3.0 ONNX model."""

    def __init__(
        self,
        model: str = "onnx-community/pyannote-segmentation-3.0",
        filename: str = "onnx/model.onnx",
        step_s: float = 1.0,
        batch_size: int = 32,
        device: str = "cpu",
        boundary_s: float = 0.75,
        supplement: bool = True,
        min_segment_s: float = 0.1,
        fill_gap_s: float = 0.2,
        min_speech_ratio: float = 0.0,
        refine: bool = True,
        **_: object,
    ):
        """
        Args:
            model: Local ``.onnx`` path or Hugging Face repo id.
            filename: ONNX file inside the repo.
            step_s: Hop between 10 s windows (smaller = smoother, slower).
            batch_size: Windows per ONNX call.
            device: "cpu" or "cuda[:N]" (CUDA needs onnxruntime-gpu).
            boundary_s: How far from a cam++ speaker change segmentation may
                move the change.
            supplement: Keep cam++ speech where segmentation hears nobody.
            min_segment_s: Drop speaker pieces shorter than this.
            fill_gap_s: Close same-speaker gaps shorter than this.
            min_speech_ratio: Drop VAD segments where segmentation hears a
                voice in less than this share of the time (0 = keep all).
            refine: Rebuild speaker turns from segmentation (overlap-aware).
                False keeps cam++ turns and only uses the speech gate.

        Raises:
            ValueError: If ``step_s`` is not in (0, 10].
        """
        if step_s <= 0 or step_s > WINDOW_SAMPLES / SAMPLE_RATE:
            raise ValueError("step_s must be in (0, 10]")
        self.model = model
        self.filename = filename
        self.step_s = step_s
        self.batch_size = batch_size
        self.device = device
        self.boundary_s = boundary_s
        self.supplement = supplement
        self.min_segment_s = min_segment_s
        self.fill_gap_s = fill_gap_s
        self.min_speech_ratio = min_speech_ratio
        self.refine_turns = refine
        self.provider: str | None = None
        self._session = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort

        path = self.model
        if not os.path.isfile(path):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self.model, self.filename)
        providers = ["CPUExecutionProvider"]
        if self.device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            # onnxruntime-gpu >= 1.21 can load the CUDA/cuDNN libraries that
            # torch's pip wheels already installed.
            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception as exc:
                    logger.warning("onnxruntime preload_dlls failed: %s", exc)
            providers.insert(0, "CUDAExecutionProvider")
        self._session = ort.InferenceSession(path, providers=providers)
        self._input = self._session.get_inputs()[0].name
        # ORT silently falls back to CPU when CUDA libraries are missing.
        self.provider = self._session.get_providers()[0]
        logger.info("Segmentation model loaded: %s (%s)", path, self.provider)

    def __call__(self, audio: np.ndarray) -> LocalSegmentation:
        """Predict local speaker activity over the whole audio.

        Args:
            audio: Float32 mono audio at 16kHz.

        Returns:
            LocalSegmentation with per-window activity and per-frame counts.
        """
        self._ensure_loaded()
        audio = np.asarray(audio, dtype=np.float32)
        step = int(self.step_s * SAMPLE_RATE)
        starts = list(range(0, max(len(audio) - WINDOW_SAMPLES, 0) + 1, step))
        if starts[-1] + WINDOW_SAMPLES < len(audio):
            starts.append(len(audio) - WINDOW_SAMPLES)

        windows = np.zeros((len(starts), 1, WINDOW_SAMPLES), dtype=np.float32)
        for index, start in enumerate(starts):
            piece = audio[start:start + WINDOW_SAMPLES]
            windows[index, 0, :len(piece)] = piece

        outputs = []
        for index in range(0, len(windows), self.batch_size):
            logits = self._session.run(None, {self._input: windows[index:index + self.batch_size]})[0]
            outputs.append(logits)
        logits = np.concatenate(outputs, axis=0)
        # Hard powerset decoding: most likely class -> set of local speakers.
        activity = _POWERSET[np.argmax(logits, axis=-1)]

        frames_per_chunk = activity.shape[1]
        offsets = [int(round(start / FRAME_STEP_SAMPLES)) for start in starts]
        total_frames = int(np.ceil(len(audio) / FRAME_STEP_SAMPLES))
        summed = np.zeros(total_frames + frames_per_chunk, dtype=np.float32)
        covered = np.zeros_like(summed)
        for offset, chunk in zip(offsets, activity):
            summed[offset:offset + frames_per_chunk] += chunk.sum(axis=1)
            covered[offset:offset + frames_per_chunk] += 1.0
        count = np.rint(summed[:total_frames] / np.maximum(covered[:total_frames], 1.0))
        return LocalSegmentation(activity, offsets, count.astype(np.int64))

    def refine(
        self,
        audio: np.ndarray,
        cluster_segments: list[list],
        segmentation: LocalSegmentation | None = None,
    ) -> tuple[list[list], float]:
        """Turn cam++ segments into overlap-aware speaker segments.

        Args:
            audio: Float32 mono audio at 16kHz.
            cluster_segments: ``[start_s, end_s, speaker]`` from clustering.
            segmentation: Precomputed output of ``__call__`` (optional).

        Returns:
            Segments (may overlap) and the number of overlapped seconds.
        """
        binary = overlap_aware_activity(
            cluster_segments,
            segmentation if segmentation is not None else self(audio),
            supplement=self.supplement,
            boundary_frames=int(round(self.boundary_s / FRAME_STEP_S)),
        )
        segments = activity_to_segments(
            binary,
            min_duration_s=self.min_segment_s,
            fill_gap_s=self.fill_gap_s,
        )
        overlapped_s = float((binary.sum(axis=1) >= 2).sum()) * FRAME_STEP_S
        return segments, overlapped_s
