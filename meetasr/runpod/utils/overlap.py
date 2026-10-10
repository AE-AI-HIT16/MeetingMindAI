"""Overlap-aware speaker activity with pyannote segmentation-3.0 (ONNX).

The cam++ clustering gives one speaker per 1.5 s window, so it can neither see
two people talking at once nor place a speaker change more precisely than
~0.75 s. segmentation-3.0 predicts, every ~17 ms inside 10 s windows, which of
up to 3 local speakers are active (including pairs). Mapping those local
speakers onto the global cam++ clusters yields frame-level, overlap-aware
speaker activity.

The post-processing is ported from 3D-Speaker ``infer_diarization.py``
(``--include_overlap``, Apache 2.0). The ONNX export is
``onnx-community/pyannote-segmentation-3.0`` (MIT, not gated), so neither
``pyannote.audio`` nor a Hugging Face token is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 160000          # 10 s model input
FRAME_STEP_SAMPLES = 270         # model output resolution (~16.9 ms)
FRAME_SIZE_SAMPLES = 991         # receptive field of one output frame
FRAME_STEP_S = FRAME_STEP_SAMPLES / SAMPLE_RATE
FRAME_SIZE_S = FRAME_SIZE_SAMPLES / SAMPLE_RATE
# Powerset classes -> active local speakers (none, 1, 2, 3, 1+2, 1+3, 2+3).
_POWERSET = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
    dtype=np.float32,
)


@dataclass
class LocalSegmentation:
    """Per-window local speaker activity plus global speaker count."""

    activity: np.ndarray      # [chunks, frames, 3] binary
    chunk_offsets: list[int]  # first global frame index of each chunk
    count: np.ndarray         # [global_frames] number of active speakers

    @property
    def num_frames(self) -> int:
        return int(self.count.shape[0])


class SpeakerSegmenter:
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
        **_: object,
    ):
        """
        Args:
            model: Local ``.onnx`` path or Hugging Face repo id.
            filename: ONNX file inside the repo.
            step_s: Hop between 10 s windows (smaller = smoother, slower).
            boundary_s: How far from a cam++ speaker change segmentation may
                move the change.
            supplement: Keep cam++ speech where segmentation hears nobody.
            min_segment_s: Drop speaker pieces shorter than this.
            fill_gap_s: Close same-speaker gaps shorter than this.
            min_speech_ratio: Drop VAD segments where segmentation hears a
                voice in less than this share of the time (0 = keep all).
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
        self._session = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        import os

        import onnxruntime as ort

        path = self.model
        if not os.path.isfile(path):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self.model, self.filename)
        providers = ["CPUExecutionProvider"]
        if self.device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self._session = ort.InferenceSession(path, providers=providers)
        self._input = self._session.get_inputs()[0].name
        logger.info("Segmentation model loaded: %s (%s)", path, providers[0])

    def __call__(self, audio: np.ndarray) -> LocalSegmentation:
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
        """cam++ segments -> overlap-aware segments, plus overlapped seconds."""
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


def frame_of(seconds: float) -> int:
    return int(round(seconds / FRAME_STEP_S))


def speech_ratio(segmentation: LocalSegmentation, start_s: float, end_s: float) -> float:
    """Share of ``[start_s, end_s]`` where segmentation hears a speaker."""
    a, b = frame_of(start_s), max(frame_of(end_s), frame_of(start_s) + 1)
    window = segmentation.count[a:b]
    return float((window > 0).mean()) if window.size else 0.0


def speech_regions(segmentation: LocalSegmentation, min_duration_s: float = 0.0) -> list[tuple[float, float]]:
    """Regions where segmentation hears at least one speaker (seconds)."""
    active = segmentation.count > 0
    regions = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            regions.append((start * FRAME_STEP_S, index * FRAME_STEP_S))
            start = None
    if start is not None:
        regions.append((start * FRAME_STEP_S, len(active) * FRAME_STEP_S))
    return [(a, b) for a, b in regions if b - a >= min_duration_s]


def overlap_aware_activity(
    cluster_segments: list[list],
    segmentation: LocalSegmentation,
    *,
    supplement: bool = True,
    min_exclusive_frames: int = 12,
    boundary_frames: int = 45,
) -> np.ndarray:
    """Combine global clusters with local segmentation -> [frames, speakers].

    ``cluster_segments`` are ``[start_s, end_s, speaker]`` from cam++
    clustering (non-overlapping). Each 10 s chunk's local speakers are matched
    to global speakers (Hungarian) by co-activity; per frame the ``count``
    most active speakers are kept, which is how overlap appears.

    Unlike 3D-Speaker, co-activity is counted on frames where the local
    speaker talks alone: cam++ windows inside overlapped speech carry the
    louder voice's label and made whole chunks swap speakers. A local speaker
    heard only on top of someone else (a short "dạ", "vâng") falls back to
    all of its frames.

    Who speaks in single-speaker frames comes from cam++ (it tells voices
    apart better than the 10 s local model); segmentation decides whether
    anyone speaks, moves speaker changes within ``boundary_frames`` of a
    cam++ change to the right frame, and adds the second voice in overlap.
    """
    num_speakers = max(int(seg[2]) for seg in cluster_segments) + 1
    num_frames = segmentation.num_frames
    cluster_frames = np.zeros((num_frames, num_speakers), dtype=np.float32)
    for start_s, end_s, speaker in cluster_segments:
        cluster_frames[frame_of(start_s):frame_of(end_s), int(speaker)] = 1.0

    activations = np.zeros((num_frames, num_speakers), dtype=np.float32)
    for offset, local in zip(segmentation.chunk_offsets, segmentation.activity):
        local = local[: max(0, num_frames - offset)]
        if local.size == 0:
            continue
        global_frames = cluster_frames[offset:offset + len(local)]
        exclusive = local * (local.sum(axis=1, keepdims=True) == 1)
        use_exclusive = exclusive.sum(axis=0) >= min_exclusive_frames
        basis = np.where(use_exclusive[None, :], exclusive, local)
        totals = basis.sum(axis=0)
        share = (basis.T @ global_frames) / np.maximum(totals, 1.0)[:, None]
        share[totals == 0] = -1.0
        rows, cols = linear_sum_assignment(-share)
        aligned = np.zeros((len(local), num_speakers), dtype=np.float32)
        for row, col in zip(rows, cols):
            if share[row, col] > 0:
                aligned[:, col] = np.maximum(aligned[:, col], local[:, row])
        activations[offset:offset + len(local)] += aligned

    binary = _select_speakers(
        activations,
        cluster_frames,
        segmentation.count,
        boundary_frames=boundary_frames,
    )

    if supplement:
        # Frames the clustering covered but segmentation left empty.
        empty = (binary.sum(axis=1) == 0) & (cluster_frames.sum(axis=1) > 0)
        binary[empty] = cluster_frames[empty]
    return binary


def _select_speakers(
    activations: np.ndarray,
    cluster_frames: np.ndarray,
    count: np.ndarray,
    *,
    boundary_frames: int,
) -> np.ndarray:
    num_frames, num_speakers = activations.shape
    cluster_label = np.where(
        cluster_frames.sum(axis=1) > 0, np.argmax(cluster_frames, axis=1), -1
    )
    # Distance (frames) to the nearest cam++ speaker change.
    changes = np.flatnonzero(np.diff(cluster_label) != 0) + 1
    near_change = np.zeros(num_frames, dtype=bool)
    for change in changes:
        near_change[max(0, change - boundary_frames):change + boundary_frames] = True

    ranked = np.argsort(-activations, axis=1)
    binary = np.zeros_like(activations)
    for frame in range(num_frames):
        wanted = min(num_speakers, int(count[frame]))
        if wanted == 0:
            continue
        top = ranked[frame, 0]
        primary = cluster_label[frame]
        if primary < 0 or (
            near_change[frame]
            and activations[frame, top] > activations[frame, primary]
        ):
            primary = top if activations[frame, top] > 0 else primary
        if primary < 0:
            continue
        binary[frame, primary] = 1.0
        added = 1
        for speaker in ranked[frame]:
            if added >= wanted:
                break
            if speaker != primary and activations[frame, speaker] > 0:
                binary[frame, speaker] = 1.0
                added += 1
    return binary


def activity_to_segments(
    binary: np.ndarray,
    *,
    min_duration_s: float = 0.0,
    fill_gap_s: float = 0.0,
) -> list[list]:
    """Frame activity -> sorted ``[start_s, end_s, speaker]`` (may overlap).

    Gaps shorter than ``fill_gap_s`` inside one speaker's activity are closed
    first, then pieces shorter than ``min_duration_s`` are dropped.
    """
    segments = []
    max_gap = int(round(fill_gap_s / FRAME_STEP_S))
    for speaker in range(binary.shape[1]):
        column = binary[:, speaker] > 0.5
        if max_gap > 0:
            column = _close_gaps(column, max_gap)
        start = None
        for frame, value in enumerate(column):
            if value and start is None:
                start = frame
            elif not value and start is not None:
                segments.append([start, frame, speaker])
                start = None
        if start is not None:
            segments.append([start, len(column), speaker])
    result = [
        [round(a * FRAME_STEP_S, 3), round(b * FRAME_STEP_S, 3), int(spk)]
        for a, b, spk in segments
        if (b - a) * FRAME_STEP_S >= min_duration_s
    ]
    return sorted(result, key=lambda item: (item[0], item[1]))


def _close_gaps(column: np.ndarray, max_gap: int) -> np.ndarray:
    column = column.copy()
    active = np.flatnonzero(column)
    if active.size < 2:
        return column
    for left, right in zip(active[:-1], active[1:]):
        if 1 < right - left <= max_gap + 1:
            column[left:right] = True
    return column
