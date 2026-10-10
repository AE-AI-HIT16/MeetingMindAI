"""Frame-level output of the speaker segmentation model and how to read it.

pyannote segmentation-3.0 predicts, every ~17 ms inside 10 s windows, which of
up to 3 local speakers are active (including pairs). ``LocalSegmentation``
holds that output; the helpers turn it into speech/overlap measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 160000          # 10 s model input
FRAME_STEP_SAMPLES = 270         # model output resolution (~16.9 ms)
FRAME_STEP_S = FRAME_STEP_SAMPLES / SAMPLE_RATE


@dataclass
class LocalSegmentation:
    """Per-window local speaker activity plus global speaker count."""

    activity: np.ndarray      # [chunks, frames, 3] binary
    chunk_offsets: list[int]  # first global frame index of each chunk
    count: np.ndarray         # [global_frames] number of active speakers

    @property
    def num_frames(self) -> int:
        return int(self.count.shape[0])


def frame_of(seconds: float) -> int:
    """Global frame index of a time in seconds.

    Args:
        seconds: Time from the start of the audio.

    Returns:
        Nearest frame index (frames are ``FRAME_STEP_S`` apart).
    """
    return int(round(seconds / FRAME_STEP_S))


def speech_ratio(segmentation: LocalSegmentation, start_s: float, end_s: float) -> float:
    """Share of a time range where segmentation hears at least one speaker.

    Args:
        segmentation: Model output for the whole audio.
        start_s: Range start in seconds.
        end_s: Range end in seconds.

    Returns:
        Fraction in [0, 1]; 0.0 for a range outside the audio.
    """
    a, b = frame_of(start_s), max(frame_of(end_s), frame_of(start_s) + 1)
    window = segmentation.count[a:b]
    return float((window > 0).mean()) if window.size else 0.0


def overlap_regions(
    segmentation: LocalSegmentation,
    min_duration_s: float = 0.3,
) -> list[tuple[int, int]]:
    """Ranges where two or more people talk at once.

    Args:
        segmentation: Model output for the whole audio.
        min_duration_s: Shorter overlaps are ignored.

    Returns:
        Sorted ``(start_ms, end_ms)`` ranges.
    """
    regions = []
    start = None
    active = segmentation.count >= 2
    for index, value in enumerate(np.append(active, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if (index - start) * FRAME_STEP_S >= min_duration_s:
                regions.append((
                    int(round(start * FRAME_STEP_S * 1000)),
                    int(round(index * FRAME_STEP_S * 1000)),
                ))
            start = None
    return regions
