"""Lightweight audio preprocessing helpers."""

from __future__ import annotations

import numpy as np


def rms_normalize(
    audio: np.ndarray,
    target_rms: float = 0.05,
    max_gain: float = 3.0,
    peak_limit: float = 0.99,
    min_rms: float = 1e-6,
) -> np.ndarray:
    """Raise quiet audio toward a target RMS without introducing clipping.

    Audio that is silent or already at least as loud as ``target_rms`` is
    returned unchanged apart from conversion to a contiguous float32 array.

    Args:
        audio: Mono waveform with shape ``[samples]``.
        target_rms: Desired minimum root-mean-square amplitude.
        max_gain: Maximum amplification factor.
        peak_limit: Maximum absolute output amplitude.
        min_rms: Values below this RMS are treated as silence.

    Returns:
        A new contiguous float32 mono waveform.

    Raises:
        TypeError: If ``audio`` is not a numpy array.
        ValueError: If the waveform shape or parameters are invalid.
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError("audio must be a numpy.ndarray")
    if audio.ndim != 1:
        raise ValueError(f"audio must be mono 1-D array, got shape {audio.shape}")
    if target_rms <= 0:
        raise ValueError("target_rms must be positive")
    if max_gain < 1:
        raise ValueError("max_gain must be at least 1")
    if not 0 < peak_limit <= 1:
        raise ValueError("peak_limit must be in the range (0, 1]")
    if min_rms < 0:
        raise ValueError("min_rms must be non-negative")

    waveform = np.ascontiguousarray(audio, dtype=np.float32)
    if waveform.size == 0:
        return waveform.copy()

    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    if rms <= min_rms or rms >= target_rms:
        return waveform.copy()

    peak = float(np.max(np.abs(waveform)))
    gain = min(target_rms / rms, max_gain)
    if peak > 0:
        gain = min(gain, peak_limit / peak)

    normalized = waveform * np.float32(gain)
    return np.ascontiguousarray(
        np.clip(normalized, -peak_limit, peak_limit),
        dtype=np.float32,
    )
