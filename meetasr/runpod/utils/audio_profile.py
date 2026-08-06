"""Audio profiling helpers for preprocessing decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from meetasr.runpod.utils.audio import SAMPLE_RATE, load_audio


@dataclass
class AudioProfile:
    """Summary statistics for a loaded mono waveform."""

    duration: float
    sample_rate: int
    dtype: str
    shape: tuple[int, ...]
    peak: float
    rms: float
    dc_offset: float
    clipping_ratio: float
    near_silence_ratio: float
    noise_rms: float
    speech_rms: float
    estimated_snr_db: float

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data


def profile_audio(
    source,
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: float = 1e-3,
    clipping_threshold: float = 0.999,
    frame_duration_ms: float = 20.0,
    noise_percentile: float = 20.0,
    speech_percentile: float = 80.0,
) -> AudioProfile:
    """Load audio and compute simple signal statistics."""
    audio = load_audio(source, target_sr=sample_rate)
    return profile_waveform(
        audio,
        sample_rate=sample_rate,
        silence_threshold=silence_threshold,
        clipping_threshold=clipping_threshold,
        frame_duration_ms=frame_duration_ms,
        noise_percentile=noise_percentile,
        speech_percentile=speech_percentile,
    )


def profile_waveform(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: float = 1e-3,
    clipping_threshold: float = 0.999,
    frame_duration_ms: float = 20.0,
    noise_percentile: float = 20.0,
    speech_percentile: float = 80.0,
) -> AudioProfile:
    """Compute signal statistics and a frame-based SNR estimate."""
    if not isinstance(audio, np.ndarray):
        raise TypeError("audio must be a numpy.ndarray")
    if audio.ndim != 1:
        raise ValueError(f"audio must be mono 1-D array, got shape {audio.shape}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")
    if not 0 <= noise_percentile < speech_percentile <= 100:
        raise ValueError(
            "percentiles must satisfy 0 <= noise_percentile "
            "< speech_percentile <= 100"
        )

    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return AudioProfile(
            duration=0.0,
            sample_rate=sample_rate,
            dtype=str(audio.dtype),
            shape=audio.shape,
            peak=0.0,
            rms=0.0,
            dc_offset=0.0,
            clipping_ratio=0.0,
            near_silence_ratio=0.0,
            noise_rms=0.0,
            speech_rms=0.0,
            estimated_snr_db=0.0,
        )

    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    dc_offset = float(np.mean(audio, dtype=np.float64))
    clipping_ratio = float(np.mean(abs_audio >= clipping_threshold))
    near_silence_ratio = float(np.mean(abs_audio < silence_threshold))
    frame_length = max(1, round(sample_rate * frame_duration_ms / 1000.0))
    frame_rms = _compute_frame_rms(audio, frame_length)
    noise_rms = float(np.percentile(frame_rms, noise_percentile))
    speech_rms = float(np.percentile(frame_rms, speech_percentile))
    estimated_snr_db = _estimate_snr_db(noise_rms, speech_rms)

    return AudioProfile(
        duration=round(float(audio.size / sample_rate), 3),
        sample_rate=sample_rate,
        dtype=str(audio.dtype),
        shape=audio.shape,
        peak=round(peak, 6),
        rms=round(rms, 6),
        dc_offset=round(dc_offset, 6),
        clipping_ratio=round(clipping_ratio, 6),
        near_silence_ratio=round(near_silence_ratio, 6),
        noise_rms=round(noise_rms, 6),
        speech_rms=round(speech_rms, 6),
        estimated_snr_db=round(estimated_snr_db, 2),
    )


def _compute_frame_rms(audio: np.ndarray, frame_length: int) -> np.ndarray:
    """Return RMS values for consecutive non-overlapping frames."""
    full_frame_count = audio.size // frame_length
    values = []

    if full_frame_count:
        framed = audio[: full_frame_count * frame_length].reshape(
            full_frame_count,
            frame_length,
        )
        values.extend(
            np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=1))
        )

    remainder = audio[full_frame_count * frame_length :]
    if remainder.size:
        values.append(
            float(np.sqrt(np.mean(np.square(remainder, dtype=np.float64))))
        )

    return np.asarray(values, dtype=np.float64)


def _estimate_snr_db(noise_rms: float, speech_rms: float) -> float:
    """Estimate SNR from quiet and active frame percentiles."""
    if speech_rms <= 0:
        return 0.0
    if noise_rms <= 1e-8:
        return 100.0
    return float(min(100.0, max(0.0, 20.0 * np.log10(speech_rms / noise_rms))))
