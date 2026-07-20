from __future__ import annotations

import numpy as np
import librosa


def normalize_audio(
    audio: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
    target_sr: int = 16000,
) -> bytes:
    """
    Chuẩn hóa audio về PCM16 mono 16kHz.

    Parameters
    ----------
    audio:
        Raw PCM bytes.

    sample_rate:
        Sample rate của audio đầu vào.

    channels:
        Số kênh của audio đầu vào.

    sample_width:
        Bytes mỗi sample.
            2 -> PCM16
            4 -> float32

    target_sr:
        Sample rate mong muốn.

    Returns
    -------
    bytes
        PCM16 little-endian mono @ target_sr.
    """

    # -----------------------------
    # Decode bytes
    # -----------------------------
    if sample_width == 2:
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        samples /= 32768.0

    elif sample_width == 4:
        samples = np.frombuffer(audio, dtype=np.float32)

    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # -----------------------------
    # Stereo -> Mono
    # -----------------------------
    if channels > 1:
        samples = samples.reshape(-1, channels)
        samples = samples.mean(axis=1)

    # -----------------------------
    # Resample
    # -----------------------------
    if sample_rate != target_sr:
        samples = librosa.resample(
            samples,
            orig_sr=sample_rate,
            target_sr=target_sr,
        )

    # -----------------------------
    # Clip
    # -----------------------------
    samples = np.clip(samples, -1.0, 1.0)

    # -----------------------------
    # Float -> PCM16
    # -----------------------------
    pcm16 = (samples * 32767).astype(np.int16)

    return pcm16.tobytes()