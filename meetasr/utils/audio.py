"""Audio loading and preprocessing utilities."""

import logging
import os
from typing import Union

import numpy as np

SAMPLE_RATE = 16000  # All models expect 16kHz mono float32


def load_audio(
    source: Union[str, bytes, np.ndarray],
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Load audio from file path, URL, bytes, or numpy array.

    Automatically resamples to target_sr and converts to mono float32.

    Args:
        source: Audio file path (str), raw bytes, or numpy array.
        target_sr: Target sample rate. Default 16000.

    Returns:
        np.ndarray of shape (N,), dtype float32, at target_sr Hz.

    Raises:
        FileNotFoundError: If source is a path that does not exist.
        ValueError: If source type is not supported.
    """
    if isinstance(source, np.ndarray):
        return _normalize_array(source, target_sr)

    if isinstance(source, bytes):
        return _load_from_bytes(source, target_sr)

    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            return _load_from_url(source, target_sr)
        if not os.path.exists(source):
            raise FileNotFoundError(f"Audio file not found: {source}")
        return _load_from_file(source, target_sr)

    raise ValueError(
        f"Unsupported audio source type: {type(source)}. "
        "Expected str (path/url), bytes, or np.ndarray."
    )


def _load_from_file(path: str, target_sr: int) -> np.ndarray:
    """Load audio from file using soundfile/librosa."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        # Fallback to librosa for mp3, m4a, etc.
        import librosa
        audio, sr = librosa.load(path, sr=None, mono=False, dtype=np.float32)

    return _postprocess(audio, sr, target_sr)


def _load_from_bytes(data: bytes, target_sr: int) -> np.ndarray:
    """Load audio from raw bytes."""
    import io
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return _postprocess(audio, sr, target_sr)


def _load_from_url(url: str, target_sr: int) -> np.ndarray:
    """Download audio from URL then load."""
    import io
    import urllib.request
    logging.info(f"Downloading audio from URL: {url}")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return _load_from_bytes(data, target_sr)


def _normalize_array(audio: np.ndarray, target_sr: int) -> np.ndarray:
    """Normalize numpy array to float32 mono."""
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)   # stereo → mono (average channels)
    elif audio.ndim > 2:
        raise ValueError(f"Unexpected audio ndim: {audio.ndim}")
    return audio


def _postprocess(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Convert to mono float32 and resample if needed."""
    audio = audio.astype(np.float32)
    # Stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=-1)
    # Resample
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio
