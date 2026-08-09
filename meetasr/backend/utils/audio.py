"""Audio loading and preprocessing utilities."""

import logging
import os
import shutil
import subprocess
from typing import Union

import numpy as np

SAMPLE_RATE = 16000  # All models expect 16kHz mono float32
FFMPEG_FORMATS = {".aac", ".m4a", ".mp3", ".mp4", ".webm"}


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
    """Load audio from a file, using FFmpeg for compressed containers."""
    if os.path.splitext(path)[1].lower() in FFMPEG_FORMATS:
        return _load_with_ffmpeg(path, target_sr)

    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        # Fallback to librosa for mp3, m4a, etc.
        logging.debug(f"soundfile failed to load {path} ({e}), falling back to librosa.")
        import librosa
        audio, sr = librosa.load(path, sr=None, mono=False, dtype=np.float32)
        # librosa returns multi-channel audio as [channels, samples], unlike
        # soundfile's [samples, channels] layout handled by _postprocess.
        if audio.ndim == 2:
            audio = audio.mean(axis=0, dtype=np.float32)

    return _postprocess(audio, sr, target_sr)


def _load_with_ffmpeg(path: str, target_sr: int) -> np.ndarray:
    """Decode a file to mono float32 PCM with the FFmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg is required to decode this audio format. "
            "Install ffmpeg and ensure it is available on PATH."
        )

    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"Failed to start FFmpeg: {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"FFmpeg could not decode audio file {path!r}: {detail}")

    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.size == 0:
        raise ValueError(f"FFmpeg decoded no audio samples from {path!r}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Decoded audio contains non-finite samples: {path!r}")
    return audio


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
