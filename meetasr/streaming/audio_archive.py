"""Bounded-memory archive for raw PCM16 realtime audio."""

from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_SPOOL_THRESHOLD_BYTES = 16 * 1024 * 1024


@dataclass(slots=True)
class ArchivedAudio:
    """Detached audio payload owned by a finalization job."""

    sample_count: int
    pcm_bytes: bytes | None = None
    path: str | None = None

    @property
    def duration_ms(self) -> int:
        return self.sample_count * 1000 // SAMPLE_RATE

    def load_numpy(self) -> np.ndarray:
        """Materialize float32 audio only when a worker is ready to process it."""
        raw = Path(self.path).read_bytes() if self.path else self.pcm_bytes or b""
        if len(raw) % 2:
            raise ValueError("PCM16 archive contains an incomplete sample")
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def cleanup(self) -> None:
        """Release a detached spool file after terminal processing."""
        if self.path is not None:
            Path(self.path).unlink(missing_ok=True)
            self.path = None
        self.pcm_bytes = None

    def open_wav(self) -> BinaryIO:
        """Return a seekable WAV stream suitable for durable storage upload."""
        stream = tempfile.SpooledTemporaryFile(
            max_size=DEFAULT_SPOOL_THRESHOLD_BYTES,
            mode="w+b",
        )
        with wave.open(stream, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            if self.path is not None:
                with Path(self.path).open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        wav_file.writeframesraw(chunk)
            else:
                wav_file.writeframesraw(self.pcm_bytes or b"")
        stream.seek(0)
        return stream


class AudioArchive:
    """Keep short recordings in memory and spill long recordings to disk."""

    def __init__(
        self,
        *,
        spool_threshold_bytes: int = DEFAULT_SPOOL_THRESHOLD_BYTES,
    ) -> None:
        if spool_threshold_bytes <= 0:
            raise ValueError("spool_threshold_bytes must be positive")
        self.spool_threshold_bytes = spool_threshold_bytes
        self._buffer = bytearray()
        self._file: BinaryIO | None = None
        self._path: str | None = None
        self._byte_count = 0

    @property
    def is_spooled(self) -> bool:
        return self._file is not None

    def append(self, audio: bytes) -> None:
        if not audio:
            return
        if self._file is None and (
            self._byte_count + len(audio) > self.spool_threshold_bytes
        ):
            self._spill_to_disk()
        if self._file is not None:
            self._file.write(audio)
        else:
            self._buffer.extend(audio)
        self._byte_count += len(audio)

    def get_pcm_bytes(self) -> bytes:
        if self._file is None:
            return bytes(self._buffer)
        self._file.flush()
        return Path(self._path or "").read_bytes()

    def get_numpy(self) -> np.ndarray:
        raw = self.get_pcm_bytes()
        if len(raw) % 2:
            raise ValueError("PCM16 archive contains an incomplete sample")
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def detach(self) -> ArchivedAudio:
        """Transfer archive ownership without materializing spooled audio."""
        sample_count = self._byte_count // 2
        if self._file is None:
            payload = ArchivedAudio(
                sample_count=sample_count,
                pcm_bytes=bytes(self._buffer),
            )
        else:
            self._file.flush()
            self._file.close()
            payload = ArchivedAudio(
                sample_count=sample_count,
                path=self._path,
            )
        self._buffer = bytearray()
        self._file = None
        self._path = None
        self._byte_count = 0
        return payload

    def reset(self) -> None:
        """Delete archive data still owned by this session."""
        if self._file is not None:
            self._file.close()
        if self._path is not None:
            Path(self._path).unlink(missing_ok=True)
        self._buffer.clear()
        self._file = None
        self._path = None
        self._byte_count = 0

    def _spill_to_disk(self) -> None:
        file_descriptor, path = tempfile.mkstemp(
            prefix="meetasr-realtime-",
            suffix=".pcm",
        )
        self._path = path
        self._file = os.fdopen(file_descriptor, "w+b")
        if self._buffer:
            self._file.write(self._buffer)
            self._buffer.clear()
