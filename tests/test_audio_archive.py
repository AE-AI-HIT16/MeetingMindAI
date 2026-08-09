"""Tests for bounded-memory realtime audio archival."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from meetasr.streaming.audio_archive import AudioArchive


def test_short_audio_stays_in_memory() -> None:
    samples = np.array([0, 16384, -16384], dtype=np.int16)
    archive = AudioArchive(spool_threshold_bytes=64)
    archive.append(samples.tobytes())

    payload = archive.detach()

    assert payload.path is None
    assert payload.duration_ms == 0
    assert np.allclose(payload.load_numpy(), [0.0, 0.5, -0.5])


def test_long_audio_spools_to_file_and_handoff_owns_cleanup() -> None:
    samples = np.array([1, 2, 3, 4], dtype=np.int16)
    archive = AudioArchive(spool_threshold_bytes=4)
    archive.append(samples[:2].tobytes())
    assert archive.is_spooled is False
    archive.append(samples[2:].tobytes())
    assert archive.is_spooled is True

    payload = archive.detach()
    assert payload.path is not None
    path = Path(payload.path)
    assert path.exists()
    archive.reset()
    assert path.exists()
    assert np.allclose(
        payload.load_numpy(),
        samples.astype(np.float32) / 32768.0,
    )

    payload.cleanup()
    assert not path.exists()
