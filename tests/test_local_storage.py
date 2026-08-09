"""Tests for the local filesystem storage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from meetasr.storage.local import LocalStorage


@pytest.mark.asyncio
async def test_local_storage_save_load_delete_round_trip(tmp_path: Path) -> None:
    """LocalStorage keeps the complete lifecycle inside the temporary root."""
    storage_root = tmp_path / "media"
    storage = LocalStorage(root_dir=str(storage_root))
    payload = b"fake audio bytes"

    key = await storage.save(payload, "meeting.wav")

    stored_path = storage_root / key
    assert key.endswith("/meeting.wav")
    assert stored_path.is_file()
    assert await storage.load(key) == payload

    await storage.delete(key)

    assert not stored_path.exists()
    with pytest.raises(FileNotFoundError):
        await storage.load(key)
