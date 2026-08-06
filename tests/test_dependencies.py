"""Tests for dependencies.py — save_upload() and safe_remove().

Run with: pytest tests/test_dependencies.py -v
"""
from __future__ import annotations

import os
import tempfile
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from meetasr.api.dependencies import MAX_FILE_BYTES, safe_remove, save_upload


# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------

def make_upload_file(filename: str, content: bytes = b"fake data") -> UploadFile:
    """Create a fake UploadFile object for testing.

    Args:
        filename: Name of the simulated file.
        content: Raw bytes content of the file.

    Returns:
        UploadFile instance backed by an in-memory BytesIO buffer.
    """
    return UploadFile(filename=filename, file=BytesIO(content))


# ---------------------------------------------------------
# save_upload() — Valid formats
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_save_upload_valid_wav():
    """A valid .wav file must be saved and its path must end with .wav."""
    path = await save_upload(make_upload_file("test.wav"))
    assert path.endswith(".wav")
    assert os.path.exists(path)
    os.remove(path)


@pytest.mark.asyncio
async def test_save_upload_valid_mp3():
    """A valid .mp3 file must be saved successfully."""
    path = await save_upload(make_upload_file("recording.mp3"))
    assert path.endswith(".mp3")
    assert os.path.exists(path)
    os.remove(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["a.m4a", "a.mp4", "a.flac", "a.ogg", "a.webm"])
async def test_save_upload_all_supported_formats(filename: str):
    """All 7 formats listed in technical.md must be accepted by save_upload."""
    ext = os.path.splitext(filename)[-1]
    path = await save_upload(make_upload_file(filename))
    assert path.endswith(ext)
    assert os.path.exists(path)
    os.remove(path)


# ---------------------------------------------------------
# save_upload() — Invalid formats → 400
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_save_upload_invalid_exe_returns_400():
    """An .exe file must be rejected with 400 and the correct error code from api_spec.md."""
    with pytest.raises(HTTPException) as exc_info:
        await save_upload(make_upload_file("malware.exe"))
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"]["code"] == "invalid_file_format"


@pytest.mark.asyncio
async def test_save_upload_no_extension_returns_400():
    """A file with no extension must be rejected with 400."""
    with pytest.raises(HTTPException) as exc_info:
        await save_upload(make_upload_file("justname"))
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"]["code"] == "invalid_file_format"


# ---------------------------------------------------------
# save_upload() — File size → 413
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_save_upload_file_too_large_returns_413():
    """A file exceeding 500 MB must be rejected with 413 and the correct error code."""
    oversized_content = b"x" * (MAX_FILE_BYTES + 1)
    with pytest.raises(HTTPException) as exc_info:
        await save_upload(make_upload_file("big.wav", oversized_content))
    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["error"]["code"] == "file_too_large"


# ---------------------------------------------------------
# save_upload() — Edge cases
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_save_upload_empty_file_is_valid():
    """An empty file (0 bytes) is format-valid and must be accepted."""
    path = await save_upload(make_upload_file("empty.wav", b""))
    assert os.path.exists(path)
    assert os.path.getsize(path) == 0
    os.remove(path)


@pytest.mark.asyncio
async def test_save_upload_none_filename_defaults_to_wav():
    """When the client sends no filename (None), the system must fall back to .wav extension.

    save_upload() uses: os.path.splitext(file.filename or 'audio.wav')
    → ext = '.wav' → tempfile suffix = '.wav'
    → resulting path is /tmp/tmpXXXXXX.wav (not 'audio.wav')
    """
    upload_file = UploadFile(file=BytesIO(b"data"))
    upload_file.filename = None
    path = await save_upload(upload_file)
    # Assert only the suffix — tempfile generates a random base name
    assert path.endswith(".wav")
    assert os.path.exists(path)
    os.remove(path)


# ---------------------------------------------------------
# safe_remove()
# ---------------------------------------------------------

def test_safe_remove_deletes_existing_file():
    """safe_remove must successfully delete a file that exists."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    assert os.path.exists(tmp.name)

    safe_remove(tmp.name)
    assert not os.path.exists(tmp.name)


def test_safe_remove_nonexistent_file_no_crash():
    """safe_remove must not raise an exception when the path does not exist."""
    safe_remove("/tmp/this_file_does_not_exist_xyz.wav")  # must not crash