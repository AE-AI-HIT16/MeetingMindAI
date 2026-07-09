"""Shared dependencies for API routes."""
from __future__ import annotations
import os
import tempfile
import logging
from typing import Optional
from fastapi import HTTPException, UploadFile, status
from meetasr.pipeline import MeetPipeline


SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".webm"}
MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB

_pipeline: Optional[MeetPipeline] = None
CONFIG_PATH = os.environ.get("MEETASR_CONFIG", "meeting_config.yaml")
logger = logging.getLogger(__name__)


def get_pipeline() -> MeetPipeline:
    """Return the loaded pipeline instance.
    Raises:
        HTTPException 503: If pipeline is not loaded.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "code": "model_not_loaded",
                    "message": (
                        "ASR model is not loaded. "
                        f"Create '{CONFIG_PATH}' and restart the server. "
                        "Call GET /v1/health to check status."
                    ),
                }
            },
        )
    return _pipeline


def set_pipeline(p: Optional[MeetPipeline]) -> None:
    """Set the global pipeline instance (called at startup)."""
    global _pipeline
    _pipeline = p


async def save_upload(file: UploadFile) -> str:
    """Save uploaded file to a temp path and return the path.
    Raises:
        HTTPException 400: Unsupported file format.
        HTTPException 413: File exceeds 500 MB limit.
    """
    ext = os.path.splitext(file.filename or "audio.wav")[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "invalid_file_format",
                    "message": (
                        f"File format not supported. "
                        f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
                    ),
                }
            },
        )
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "error": {
                    "code": "file_too_large",
                    "message": (
                        f"File size exceeds maximum limit of 500MB. "
                        f"Got: {len(content) // 1024 // 1024} MB."
                    ),
                }
            },
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(content)
    tmp.close()
    return tmp.name


def safe_remove(path: str) -> None:
    """Remove a temporary file, ignoring errors."""
    try:
        os.remove(path)
    except Exception:
        logger.debug(f"Failed to remove temp file: {path}")
