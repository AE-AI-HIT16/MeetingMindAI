"""In-memory store for meeting status tracking.
Temporary implementation — will be replaced by real DB (SQLModel)
when the Data Engineer completes the database layer.
"""
from __future__ import annotations
import logging
from typing import Optional
from datetime import datetime

_logger = logging.getLogger(__name__)

_meetings: dict[str, dict] = {}


def create_meeting(meeting_id: str, title: str, audio_path: str) -> dict:
    """Create a new meeting record with 'pending' status.
    Args:
        meeting_id: Unique identifier for the meeting.
        title: Display name (typically the filename).
        audio_path: Path to the temporary audio file.
    Returns:
        The newly created meeting record dict.
    """
    record = {
        "id": meeting_id,
        "title": title,
        "audio_path": audio_path,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    }
    _meetings[meeting_id] = record
    return record


def update_status(meeting_id: str, status: str, result: Optional[dict] = None) -> None:
    """Update the status of an existing meeting record.
    Args:
        meeting_id: Target meeting identifier.
        status: New status (processing, completed, failed).
        result: Optional result dict to attach to the record.
    """
    if meeting_id in _meetings:
        _meetings[meeting_id]["status"] = status
        if result is not None:
            _meetings[meeting_id]["result"] = result


def get_meeting(meeting_id: str) -> dict | None:
    """Retrieve a meeting record by its ID.

    Args:
        meeting_id: The unique identifier of the meeting.

    Returns:
        The meeting dictionary if found, else None.
    """
    return _meetings.get(meeting_id)
