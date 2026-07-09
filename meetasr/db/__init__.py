"""MeetASR database package.

Exports the session dependency, initialisation helper, and all ORM models
so callers can import everything from a single location::

    from meetasr.db import get_db, init_db
    from meetasr.db import Meeting, Transcript, Report
"""

from meetasr.db.connection import get_db, init_db
from meetasr.db.models import (
    ActionItem,
    Decision,
    Meeting,
    Report,
    Sentence,
    Topic,
    Transcript,
)

__all__ = [
    # Connection helpers
    "get_db",
    "init_db",
    # ORM models
    "Meeting",
    "Transcript",
    "Sentence",
    "Report",
    "Topic",
    "ActionItem",
    "Decision",
]
