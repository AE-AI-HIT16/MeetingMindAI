"""MeetASR database package.

Exports the session dependency, initialisation helper, and all ORM models
so callers can import everything from a single location::

    from meetasr.db import get_db, init_db
    from meetasr.db import Meeting, Transcript, Report
    from meetasr.db import Source, Job, TranscriptSegment, Document
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
from meetasr.db.models_phase2 import (
    Document,
    DocumentMode,
    Job,
    JobStage,
    JobStatus,
    MediaType,
    Source,
    TranscriptSegment,
)

__all__ = [
    # Connection helpers
    "get_db",
    "init_db",
    # Phase 1 ORM models
    "Meeting",
    "Transcript",
    "Sentence",
    "Report",
    "Topic",
    "ActionItem",
    "Decision",
    # Phase 2 ORM models
    "Source",
    "Job",
    "TranscriptSegment",
    "Document",
    # Phase 2 constants
    "JobStatus",
    "JobStage",
    "DocumentMode",
    "MediaType",
]
