# Re-export from meetasr.db.models to avoid duplicate SQLAlchemy table definitions.
# meetasr/backend/db/ and meetasr/db/ are identical packages; importing both would
# cause "Table X is already defined" errors because SQLModel uses a shared MetaData.
from meetasr.db.models import (  # noqa: F401
    ActionItem,
    Decision,
    Meeting,
    Report,
    Sentence,
    Topic,
    Transcript,
)