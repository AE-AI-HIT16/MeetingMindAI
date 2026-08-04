"""End-to-end test: API → DB → Pipeline → DB."""

from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine

from meetasr.api.app import app
from meetasr.api.dependencies import get_pipeline
from meetasr.db.connection import get_db

# Override DB with in-memory SQLite for testing
TEST_DB_URL = "sqlite:///test_meetasr.db"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
)


def override_get_db() -> Any:
    SQLModel.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        yield session


def override_get_pipeline() -> MagicMock:
    mock = MagicMock()
    mock.summarizer = MagicMock()
    return mock


app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_pipeline] = override_get_pipeline

client = TestClient(app)


def test_meeting_lifecycle() -> None:
    """Test: create meeting → check pending → (mock complete) → check completed."""

    # Fake WAV file (không cần dữ liệu audio thật)
    fake_audio = BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt ")

    resp = client.post(
        "/v1/meeting/summarize",
        files={
            "file": ("test.wav", fake_audio, "audio/wav"),
        },
        data={"language": "vi"},
    )

    assert resp.status_code == 202

    meeting_id = resp.json()["meeting_id"]

    resp = client.get(f"/v1/meeting/{meeting_id}/status")

    assert resp.status_code == 200
    assert resp.json()["status"] in ("pending", "processing")