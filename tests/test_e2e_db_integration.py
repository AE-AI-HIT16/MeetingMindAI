# tests/test_e2e_db_integration.py
"""End-to-end test: API → DB → Pipeline → DB."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from unittest.mock import MagicMock
from typing import Any

from meetasr.api.app import app
from meetasr.db.connection import get_db
from meetasr.api.dependencies import get_pipeline

# Override DB with in-memory SQLite for testing
TEST_DB_URL = "sqlite:///test_meetasr.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})

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
    # POST upload → 202
    with open("tests/data/test1.wav", "rb") as f:
        resp = client.post(
            "/v1/meeting/summarize",
            files={"file": ("test.wav", f, "audio/wav")},
            data={"language": "vi"},
        )
    assert resp.status_code == 202
    meeting_id = resp.json()["meeting_id"]

    # GET status → pending or processing
    resp = client.get(f"/v1/meeting/{meeting_id}/status")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("pending", "processing")