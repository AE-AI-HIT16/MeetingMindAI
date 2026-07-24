"""API endpoint tests — covers health, transcribe, summarize, mock_db.

Run with: pytest tests/test_api.py -v
All tests run without downloading ML models (pipeline is mocked).
"""
from __future__ import annotations

from collections.abc import Generator
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from meetasr.api.app import app
from meetasr.api.dependencies import get_pipeline
from meetasr.api.routes import summarize
from meetasr.db.connection import get_db

# ---------------------------------------------------------
# HELPERS & MOCK SETUP
# ---------------------------------------------------------

class MockTranscriptResult:
    """Stub for TranscriptResult returned by pipeline.transcribe()."""

    text = "Hello world."

    def to_srt(self) -> str:
        return "1\n00:00:00,000 --> 00:00:01,000\nHello world."

    def to_dict(self) -> dict:
        # verbose_json format per api_spec.md
        return {
            "task": "transcribe",
            "language": "vi",
            "duration": 1.0,
            "text": self.text,
            "segments": [],
        }


def _make_mock_pipeline(with_summarizer: bool = True) -> MagicMock:
    """Build a mock MeetPipeline with configurable LLM summarizer.

    Args:
        with_summarizer: If True, attaches a MagicMock summarizer; else None.

    Returns:
        Configured MagicMock mimicking MeetPipeline.
    """
    mock = MagicMock()
    mock.vad = MagicMock()
    mock.asr = MagicMock()
    mock.asr.model_name = "sensevoice-small"
    mock.punc = MagicMock()
    mock.spk = None
    mock.summarizer = MagicMock() if with_summarizer else None
    mock.transcribe.return_value = MockTranscriptResult()
    return mock


def _override_get_pipeline() -> MagicMock:
    """Dependency override: full pipeline including LLM summarizer."""
    return _make_mock_pipeline(with_summarizer=True)


def _override_no_llm() -> MagicMock:
    """Dependency override: pipeline without LLM summarizer (summarizer=None)."""
    return _make_mock_pipeline(with_summarizer=False)


def _dummy_audio(filename: str = "test.wav") -> tuple[str, BytesIO, str]:
    """Create a fake audio file tuple for multipart upload.

    Args:
        filename: Name of the simulated audio file.

    Returns:
        Tuple of (filename, file_bytes, mime_type) accepted by TestClient.
    """
    return (filename, BytesIO(b"fake audio data"), "audio/wav")


client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def isolated_test_app() -> Generator[None, None, None]:
    """Run API tests against an isolated in-memory SQLite database."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)

    def override_get_db() -> Generator[Session, None, None]:
        with Session(test_engine) as session:
            yield session

    previous_db_override = app.dependency_overrides.get(get_db)
    previous_pipeline_override = app.dependency_overrides.get(get_pipeline)
    previous_background_engine = summarize.engine

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_pipeline] = _override_get_pipeline
    summarize.engine = test_engine
    try:
        yield
    finally:
        if previous_db_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = previous_db_override

        if previous_pipeline_override is None:
            app.dependency_overrides.pop(get_pipeline, None)
        else:
            app.dependency_overrides[get_pipeline] = previous_pipeline_override

        summarize.engine = previous_background_engine
        test_engine.dispose()


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

def test_health_check_status_ok():
    """GET /v1/health must return 200 with status='ok'."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_check_has_all_required_fields():
    """GET /v1/health must return all 4 fields required by api_spec.md."""
    response = client.get("/v1/health")
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "models_loaded" in data
    assert "llm_available" in data


# ---------------------------------------------------------
# TRANSCRIBE — Happy paths (all 4 response_format values)
# ---------------------------------------------------------

def test_transcribe_json_format():
    """POST /v1/audio/transcriptions with response_format=json must return {"text": ...}."""
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": _dummy_audio()},
        data={"response_format": "json"},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "Hello world."


def test_transcribe_text_format():
    """POST with response_format=text must return plain text, not JSON."""
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": _dummy_audio()},
        data={"response_format": "text"},
    )
    assert response.status_code == 200
    assert response.text == "Hello world."


def test_transcribe_srt_format():
    """POST with response_format=srt must return a valid SRT string."""
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": _dummy_audio()},
        data={"response_format": "srt"},
    )
    assert response.status_code == 200
    assert "00:00:00,000" in response.text


def test_transcribe_verbose_json_format():
    """POST with response_format=verbose_json must include task, language, duration, segments."""
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": _dummy_audio()},
        data={"response_format": "verbose_json"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "duration" in data
    assert "segments" in data


# ---------------------------------------------------------
# TRANSCRIBE — Error paths
# ---------------------------------------------------------

def test_transcribe_invalid_file_format_returns_400():
    """Uploading an unsupported format must return 400 with the correct error code per api_spec.md."""
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("malware.exe", BytesIO(b"data"), "application/octet-stream")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["detail"]["error"]["code"] == "invalid_file_format"


# ---------------------------------------------------------
# SUMMARIZE — Happy paths
# ---------------------------------------------------------

def test_summarize_returns_202_with_meeting_id():
    """POST /v1/meeting/summarize must return 202 Accepted with meeting_id and status=pending."""
    response = client.post(
        "/v1/meeting/summarize",
        files={"file": _dummy_audio()},
    )
    assert response.status_code == 202
    data = response.json()
    assert "meeting_id" in data
    assert data["status"] == "pending"


# ---------------------------------------------------------
# SUMMARIZE — Error paths
# ---------------------------------------------------------

def test_summarize_no_llm_returns_503():
    """POST /v1/meeting/summarize when LLM is not configured must return 503."""
    app.dependency_overrides[get_pipeline] = _override_no_llm
    try:
        response = client.post(
            "/v1/meeting/summarize",
            files={"file": _dummy_audio()},
        )
        assert response.status_code == 503
        body = response.json()
        assert body["detail"]["error"]["code"] == "llm_not_configured"
    finally:
        # Always restore the default override to prevent test pollution
        app.dependency_overrides[get_pipeline] = _override_get_pipeline
