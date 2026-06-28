"""API endpoint tests — covers health, transcribe, summarize, mock_db.

Run with: pytest tests/test_api.py -v
All tests run without downloading ML models (pipeline is mocked).
"""
from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from meetasr.api.app import app
from meetasr.api.dependencies import get_pipeline
from meetasr.api import mock_db


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


# Inject default mock pipeline (with LLM) for the entire test module
app.dependency_overrides[get_pipeline] = _override_get_pipeline

client = TestClient(app)


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


# ---------------------------------------------------------
# MEETING STATUS
# ---------------------------------------------------------

def test_meeting_status_returns_record_for_valid_id():
    """GET /v1/meeting/{id}/status with a valid ID must return 200 and the meeting record."""
    meeting_id = "status-test-valid"
    mock_db.create_meeting(meeting_id, "test.wav", "/tmp/test.wav")
    response = client.get(f"/v1/meeting/{meeting_id}/status")
    assert response.status_code == 200
    assert response.json()["status"] == "pending"


def test_meeting_status_not_found_returns_404_with_error_body():
    """GET /v1/meeting/{id}/status with unknown ID must return 404 with correct error body."""
    response = client.get("/v1/meeting/nonexistent-id-xyz/status")
    assert response.status_code == 404
    body = response.json()
    assert body["detail"]["error"]["code"] == "meeting_not_found"


# ---------------------------------------------------------
# MOCK DB — Direct unit tests (no HTTP)
# ---------------------------------------------------------

def test_mock_db_full_state_machine():
    """mock_db must transition correctly: pending → processing → completed with result."""
    test_id = "db-sm-001"
    mock_db.create_meeting(test_id, "meeting.wav", "/tmp/audio.wav")

    record = mock_db.get_meeting(test_id)
    assert record is not None
    assert record["status"] == "pending"

    mock_db.update_status(test_id, "processing")
    assert mock_db.get_meeting(test_id)["status"] == "processing"

    mock_db.update_status(test_id, "completed", result={"summary": "done"})
    final = mock_db.get_meeting(test_id)
    assert final["status"] == "completed"
    assert final["result"]["summary"] == "done"


def test_mock_db_get_nonexistent_returns_none():
    """mock_db.get_meeting with an unknown ID must return None."""
    assert mock_db.get_meeting("nonexistent-id-does-not-exist") is None


def test_mock_db_update_invalid_id_no_crash():
    """mock_db.update_status with an unknown ID must not raise an exception."""
    mock_db.update_status("nonexistent-id-does-not-exist", "processing")  # must not crash