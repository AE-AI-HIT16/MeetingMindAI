"""API tests for the Phase 2 DocumentPlanner prototype routes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import pytest
from fastapi import BackgroundTasks, HTTPException
from pydantic import TypeAdapter, ValidationError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from meetasr.api.routes import document
from meetasr.schemas_doc import DocSection, DocumentReport


class StubPlanner:
    """Deterministic planner used by API tests."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.transcript = None

    def plan_and_write(self, transcript) -> DocumentReport:
        """Return a fixed report or simulate an LLM failure."""
        self.transcript = transcript
        if self.fail:
            raise RuntimeError("simulated planner failure")
        return DocumentReport(
            content_kind="Bài giảng",
            sections=[
                DocSection(
                    id="s1",
                    heading="Tóm tắt",
                    kind="summary",
                    markdown="Nội dung chính.",
                )
            ],
            language="vi",
            llm_model="stub-model",
            processing_time=0.1,
        )


@pytest.fixture
def document_context(monkeypatch):
    """Create an isolated database and planner for each route unit test."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)
    monkeypatch.setattr(document, "engine", test_engine)

    pipeline = SimpleNamespace(doc_planner=StubPlanner())
    with Session(test_engine) as session:
        yield session, pipeline


def _payload() -> dict:
    return {
        "transcript_data": {
            "key": "lesson",
            "text": "Nội dung bài giảng.",
            "duration": 12.0,
            "language": "vi",
            "sentence_info": [
                {"text": "Nội dung bài giảng.", "start": 0.0, "end": 2.0}
            ],
        }
    }


@pytest.mark.asyncio
async def test_document_generation_lifecycle(document_context) -> None:
    """Generate, poll, and retrieve JSON/Markdown document output."""
    db, pipeline = document_context
    request = document.GenerateDocumentRequest.model_validate(_payload())
    background_tasks = BackgroundTasks()

    response = await document.generate_document(
        request,
        background_tasks,
        db,
        pipeline,
    )
    assert response.status == "pending"
    assert len(background_tasks.tasks) == 1

    task = background_tasks.tasks[0]
    task.func(*task.args, **task.kwargs)
    db.expire_all()

    status_response = await document.get_document_status(response.meeting_id, db)
    assert status_response.status == "completed"

    json_response = await document.get_document_report(response.meeting_id, "json", db)
    assert json_response.content_kind == "Bài giảng"

    markdown_response = await document.get_document_report(
        response.meeting_id,
        "markdown",
        db,
    )
    assert "# Bài giảng" in markdown_response.content
    assert pipeline.doc_planner.transcript.language == "vi"


def test_document_rejects_empty_transcript() -> None:
    """An empty transcript must be rejected before creating a background job."""
    with pytest.raises(ValidationError):
        document.GenerateDocumentRequest.model_validate(
            {"transcript_data": {"text": "", "sentence_info": []}}
        )


@pytest.mark.asyncio
async def test_document_requires_configured_planner(document_context) -> None:
    """Generation must return 503 when DocumentPlanner is unavailable."""
    db, pipeline = document_context
    pipeline.doc_planner = None
    request = document.GenerateDocumentRequest.model_validate(_payload())
    with pytest.raises(HTTPException) as exc_info:
        await document.generate_document(request, BackgroundTasks(), db, pipeline)
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"]["code"] == "planner_not_configured"


def test_document_rejects_invalid_sentence_range() -> None:
    """Sentence end timestamps must not precede start timestamps."""
    payload = _payload()
    payload["transcript_data"]["sentence_info"][0]["end"] = -1
    with pytest.raises(ValidationError):
        document.GenerateDocumentRequest.model_validate(payload)


def test_document_rejects_unknown_report_format() -> None:
    """Unknown output formats should return validation errors."""
    report_format = TypeAdapter(Literal["json", "markdown"])
    with pytest.raises(ValidationError):
        report_format.validate_python("pdf")


@pytest.mark.asyncio
async def test_document_reports_failed_background_job(document_context) -> None:
    """Planner failures must persist a failed status instead of crashing the API."""
    db, pipeline = document_context
    pipeline.doc_planner.fail = True
    request = document.GenerateDocumentRequest.model_validate(_payload())
    background_tasks = BackgroundTasks()
    response = await document.generate_document(
        request,
        background_tasks,
        db,
        pipeline,
    )

    task = background_tasks.tasks[0]
    task.func(*task.args, **task.kwargs)
    db.expire_all()

    status_response = await document.get_document_status(response.meeting_id, db)
    assert status_response.status == "failed"

    with pytest.raises(HTTPException) as exc_info:
        await document.get_document_report(response.meeting_id, "json", db)
    assert exc_info.value.status_code == 409
