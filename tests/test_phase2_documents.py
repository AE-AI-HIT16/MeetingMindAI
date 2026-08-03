"""Unit tests for the persisted Phase 2 document contract."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from meetasr.api.routes import documents_phase2
from meetasr.api.schemas_phase2 import FinalizeDocumentRequest
from meetasr.db.models_phase2 import (
    Document,
    DocumentGenerationJob,
    DocumentGenerationStatus,
    DocumentMode,
    Job,
    JobStatus,
    MediaType,
    Source,
    TranscriptSegment,
)
from meetasr.db.user_model import User
from meetasr.realtime.document_generation import DocumentGenerationQueue
from meetasr.schemas_doc import DocSection, DocumentReport
from meetasr.services import document_service
from meetasr.services.document_service import DocumentService


@pytest.fixture
def document_context():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        source = Source(
            filename="meeting.wav",
            media_type=MediaType.AUDIO,
            duration=3.0,
            storage_path="meeting.wav",
        )
        db.add(source)
        db.flush()
        job = Job(source_id=source.id, status=JobStatus.DONE)
        db.add(job)
        db.flush()
        db.add(
            TranscriptSegment(
                job_id=job.id,
                start_ms=0,
                end_ms=3000,
                speaker=None,
                text="Nội dung cuộc họp.",
            )
        )
        live = Document(
            source_id=source.id,
            mode=DocumentMode.LIVE,
            markdown="# Live",
        )
        db.add(live)
        db.commit()
        db.refresh(live)
        yield engine, db, source.id, live.id
    engine.dispose()


@pytest.mark.asyncio
async def test_finalize_full_text_uses_live_document_id(document_context) -> None:
    _, db, source_id, live_id = document_context
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(pipeline=None))
    )

    response = await documents_phase2.finalize_document(
        live_id,
        FinalizeDocumentRequest(mode="full_text"),
        request,
        db,
    )

    assert response.sourceId == source_id
    assert response.mode == DocumentMode.FULL_TEXT
    assert response.status == DocumentGenerationStatus.QUEUED

    stored = db.get(Document, response.documentId)
    assert stored is not None
    assert stored.mode == DocumentMode.FULL_TEXT
    assert stored.markdown == ""


def test_finalize_route_declares_202_accepted() -> None:
    route = next(
        route
        for route in documents_phase2.router.routes
        if route.path.endswith("/{document_id}/finalize")
    )
    assert route.status_code == 202


@pytest.mark.asyncio
async def test_finalize_summary_calls_document_planner(
    document_context,
) -> None:
    _, db, _, live_id = document_context

    class StubPlanner:
        def plan_and_write(self, transcript):
            assert transcript.text == "Nội dung cuộc họp."
            return DocumentReport(
                content_kind="Cuộc họp",
                sections=[
                    DocSection(
                        id="summary",
                        heading="Tóm tắt",
                        kind="summary",
                        markdown="Kết quả tóm tắt.",
                    )
                ],
            )

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                pipeline=SimpleNamespace(doc_planner=StubPlanner())
            )
        )
    )
    response = await documents_phase2.finalize_document(
        live_id,
        FinalizeDocumentRequest(mode="summary"),
        request,
        db,
    )

    assert response.mode == DocumentMode.SUMMARY
    assert response.status == DocumentGenerationStatus.QUEUED


@pytest.mark.asyncio
async def test_finalize_returns_existing_summary_without_calling_planner(
    document_context,
) -> None:
    _, db, source_id, live_id = document_context
    existing = Document(
        source_id=source_id,
        mode=DocumentMode.SUMMARY,
        markdown="# Tóm tắt đã lưu",
    )
    db.add(existing)
    db.commit()
    db.refresh(existing)

    class PlannerMustNotRun:
        def plan_and_write(self, transcript):
            raise AssertionError("existing summary must not call the LLM")

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                pipeline=SimpleNamespace(doc_planner=PlannerMustNotRun())
            )
        )
    )

    response = await documents_phase2.finalize_document(
        live_id,
        FinalizeDocumentRequest(mode="summary"),
        request,
        db,
    )

    assert response.documentId == existing.id
    assert response.status == DocumentGenerationStatus.DONE


@pytest.mark.asyncio
async def test_background_summary_provider_failure_is_persisted(
    document_context,
    monkeypatch,
) -> None:
    test_engine, db, _, live_id = document_context

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(document_service.asyncio, "to_thread", run_inline)

    class FailingPlanner:
        def plan_and_write(self, transcript, progress_callback=None):
            raise RuntimeError("429 Too Many Requests")

    queue = DocumentGenerationQueue(database_engine=test_engine)
    await queue.start(FailingPlanner())
    try:
        generation = await queue.submit(
            db,
            live_id,
            DocumentMode.SUMMARY,
            planner=FailingPlanner(),
        )
        await queue.join()
        db.expire_all()
        stored = db.get(DocumentGenerationJob, generation.id)
        assert stored is not None
        assert stored.status == DocumentGenerationStatus.FAILED
        assert "đợi khoảng một phút" in (stored.error or "")
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_duplicate_finalize_reuses_one_job_and_calls_planner_once(
    document_context,
    monkeypatch,
) -> None:
    test_engine, db, _, live_id = document_context

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(document_service.asyncio, "to_thread", run_inline)

    class CountingPlanner:
        calls = 0

        def plan_and_write(self, transcript, progress_callback=None):
            self.calls += 1
            if progress_callback is not None:
                progress_callback(0.5)
            return DocumentReport(
                content_kind="Cuộc họp",
                sections=[
                    DocSection(
                        id="summary",
                        heading="Tóm tắt",
                        kind="summary",
                        markdown="Kết quả duy nhất.",
                    )
                ],
            )

    planner = CountingPlanner()
    queue = DocumentGenerationQueue(database_engine=test_engine)
    first = await queue.submit(
        db,
        live_id,
        DocumentMode.SUMMARY,
        planner=planner,
    )
    second = await queue.submit(
        db,
        live_id,
        DocumentMode.SUMMARY,
        planner=planner,
    )
    assert second.id == first.id

    await queue.start(planner)
    try:
        await queue.join()
        db.expire_all()
        generation = db.get(DocumentGenerationJob, first.id)
        document = db.get(Document, first.document_id)
        assert generation is not None
        assert generation.status == DocumentGenerationStatus.DONE
        assert document is not None
        assert "Kết quả duy nhất." in document.markdown
        assert planner.calls == 1
    finally:
        await queue.stop()


def test_export_markdown_returns_attachment(document_context) -> None:
    _, db, _, live_id = document_context

    response = documents_phase2.export_document(live_id, db, "md")

    assert response.media_type.startswith("text/markdown")
    assert response.body == b"# Live"
    assert "attachment" in response.headers["content-disposition"]


def test_full_text_export_uses_friendly_title_and_filename(document_context) -> None:
    _, db, source_id, _ = document_context
    document = Document(
        source_id=source_id,
        mode=DocumentMode.FULL_TEXT,
        markdown="# Toàn văn: meeting.wav\n\nNội dung.",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    artifact = DocumentService(db).export_document(document.id, "md")

    assert artifact.filename == "Toàn văn - meeting.md"
    assert artifact.content.startswith("# Toàn văn cuộc họp".encode())
    assert b"To\xc3\xa0n v\xc4\x83n: meeting.wav" not in artifact.content
    assert "_Nguồn: meeting_".encode() in artifact.content


def test_export_route_passes_selected_pdf_preset(document_context) -> None:
    _, db, _, live_id = document_context

    response = documents_phase2.export_document(
        live_id,
        db,
        "pdf",
        "blue_modern",
    )

    assert response.media_type == "application/pdf"
    assert response.body.startswith(b"%PDF-")


def test_export_route_rejects_preset_from_another_format(document_context) -> None:
    _, db, _, live_id = document_context

    with pytest.raises(HTTPException) as exc_info:
        documents_phase2.export_document(
            live_id,
            db,
            "docx",
            "blue_modern",
        )

    assert exc_info.value.status_code == 400
    assert "Unsupported preset 'blue_modern' for format 'docx'" in str(
        exc_info.value.detail
    )


def test_docx_modern_uses_source_owner_as_author(document_context) -> None:
    _, db, source_id, live_id = document_context
    owner = User(
        provider="google",
        provider_id="export-owner",
        email="owner@example.com",
        name="Anh Tú",
    )
    db.add(owner)
    db.flush()
    source = db.get(Source, source_id)
    assert source is not None
    source.user_id = owner.id
    db.add(source)
    db.commit()

    artifact = DocumentService(db).export_document(
        live_id,
        "docx",
        "modern",
    )
    with ZipFile(BytesIO(artifact.content)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    assert "Anh Tú" in document_xml


def test_document_service_rehydrates_canonical_transcript(
    document_context,
) -> None:
    _, db, source_id, _ = document_context

    transcript = DocumentService(db).build_transcript(source_id)

    assert transcript.key == "meeting.wav"
    assert transcript.text == "Nội dung cuộc họp."
    assert transcript.duration == 3.0
    assert transcript.sentence_info[0].start == 0.0
    assert transcript.sentence_info[0].end == 3.0

    markdown = DocumentService.full_text_markdown(transcript)
    assert markdown.startswith("# Toàn văn cuộc họp\n")
    assert "_Nguồn: meeting_" in markdown
    assert "Toàn văn: meeting.wav" not in markdown
