"""Vertical-slice test for the complete uploaded-media user journey."""

from __future__ import annotations

import io
import wave

import pytest
from fastapi import UploadFile
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select
from starlette.datastructures import Headers

from meetasr.api.routes import jobs, sources
from meetasr.db.models_phase2 import (
    Document,
    DocumentMode,
    Job,
    JobStatus,
    TranscriptSegment,
)
from meetasr.pipeline import MeetPipeline
from meetasr.realtime import job_worker
from meetasr.realtime.events import EventBus
from meetasr.schemas_doc import DocSection, DocumentReport
from meetasr.services import asr_service, document_service
from meetasr.services.asr_service import ASRService
from meetasr.services.document_service import DocumentService
from meetasr.storage.local import LocalStorage
from types import SimpleNamespace


def _short_wav() -> bytes:
    """Return one second of valid mono 16 kHz PCM WAV audio."""
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 16_000)
    return payload.getvalue()


@pytest.mark.asyncio
async def test_upload_to_pdf_and_reopen_from_library(
    tmp_path,
    monkeypatch,
) -> None:
    """Upload → worker/WS → summary → PDF → persisted Library document."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)
    monkeypatch.setattr(job_worker, "engine", test_engine)
    monkeypatch.setattr(jobs, "engine", test_engine)

    # Keep this integration test deterministic and avoid the environment's
    # background-thread shutdown issue while still exercising real services.
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asr_service.asyncio, "to_thread", run_inline)
    monkeypatch.setattr(document_service.asyncio, "to_thread", run_inline)

    storage = LocalStorage(str(tmp_path / "media"))
    enqueued_job_ids: list[str] = []

    class CapturingQueue:
        async def enqueue(self, job_id: str) -> bool:
            enqueued_job_ids.append(job_id)
            return True

    monkeypatch.setattr(sources, "job_queue", CapturingQueue())

    upload = UploadFile(
        io.BytesIO(_short_wav()),
        filename="vertical-slice.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )
    fake_user = SimpleNamespace(
        id="test-user-id",
    )

    with Session(test_engine) as db:
        created = await sources.create_source(
            upload,
            db,
            storage,
            fake_user,
        )

    assert created.status == JobStatus.QUEUED
    assert enqueued_job_ids == [created.jobId]
    with Session(test_engine) as db:
        queued_job = db.get(Job, created.jobId)
        assert queued_job is not None
        assert queued_job.source_id == created.sourceId
        assert queued_job.status == JobStatus.QUEUED

    class StubASR:
        """Only the expensive model is stubbed; audio decoding/pipeline are real."""

        def recognize(self, chunks, **kwargs):
            return [
                {
                    "text": "Hôm nay chúng ta thống nhất hoàn thiện tích hợp.",
                    "timestamp": [],
                }
                for _ in chunks
            ]

    bus = EventBus(max_queue_size=50)

    monkeypatch.setattr(
        job_worker,
        "event_bus",
        bus,
    )

    fake_queue = CapturingQueue()

    monkeypatch.setattr(
        sources,
        "job_queue",
        fake_queue,
    )
    monkeypatch.setattr(
        job_worker,
        "job_queue",
        fake_queue,
    )
    subscriber = await bus.subscribe(created.jobId)
    worker = job_worker.JobQueue()
    worker._storage = storage
    worker._asr_service = ASRService(MeetPipeline(asr_model=StubASR()))

    try:
        await worker._process(created.jobId)
        live_events = []
        while not subscriber.empty():
            live_events.append(subscriber.get_nowait())
    finally:
        await bus.unsubscribe(created.jobId, subscriber)

    event_types = [event["type"] for event in live_events]
    assert event_types[0] == "status"
    assert "transcript_delta" in event_types
    assert event_types[-1] == "done"
    assert any(
        event["type"] == "status" and event["stage"] == "transcribing"
        for event in live_events
    )

    with Session(test_engine) as db:
        completed_job = db.get(Job, created.jobId)
        segments = db.exec(
            select(TranscriptSegment).where(
                TranscriptSegment.job_id == created.jobId
            )
        ).all()
        live_document = db.exec(
            select(Document)
            .where(Document.source_id == created.sourceId)
            .where(Document.mode == DocumentMode.LIVE)
        ).one()

        assert completed_job is not None
        assert completed_job.status == JobStatus.DONE
        assert len(segments) == 1
        assert segments[0].text.startswith("Hôm nay")
        assert segments[0].id is not None

    # A reconnect sees the same persisted transcript and terminal state before
    # listening for newer events.
    replay = jobs._job_snapshot(created.jobId)
    assert any(event["type"] == "transcript_delta" for event in replay)
    assert replay[-1]["type"] == "done"
    assert replay[-1]["live_document_id"] == live_document.id

    class StubPlanner:
        def plan_and_write(self, transcript):
            assert "hoàn thiện tích hợp" in transcript.text
            return DocumentReport(
                content_kind="Biên bản cuộc họp",
                sections=[
                    DocSection(
                        id="summary",
                        heading="Tóm tắt",
                        kind="summary",
                        markdown="Nhóm thống nhất hoàn thiện luồng tích hợp.",
                    )
                ],
            )

    with Session(test_engine) as db:
        service = DocumentService(db, planner=StubPlanner())
        summary = await service.finalize_document(
            live_document.id,
            DocumentMode.SUMMARY,
        )
        summary_id = summary.id
        artifact = service.export_document(summary_id, "pdf")

        assert artifact.media_type == "application/pdf"
        assert artifact.filename.endswith(".pdf")
        assert artifact.content.startswith(b"%PDF-")

    # This is the backend data used by listSources()/getSource(): it now carries
    # the real Document ID, so SourceCard can reopen the finalized document.
    with Session(test_engine) as db:
        library = sources.list_sources(
            db,
            fake_user,
        )
        detail = sources.get_source(created.sourceId, db, fake_user)
        summary_ref = next(
            document
            for document in detail.documents
            if document.mode == DocumentMode.SUMMARY
        )

        assert [source.id for source in library] == [created.sourceId]
        assert summary_ref.id == summary_id
        reopened = DocumentService(db).get_document(summary_ref.id)
        assert reopened.markdown.startswith("# Biên bản cuộc họp")

    test_engine.dispose()
