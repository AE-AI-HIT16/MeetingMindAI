"""Replay tests for the Phase 2 Job WebSocket contract."""

import asyncio

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from meetasr.api.routes import document_jobs, jobs
from meetasr.db.models_phase2 import (
    Document,
    DocumentGenerationJob,
    DocumentGenerationStage,
    DocumentGenerationStatus,
    DocumentMode,
    Job,
    JobStage,
    JobStatus,
    MediaType,
    Source,
    TranscriptSegment,
)


class _FakeEventBus:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.unsubscribed = False

    async def subscribe(self, job_id: str) -> asyncio.Queue[dict]:
        return self.queue

    async def unsubscribe(
        self,
        job_id: str,
        queue: asyncio.Queue[dict],
    ) -> None:
        assert queue is self.queue
        self.unsubscribed = True


class _FakeWebSocket:
    def __init__(
        self,
        receive_message: dict | None = None,
        *,
        block_close: bool = False,
    ) -> None:
        self.receive_message = receive_message
        self.block_close = block_close
        self.accepted = False
        self.sent: list[dict] = []
        self.close_codes: list[int] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, event: dict) -> None:
        self.sent.append(event)

    async def receive(self) -> dict:
        if self.receive_message is None:
            await asyncio.Future()
        return self.receive_message

    async def close(self, code: int = 1000) -> None:
        self.close_codes.append(code)
        if self.block_close:
            await asyncio.Future()


def test_done_job_snapshot_replays_persisted_state(monkeypatch) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(jobs, "engine", engine)

    with Session(engine) as db:
        source = Source(
            filename="meeting.wav",
            media_type=MediaType.AUDIO,
            duration=2.0,
            storage_path="meeting.wav",
        )
        db.add(source)
        db.flush()
        job = Job(
            source_id=source.id,
            status=JobStatus.DONE,
            stage=JobStage.GENERATING_DOC,
            progress=1.0,
        )
        db.add(job)
        db.flush()
        db.add(
            TranscriptSegment(
                job_id=job.id,
                start_ms=0,
                end_ms=2000,
                speaker=0,
                text="Xin chào.",
            )
        )
        live = Document(
            source_id=source.id,
            mode=DocumentMode.LIVE,
            markdown="# Live",
        )
        db.add(live)
        db.commit()
        db.refresh(job)
        db.refresh(live)
        job_id = job.id
        live_id = live.id

    snapshot = jobs._job_snapshot(job_id)

    assert [event["type"] for event in snapshot] == [
        "status",
        "transcript_delta",
        "doc_delta",
        "done",
    ]
    assert snapshot[1]["segment"]["text"] == "Xin chào."
    assert snapshot[-1] == {
        "type": "done",
        "duration_ms": 2000,
        "num_segments": 1,
        "live_document_id": live_id,
    }
    engine.dispose()


def test_missing_job_snapshot_returns_terminal_error(monkeypatch) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(jobs, "engine", engine)

    assert jobs._job_snapshot("missing") == [
        {
            "type": "error",
            "code": "job_not_found",
            "message": "Job 'missing' không tồn tại.",
        }
    ]
    engine.dispose()


@pytest.mark.asyncio
async def test_done_job_websocket_closes_after_terminal_snapshot(
    monkeypatch,
) -> None:
    bus = _FakeEventBus()
    websocket = _FakeWebSocket()
    done = {
        "type": "done",
        "duration_ms": 1600,
        "num_segments": 0,
        "live_document_id": "document-1",
    }
    monkeypatch.setattr(jobs, "event_bus", bus)
    monkeypatch.setattr(jobs, "_job_snapshot", lambda job_id: [done])

    await asyncio.wait_for(
        jobs.job_events(websocket, "job-1"),
        timeout=0.1,
    )

    assert websocket.accepted is True
    assert websocket.sent == [done]
    assert websocket.close_codes == [1000]
    assert bus.unsubscribed is True


@pytest.mark.asyncio
async def test_done_job_websocket_does_not_wait_forever_for_close_handshake(
    monkeypatch,
) -> None:
    bus = _FakeEventBus()
    websocket = _FakeWebSocket(block_close=True)
    done = {
        "type": "done",
        "duration_ms": 1600,
        "num_segments": 0,
        "live_document_id": "document-1",
    }
    monkeypatch.setattr(jobs, "event_bus", bus)
    monkeypatch.setattr(jobs, "_job_snapshot", lambda job_id: [done])

    await asyncio.wait_for(
        jobs.job_events(websocket, "job-1"),
        timeout=0.5,
    )

    assert websocket.close_codes == [1000]
    assert bus.unsubscribed is True


@pytest.mark.asyncio
async def test_processing_job_websocket_observes_client_disconnect(
    monkeypatch,
) -> None:
    bus = _FakeEventBus()
    websocket = _FakeWebSocket(
        {"type": "websocket.disconnect", "code": 1001}
    )
    status = {"type": "status", "stage": "transcribing", "progress": 0.2}
    monkeypatch.setattr(jobs, "event_bus", bus)
    monkeypatch.setattr(jobs, "_job_snapshot", lambda job_id: [status])

    await asyncio.wait_for(
        jobs.job_events(websocket, "job-1"),
        timeout=0.1,
    )

    assert websocket.sent == [status]
    assert websocket.close_codes == []
    assert bus.unsubscribed is True


@pytest.mark.asyncio
async def test_processing_job_websocket_closes_after_live_terminal_event(
    monkeypatch,
) -> None:
    bus = _FakeEventBus()
    websocket = _FakeWebSocket()
    status = {"type": "status", "stage": "transcribing", "progress": 0.2}
    done = {
        "type": "done",
        "duration_ms": 1600,
        "num_segments": 0,
        "live_document_id": "document-1",
    }
    bus.queue.put_nowait(done)
    monkeypatch.setattr(jobs, "event_bus", bus)
    monkeypatch.setattr(jobs, "_job_snapshot", lambda job_id: [status])

    await asyncio.wait_for(
        jobs.job_events(websocket, "job-1"),
        timeout=0.1,
    )

    assert websocket.sent == [status, done]
    assert websocket.close_codes == [1000]
    assert bus.unsubscribed is True


def test_document_generation_snapshot_replays_processing_and_done(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(document_jobs, "engine", engine)

    with Session(engine) as db:
        source = Source(
            filename="meeting.wav",
            media_type=MediaType.AUDIO,
            storage_path="meeting.wav",
        )
        db.add(source)
        db.flush()
        document = Document(
            source_id=source.id,
            mode=DocumentMode.SUMMARY,
        )
        db.add(document)
        db.flush()
        generation = DocumentGenerationJob(
            document_id=document.id,
            source_id=source.id,
            mode=DocumentMode.SUMMARY,
            status=DocumentGenerationStatus.PROCESSING,
            stage=DocumentGenerationStage.GENERATING,
            progress=0.4,
        )
        db.add(generation)
        db.commit()
        db.refresh(generation)
        generation_id = generation.id
        document_id = document.id

    processing = document_jobs._generation_snapshot(generation_id)
    assert processing == [
        {
            "type": "document_status",
            "generation_job_id": generation_id,
            "document_id": document_id,
            "stage": "generating",
            "progress": 0.4,
        }
    ]

    with Session(engine) as db:
        generation = db.get(DocumentGenerationJob, generation_id)
        assert generation is not None
        generation.status = DocumentGenerationStatus.DONE
        generation.stage = DocumentGenerationStage.DONE
        generation.progress = 1.0
        db.add(generation)
        db.commit()

    done = document_jobs._generation_snapshot(generation_id)
    assert done == [
        {
            "type": "document_done",
            "generation_job_id": generation_id,
            "document_id": document_id,
            "mode": DocumentMode.SUMMARY,
        }
    ]
    engine.dispose()
