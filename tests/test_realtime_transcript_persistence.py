"""Persistence-first tests for confirmed realtime transcript deltas."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from meetasr.api.routes import jobs
from meetasr.api.schemas_phase2 import TranscriptSegmentPayload
from meetasr.db.models_phase2 import (
    Job,
    JobStage,
    JobStatus,
    MediaType,
    Source,
    TranscriptSegment,
)
from meetasr.services import realtime_transcript_service
from meetasr.streaming import temp_asr_woker
from meetasr.streaming.temp_asr_woker import PartialASRRequest, TempASRWorker


async def _run_inline(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    return function(*args, **kwargs)


def _engine_with_job() -> tuple[Engine, str, str]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        source = Source(
            filename="realtime.wav",
            media_type=MediaType.AUDIO,
            storage_path="",
        )
        db.add(source)
        db.flush()
        job = Job(source_id=source.id, status=JobStatus.PROCESSING)
        db.add(job)
        db.commit()
        return engine, source.id, job.id


@pytest.mark.asyncio
async def test_confirmed_delta_is_committed_before_both_publications(
    monkeypatch,
) -> None:
    engine, source_id, job_id = _engine_with_job()
    monkeypatch.setattr(realtime_transcript_service, "engine", engine)
    monkeypatch.setattr(
        realtime_transcript_service.asyncio,
        "to_thread",
        _run_inline,
    )
    publication_order: list[str] = []

    def assert_committed(payload: dict) -> None:
        segment_id = payload["segment"]["id"]
        assert segment_id is not None
        with Session(engine) as db:
            assert db.get(TranscriptSegment, segment_id) is not None

    class FakeEventBus:
        async def publish(self, event_job_id: str, event: Any) -> None:
            assert event_job_id == job_id
            payload = event.model_dump(mode="json")
            assert_committed(payload)
            publication_order.append("job_bus")

    class FakeWebSocket:
        async def send_json(self, payload: dict) -> None:
            assert_committed(payload)
            publication_order.append("recording_ws")

    monkeypatch.setattr(
        realtime_transcript_service,
        "event_bus",
        FakeEventBus(),
    )
    service = realtime_transcript_service.RealtimeTranscriptService()
    persisted = await service.persist_and_publish(
        job_id,
        TranscriptSegmentPayload(
            start_ms=1000,
            end_ms=2600,
            speaker=None,
            text="Xin chào.",
        ),
        FakeWebSocket(),
    )

    assert persisted.id is not None
    assert publication_order == ["job_bus", "recording_ws"]
    with Session(engine) as db:
        source = db.get(Source, source_id)
        job = db.get(Job, job_id)
        assert source is not None and source.duration == 2.6
        assert job is not None and job.stage == JobStage.TRANSCRIBING
    engine.dispose()


@pytest.mark.asyncio
async def test_retry_reuses_stable_id_without_replacing_confirmed_text(
    monkeypatch,
) -> None:
    engine, _, job_id = _engine_with_job()
    monkeypatch.setattr(realtime_transcript_service, "engine", engine)
    monkeypatch.setattr(
        realtime_transcript_service.asyncio,
        "to_thread",
        _run_inline,
    )

    class FakeEventBus:
        async def publish(self, job_id: str, event: Any) -> None:
            return None

    class FakeWebSocket:
        async def send_json(self, payload: dict) -> None:
            return None

    monkeypatch.setattr(
        realtime_transcript_service,
        "event_bus",
        FakeEventBus(),
    )
    service = realtime_transcript_service.RealtimeTranscriptService()
    original = await service.persist_and_publish(
        job_id,
        TranscriptSegmentPayload(
            start_ms=0,
            end_ms=1200,
            text="Nội dung đã chốt.",
        ),
        FakeWebSocket(),
    )
    retried = await service.persist_and_publish(
        job_id,
        TranscriptSegmentPayload(
            start_ms=0,
            end_ms=1200,
            text="Kết quả retry khác.",
        ),
        FakeWebSocket(),
    )

    assert retried.id == original.id
    assert retried.text == "Nội dung đã chốt."
    with Session(engine) as db:
        records = db.exec(
            select(TranscriptSegment).where(
                TranscriptSegment.job_id == job_id
            )
        ).all()
        assert len(records) == 1
    engine.dispose()


@pytest.mark.asyncio
async def test_persisted_realtime_delta_is_replayed_from_job_snapshot(
    monkeypatch,
) -> None:
    engine, _, job_id = _engine_with_job()
    monkeypatch.setattr(realtime_transcript_service, "engine", engine)
    monkeypatch.setattr(jobs, "engine", engine)
    monkeypatch.setattr(
        realtime_transcript_service.asyncio,
        "to_thread",
        _run_inline,
    )

    class FakeEventBus:
        async def publish(self, job_id: str, event: Any) -> None:
            return None

    class FakeWebSocket:
        async def send_json(self, payload: dict) -> None:
            return None

    monkeypatch.setattr(
        realtime_transcript_service,
        "event_bus",
        FakeEventBus(),
    )
    await realtime_transcript_service.RealtimeTranscriptService().persist_and_publish(
        job_id,
        TranscriptSegmentPayload(
            start_ms=500,
            end_ms=1900,
            text="Có thể phát lại.",
        ),
        FakeWebSocket(),
    )

    snapshot = jobs._job_snapshot(job_id)

    delta = next(event for event in snapshot if event["type"] == "transcript_delta")
    assert delta["segment"]["id"] is not None
    assert delta["segment"]["text"] == "Có thể phát lại."
    engine.dispose()


@pytest.mark.asyncio
async def test_partial_preview_is_never_persisted(monkeypatch) -> None:
    engine, _, job_id = _engine_with_job()

    class FakeASR:
        def recognize(self, audio, *, language):
            return [{"text": "Đây chỉ là bản xem trước."}]

    class FakeWebSocket:
        async def send_json(self, payload: dict) -> None:
            assert payload["type"] == "transcript_partial"

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(temp_asr_woker.asyncio, "to_thread", run_inline)
    session = SimpleNamespace(
        job_id=job_id,
        partial_buffer=np.ones(16000, dtype=np.float32),
        partial_buffer_start_ms=0,
        partial_buffer_lock=asyncio.Lock(),
        websocket=FakeWebSocket(),
    )
    worker = TempASRWorker(
        session,
        SimpleNamespace(asr=FakeASR(), transcription_language="vi"),
    )

    await worker.process_request(PartialASRRequest(start_ms=0, end_ms=1000))

    with Session(engine) as db:
        records = db.exec(
            select(TranscriptSegment).where(
                TranscriptSegment.job_id == job_id
            )
        ).all()
        assert records == []
    engine.dispose()
