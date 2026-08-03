"""Tests for realtime post-session transcript events and persistence."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from meetasr.db.models_phase2 import (
    Document,
    DocumentMode,
    Job,
    JobStage,
    JobStatus,
    MediaType,
    Source,
    TranscriptSegment,
)
from meetasr.schemas import (
    SentenceInfo,
    SpeakerTurn,
    TargetedSegmentPlan,
    TranscriptResult,
)
from meetasr.streaming import final_transcript_worker
from meetasr.streaming.coverage import RealtimeCoverageTracker
from meetasr.streaming.final_transcript_queue import FinalTranscriptJob


async def _run_fake_inference(
    function: Callable[..., Any],
    *args: Any,
) -> Any:
    """Execute a fake inference function without creating test worker threads."""
    return function(*args)


def _create_job(engine: Engine) -> tuple[str, str]:
    """Create one processing realtime Source and return its identifiers."""
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
        db.refresh(source)
        db.refresh(job)
        return source.id, job.id


@pytest.mark.asyncio
async def test_final_worker_persists_before_publishing_terminal_events(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(final_transcript_worker, "engine", engine)
    monkeypatch.setattr(
        final_transcript_worker.asyncio,
        "to_thread",
        _run_fake_inference,
    )
    source_id, job_id = _create_job(engine)

    class FakePipeline:
        def transcribe(self, audio: np.ndarray) -> TranscriptResult:
            assert audio.dtype == np.float32
            return TranscriptResult(
                key="realtime",
                text="Xin chào. Tôi đồng ý.",
                duration=3.0,
                sentence_info=[
                    SentenceInfo(
                        text="Xin chào.",
                        start=0.0,
                        end=1.0,
                        speaker=0,
                    ),
                    SentenceInfo(
                        text="Tôi đồng ý.",
                        start=1.5,
                        end=3.0,
                        speaker=1,
                    ),
                ],
            )

    published: list[dict] = []

    class FakeEventBus:
        async def publish(self, event_job_id: str, event: Any) -> None:
            assert event_job_id == job_id
            payload = event.model_dump(mode="json")
            with Session(engine) as db:
                if payload["type"] == "transcript_delta":
                    segment_id = payload["segment"]["id"]
                    assert segment_id is not None
                    assert db.get(TranscriptSegment, segment_id) is not None
                elif payload["type"] == "doc_delta":
                    document = db.exec(
                        select(Document)
                        .where(Document.source_id == source_id)
                        .where(Document.mode == DocumentMode.LIVE)
                    ).one()
                    assert document.markdown == payload["markdown"]
                elif payload["type"] == "done":
                    stored_job = db.get(Job, job_id)
                    assert stored_job is not None
                    assert stored_job.status == JobStatus.DONE
            published.append(payload)

    monkeypatch.setattr(final_transcript_worker, "event_bus", FakeEventBus())
    worker = final_transcript_worker.FinalTranscriptWorker(
        queue=SimpleNamespace(),
        pipeline=FakePipeline(),
    )

    await worker._process(
        FinalTranscriptJob(
            job_id=job_id,
            audio=np.zeros(3 * 16000, dtype=np.float32),
        )
    )

    with Session(engine) as db:
        source = db.get(Source, source_id)
        job = db.get(Job, job_id)
        segments = db.exec(
            select(TranscriptSegment)
            .where(TranscriptSegment.job_id == job_id)
            .order_by(TranscriptSegment.start_ms)
        ).all()
        document = db.exec(
            select(Document)
            .where(Document.source_id == source_id)
            .where(Document.mode == DocumentMode.LIVE)
        ).one()

        assert source is not None
        assert source.duration == 3.0
        assert job is not None
        assert job.status == JobStatus.DONE
        assert job.stage == JobStage.GENERATING_DOC
        assert job.progress == 1.0
        assert job.error is None
        assert [segment.text for segment in segments] == [
            "Xin chào.",
            "Tôi đồng ý.",
        ]
        assert document.markdown
        live_document_id = document.id

    assert [event["type"] for event in published] == [
        "status",
        "transcript_delta",
        "transcript_delta",
        "status",
        "doc_delta",
        "done",
    ]
    assert published[-1] == {
        "type": "done",
        "duration_ms": 3000,
        "num_segments": 2,
        "live_document_id": live_document_id,
    }
    engine.dispose()


@pytest.mark.asyncio
async def test_final_worker_persists_and_publishes_failure(monkeypatch) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(final_transcript_worker, "engine", engine)
    monkeypatch.setattr(
        final_transcript_worker.asyncio,
        "to_thread",
        _run_fake_inference,
    )
    _, job_id = _create_job(engine)

    class FailingPipeline:
        def transcribe(self, audio: np.ndarray) -> TranscriptResult:
            raise RuntimeError("ASR unavailable")

    published: list[dict] = []

    class FakeEventBus:
        async def publish(self, event_job_id: str, event: Any) -> None:
            assert event_job_id == job_id
            payload = event.model_dump(mode="json")
            if payload["type"] == "error":
                with Session(engine) as db:
                    stored_job = db.get(Job, job_id)
                    assert stored_job is not None
                    assert stored_job.status == JobStatus.FAILED
                    assert stored_job.error == "ASR unavailable"
            published.append(payload)

    monkeypatch.setattr(final_transcript_worker, "event_bus", FakeEventBus())
    worker = final_transcript_worker.FinalTranscriptWorker(
        queue=SimpleNamespace(),
        pipeline=FailingPipeline(),
    )

    await worker._process(
        FinalTranscriptJob(
            job_id=job_id,
            audio=np.zeros(16000, dtype=np.float32),
        )
    )

    with Session(engine) as db:
        job = db.get(Job, job_id)
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error == "ASR unavailable"

    assert [event["type"] for event in published] == ["status", "error"]
    assert published[-1] == {
        "type": "error",
        "code": "final_transcript_failed",
        "message": "ASR unavailable",
    }
    engine.dispose()


@pytest.mark.asyncio
async def test_complete_coverage_retranscribes_only_mixed_segment(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(final_transcript_worker, "engine", engine)
    monkeypatch.setattr(
        final_transcript_worker.asyncio,
        "to_thread",
        _run_fake_inference,
    )
    source_id, job_id = _create_job(engine)
    with Session(engine) as db:
        records = [
            TranscriptSegment(
                job_id=job_id,
                start_ms=0,
                end_ms=1200,
                text="Câu đã chốt.",
            ),
            TranscriptSegment(
                job_id=job_id,
                start_ms=1800,
                end_ms=3000,
                text="Đoạn có lượt nói phụ.",
            ),
        ]
        db.add_all(records)
        db.commit()
        for record in records:
            db.refresh(record)
        stable_ids = [record.id for record in records]

    targeted_calls: list[tuple[int, int]] = []

    class FakePipeline:
        def transcribe(self, audio: np.ndarray) -> TranscriptResult:
            pytest.fail("complete coverage must not run full ASR")

        def prepare_realtime_targeted_retranscription(
            self,
            audio,
            sentences,
            vad_segments,
        ):
            assert [sentence.text for sentence in sentences] == [
                "Câu đã chốt.",
                "Đoạn có lượt nói phụ.",
            ]
            assert [(item.start_ms, item.end_ms) for item in vad_segments] == [
                (0, 1200),
                (1800, 3000),
            ]
            return [
                TargetedSegmentPlan(0, "keep", "single", 4),
                TargetedSegmentPlan(
                    1,
                    "retranscribe",
                    "mixed",
                    None,
                    (
                        SpeakerTurn(1800, 2300, 4),
                        SpeakerTurn(2300, 3000, 9),
                    ),
                ),
            ]

        def transcribe_vad_segment(
            self,
            audio,
            segment,
            language="auto",
        ):
            targeted_calls.append((segment.start_ms, segment.end_ms))
            text = "Phần của người một."
            if segment.start_ms != 1800:
                text = "Phần của người hai."
            return [
                SentenceInfo(
                    text=text,
                    start=segment.start_ms / 1000,
                    end=segment.end_ms / 1000,
                )
            ]

    published: list[dict] = []

    class FakeEventBus:
        async def publish(self, event_job_id: str, event: Any) -> None:
            assert event_job_id == job_id
            payload = event.model_dump(mode="json")
            if payload["type"] == "speaker_update":
                with Session(engine) as db:
                    assert db.get(TranscriptSegment, stable_ids[0]).speaker == 0
                    expected = {"segment_id": stable_ids[0], "speaker": 0}
                    assert payload["updates"] == [expected]
            elif payload["type"] == "doc_delta":
                assert "Phần của người một." in payload["markdown"]
                assert "Đoạn có lượt nói phụ." not in payload["markdown"]
            published.append(payload)

    monkeypatch.setattr(final_transcript_worker, "event_bus", FakeEventBus())
    coverage = RealtimeCoverageTracker()
    for start_ms, end_ms in [(0, 1200), (1800, 3000)]:
        coverage.record_speech(start_ms, end_ms)
        coverage.record_confirmed(start_ms, end_ms)
    coverage.mark_flush_completed()
    worker = final_transcript_worker.FinalTranscriptWorker(
        queue=SimpleNamespace(),
        pipeline=FakePipeline(),
    )

    await worker._process(
        FinalTranscriptJob(
            job_id=job_id,
            audio=np.zeros(3 * 16000, dtype=np.float32),
            coverage=coverage.snapshot(),
        )
    )

    with Session(engine) as db:
        records = db.exec(
            select(TranscriptSegment)
            .where(TranscriptSegment.job_id == job_id)
            .order_by(TranscriptSegment.start_ms)
        ).all()
        assert records[0].id == stable_ids[0]
        assert [record.text for record in records] == [
            "Câu đã chốt.",
            "Phần của người một.",
            "Phần của người hai.",
        ]
        assert [record.speaker for record in records] == [0, 0, 1]
        assert db.get(Source, source_id).duration == 3.0

    assert targeted_calls == [(1800, 2300), (2300, 3000)]
    assert [event["type"] for event in published] == [
        "status",
        "speaker_update",
        "status",
        "doc_delta",
        "done",
    ]
    assert not any(event["type"] == "transcript_delta" for event in published)
    assert published[-1]["num_segments"] == 3
    engine.dispose()


@pytest.mark.asyncio
async def test_incomplete_coverage_runs_full_fallback_and_keeps_live_segment(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(final_transcript_worker, "engine", engine)
    monkeypatch.setattr(
        final_transcript_worker.asyncio,
        "to_thread",
        _run_fake_inference,
    )
    _, job_id = _create_job(engine)
    with Session(engine) as db:
        live = TranscriptSegment(
            job_id=job_id,
            start_ms=0,
            end_ms=1000,
            text="Câu realtime giữ nguyên.",
        )
        db.add(live)
        db.commit()
        db.refresh(live)
        live_id = live.id

    calls = 0

    class FakePipeline:
        def transcribe(self, audio: np.ndarray) -> TranscriptResult:
            nonlocal calls
            calls += 1
            return TranscriptResult(
                key="fallback",
                text="Câu realtime giữ nguyên. Câu bị thiếu.",
                duration=3.0,
                sentence_info=[
                    SentenceInfo(
                        text="Bản offline không được ghi đè câu đã chốt.",
                        start=0.0,
                        end=1.0,
                        speaker=0,
                    ),
                    SentenceInfo(
                        text="Câu bị thiếu.",
                        start=1.5,
                        end=3.0,
                        speaker=1,
                    ),
                ],
            )

    published: list[dict] = []

    class FakeEventBus:
        async def publish(self, event_job_id: str, event: Any) -> None:
            published.append(event.model_dump(mode="json"))

    monkeypatch.setattr(final_transcript_worker, "event_bus", FakeEventBus())
    coverage = RealtimeCoverageTracker()
    coverage.record_speech(0, 1000)
    coverage.record_speech(1500, 3000)
    coverage.record_confirmed(0, 1000)
    coverage.record_asr_failure()
    coverage.mark_flush_completed()
    worker = final_transcript_worker.FinalTranscriptWorker(
        queue=SimpleNamespace(),
        pipeline=FakePipeline(),
    )

    await worker._process(
        FinalTranscriptJob(
            job_id=job_id,
            audio=np.zeros(3 * 16000, dtype=np.float32),
            coverage=coverage.snapshot(),
        )
    )

    with Session(engine) as db:
        records = db.exec(
            select(TranscriptSegment)
            .where(TranscriptSegment.job_id == job_id)
            .order_by(TranscriptSegment.start_ms)
        ).all()
        assert calls == 1
        assert records[0].id == live_id
        assert records[0].text == "Câu realtime giữ nguyên."
        assert [record.text for record in records] == [
            "Câu realtime giữ nguyên.",
            "Câu bị thiếu.",
        ]

    assert sum(event["type"] == "transcript_delta" for event in published) == 1
    engine.dispose()
