"""Small end-to-end unit test for the uploaded-media worker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from meetasr.db.models_phase2 import (
    Document,
    DocumentMode,
    Job,
    JobStatus,
    MediaType,
    Source,
    TranscriptSegment,
)
from meetasr.realtime import job_worker
from meetasr.schemas import Segment, SentenceInfo, SpeakerTurn
from meetasr.services.asr_service import PreparedTranscription


@pytest.mark.asyncio
async def test_upload_worker_persists_segments_document_and_done_event(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(job_worker, "engine", engine)

    with Session(engine) as db:
        source = Source(
            filename="meeting.wav",
            media_type=MediaType.AUDIO,
            storage_path="uploads/meeting.wav",
        )
        db.add(source)
        db.flush()
        job = Job(source_id=source.id, status=JobStatus.QUEUED)
        db.add(job)
        db.commit()
        db.refresh(source)
        db.refresh(job)
        source_id = source.id
        job_id = job.id

    class FakeStorage:
        def abs_path(self, key: str) -> Path:
            assert key == "uploads/meeting.wav"
            return Path("/tmp/meeting.wav")

    class FakeASRService:
        async def prepare_incremental(self, audio_source):
            assert Path(audio_source) == Path("/tmp/meeting.wav")
            return PreparedTranscription(
                audio=np.zeros(5 * 16000, dtype=np.float32),
                vad_segments=[
                    Segment(0, 2000),
                    Segment(3000, 5000),
                ],
                duration_ms=5000,
            )

        async def transcribe_segment(self, prepared, segment, *, key=None):
            assert key is not None and key.startswith("meeting.wav:chunk-")
            if segment.start_ms == 0:
                return [
                    SentenceInfo(
                        text="Nội dung",
                        start=0.0,
                        end=2.0,
                    )
                ]
            return [
                SentenceInfo(
                    text="cuộc họp",
                    start=3.0,
                    end=5.0,
                )
            ]

        async def finalize_incremental(self, prepared, sentences):
            return [
                SentenceInfo(
                    text="Nội dung.",
                    start=0.0,
                    end=2.0,
                    speaker=0,
                ),
                SentenceInfo(
                    text="Cuộc họp.",
                    start=3.0,
                    end=5.0,
                    speaker=1,
                ),
            ]

    published: list[dict] = []

    class FakeEventBus:
        async def publish(self, event_job_id, event):
            assert event_job_id == job_id
            payload = event.model_dump(mode="json")
            if payload["type"] == "transcript_delta":
                segment_id = payload["segment"]["id"]
                assert segment_id is not None
                with Session(engine) as event_db:
                    assert event_db.get(TranscriptSegment, segment_id) is not None
            if payload["type"] == "speaker_update":
                with Session(engine) as event_db:
                    for update in payload["updates"]:
                        record = event_db.get(
                            TranscriptSegment,
                            update["segment_id"],
                        )
                        assert record is not None
                        assert record.speaker == update["speaker"]
            published.append(payload)

    monkeypatch.setattr(job_worker, "event_bus", FakeEventBus())
    queue = job_worker.JobQueue()
    queue._storage = FakeStorage()
    queue._asr_service = FakeASRService()

    await queue._process(job_id)

    with Session(engine) as db:
        stored_job = db.get(Job, job_id)
        stored_source = db.get(Source, source_id)
        segments = db.exec(
            select(TranscriptSegment).where(TranscriptSegment.job_id == job_id)
        ).all()
        live_document = db.exec(
            select(Document)
            .where(Document.source_id == source_id)
            .where(Document.mode == DocumentMode.LIVE)
        ).one()

        assert stored_job is not None
        assert stored_job.status == JobStatus.DONE
        assert stored_job.progress == 1.0
        assert stored_source is not None
        assert stored_source.duration == 5.0
        assert [segment.text for segment in segments] == [
            "Nội dung.",
            "Cuộc họp.",
        ]
        assert [segment.speaker for segment in segments] == [0, 1]
        assert "Nội dung." in live_document.markdown
        assert "Cuộc họp." in live_document.markdown

    event_types = [event["type"] for event in published]
    assert event_types[:2] == ["status", "status"]
    assert event_types.count("transcript_delta") == 4
    assert event_types.count("speaker_update") == 1
    assert event_types[-3:] == ["status", "doc_delta", "done"]
    provisional_events = [
        event
        for event in published
        if event["type"] == "transcript_delta"
    ][:2]
    assert [
        event["segment"]["speaker"] for event in provisional_events
    ] == [None, None]
    speaker_event = next(
        event for event in published if event["type"] == "speaker_update"
    )
    assert [update["speaker"] for update in speaker_event["updates"]] == [0, 1]
    assert published[-1]["live_document_id"] == live_document.id
    engine.dispose()


@pytest.mark.asyncio
async def test_upload_worker_persists_diarization_first_speakers_immediately(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(job_worker, "engine", engine)

    with Session(engine) as db:
        source = Source(
            filename="meeting.wav",
            media_type=MediaType.AUDIO,
            storage_path="uploads/meeting.wav",
        )
        db.add(source)
        db.flush()
        job = Job(source_id=source.id, status=JobStatus.QUEUED)
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    class FakeStorage:
        def abs_path(self, key: str) -> Path:
            return Path("/tmp/meeting.wav")

    class FakeASRService:
        async def prepare_incremental(self, audio_source):
            return PreparedTranscription(
                audio=np.zeros(4 * 16000, dtype=np.float32),
                vad_segments=[Segment(0, 4000)],
                duration_ms=4000,
                speaker_turns=[
                    SpeakerTurn(0, 2000, 0),
                    SpeakerTurn(2000, 4000, 1),
                ],
            )

        async def transcribe_segment(self, prepared, segment, *, key=None):
            return [
                SentenceInfo(
                    text="Lượt một" if segment.start_ms == 0 else "Lượt hai",
                    start=segment.start_s,
                    end=segment.end_s,
                )
            ]

        async def finalize_incremental(self, prepared, sentences):
            return sentences

    published: list[dict] = []

    class FakeEventBus:
        async def publish(self, event_job_id, event):
            assert event_job_id == job_id
            published.append(event.model_dump(mode="json"))

    monkeypatch.setattr(job_worker, "event_bus", FakeEventBus())
    queue = job_worker.JobQueue()
    queue._storage = FakeStorage()
    queue._asr_service = FakeASRService()

    await queue._process(job_id)

    with Session(engine) as db:
        segments = db.exec(
            select(TranscriptSegment)
            .where(TranscriptSegment.job_id == job_id)
            .order_by(TranscriptSegment.start_ms)
        ).all()

    assert [segment.speaker for segment in segments] == [0, 1]
    transcript_events = [
        event for event in published if event["type"] == "transcript_delta"
    ]
    assert [
        event["segment"]["speaker"] for event in transcript_events
    ] == [0, 1]
    assert not any(event["type"] == "speaker_update" for event in published)
    engine.dispose()
