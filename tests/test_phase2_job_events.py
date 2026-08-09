"""Replay tests for the Phase 2 Job WebSocket contract."""

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
