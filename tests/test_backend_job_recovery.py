"""Jobs interrupted by a restart are requeued (or failed when no audio)."""

from __future__ import annotations

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from meetasr.backend.db.models_phase2 import Job, JobStatus, MediaType, Source
from meetasr.backend.realtime.job_worker import recover_interrupted_jobs


def test_recover_interrupted_jobs():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    cases = {
        "processing_with_audio": ("k1", JobStatus.PROCESSING),
        "processing_no_audio": ("", JobStatus.PROCESSING),
        "queued_with_audio": ("k2", JobStatus.QUEUED),
        "queued_no_audio": ("", JobStatus.QUEUED),
        "done": ("k3", JobStatus.DONE),
    }
    ids = {}
    with Session(engine) as db:
        for name, (path, status) in cases.items():
            source = Source(filename=f"{name}.wav", media_type=MediaType.AUDIO, storage_path=path)
            db.add(source)
            db.commit()
            job = Job(source_id=source.id, status=status)
            db.add(job)
            db.commit()
            ids[name] = job.id

    assert recover_interrupted_jobs(engine) == (1, 2)

    with Session(engine) as db:
        status = {name: db.get(Job, job_id).status for name, job_id in ids.items()}
        assert status == {
            "processing_with_audio": JobStatus.QUEUED,
            "processing_no_audio": JobStatus.FAILED,
            "queued_with_audio": JobStatus.QUEUED,
            "queued_no_audio": JobStatus.FAILED,
            "done": JobStatus.DONE,
        }
        assert db.get(Job, ids["processing_no_audio"]).error
