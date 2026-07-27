"""Smoke tests for the Phase 2 database schema and relationships."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import meetasr.db.connection as db_connection
from meetasr.db.models_phase2 import (
    Document,
    DocumentGenerationJob,
    DocumentMode,
    Job,
    JobStatus,
    MediaType,
    Source,
    TranscriptSegment,
)


def _sqlite_engine():
    """Create a thread-safe in-memory engine shared by all test sessions."""
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def test_init_db_creates_all_phase2_tables(monkeypatch) -> None:
    """Application database initialization must register all Phase 2 models."""
    test_engine = _sqlite_engine()
    monkeypatch.setattr(db_connection, "engine", test_engine)

    db_connection.init_db()

    table_names = set(inspect(test_engine).get_table_names())
    assert {
        "sources",
        "jobs",
        "documents",
        "document_generation_jobs",
        "document_duplicates_archive",
        "transcript_segments",
    }.issubset(table_names)


def test_deleting_source_cascades_job_document_and_segment() -> None:
    """Deleting a Source removes its complete Phase 2 object graph."""
    test_engine = _sqlite_engine()
    SQLModel.metadata.create_all(test_engine)

    source = Source(
        filename="meeting.wav",
        media_type=MediaType.AUDIO,
        storage_path="source-1/meeting.wav",
    )
    job = Job(source=source, status=JobStatus.QUEUED)
    document = Document(
        source=source,
        mode=DocumentMode.SUMMARY,
        markdown="# Tóm tắt",
    )
    segment = TranscriptSegment(
        job=job,
        start_ms=0,
        end_ms=1_000,
        speaker=0,
        text="Nội dung kiểm thử.",
    )
    generation = DocumentGenerationJob(
        document_id=document.id,
        source=source,
        mode=DocumentMode.SUMMARY,
    )

    with Session(test_engine) as session:
        session.add(source)
        session.commit()

        source_id = source.id
        job_id = job.id
        document_id = document.id
        generation_id = generation.id
        segment_id = segment.id

        assert session.get(Source, source_id) is not None
        assert session.get(Job, job_id) is not None
        assert session.get(Document, document_id) is not None
        assert (
            session.get(DocumentGenerationJob, generation_id) is not None
        )
        assert segment_id is not None
        assert session.get(TranscriptSegment, segment_id) is not None

        session.delete(source)
        session.commit()

        assert session.get(Source, source_id) is None
        assert session.get(Job, job_id) is None
        assert session.get(Document, document_id) is None
        assert session.get(DocumentGenerationJob, generation_id) is None
        assert session.get(TranscriptSegment, segment_id) is None


def test_document_source_and_mode_are_unique() -> None:
    test_engine = _sqlite_engine()
    SQLModel.metadata.create_all(test_engine)

    with Session(test_engine) as session:
        source = Source(
            filename="meeting.wav",
            media_type=MediaType.AUDIO,
            storage_path="meeting.wav",
        )
        session.add(source)
        session.flush()
        session.add(
            Document(
                source_id=source.id,
                mode=DocumentMode.SUMMARY,
                markdown="# First",
            )
        )
        session.commit()

        session.add(
            Document(
                source_id=source.id,
                mode=DocumentMode.SUMMARY,
                markdown="# Duplicate",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
