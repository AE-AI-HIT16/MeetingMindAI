"""Deleting a source that had generated documents must not hit FK errors."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from meetasr.backend.api.auth_deps import get_current_user
from meetasr.backend.api.routes import sources
from meetasr.backend.db.connection import get_db
from meetasr.backend.db.models_phase2 import (
    Document,
    DocumentGenerationJob,
    Job,
    MediaType,
    Source,
)


class FakeStorage:
    async def delete(self, key):
        return None


def test_delete_source_with_generation_jobs():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _):  # enforce FKs like Postgres
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        source = Source(filename="a.wav", media_type=MediaType.AUDIO, storage_path="k")
        db.add(source)
        db.commit()
        db.add(Job(source_id=source.id))
        live = Document(source_id=source.id, mode="live", markdown="# x")
        summary = Document(source_id=source.id, mode="summary", markdown="# s")
        db.add(live)
        db.add(summary)
        db.commit()
        db.add(DocumentGenerationJob(source_id=source.id, document_id=summary.id, mode="summary"))
        db.commit()
        source_id = source.id

    app = FastAPI()
    app.include_router(sources.router)

    def _db():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[sources.get_storage_backend] = lambda: FakeStorage()
    app.dependency_overrides[get_current_user] = lambda: None

    response = TestClient(app).delete(f"/v1/sources/{source_id}")

    assert response.status_code == 204, response.text
    with Session(engine) as db:
        assert db.exec(select(Source)).all() == []
        assert db.exec(select(Document)).all() == []
        assert db.exec(select(DocumentGenerationJob)).all() == []
