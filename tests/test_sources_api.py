"""Isolated CRUD tests for the Phase 2 Sources API."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import BinaryIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from meetasr.api.routes import sources
from meetasr.db.connection import get_db
from meetasr.db.models_phase2 import Job, JobStatus, MediaType, Source
from meetasr.storage.backend import StorageBackend


@dataclass
class FakeStorage(StorageBackend):
    """In-memory storage double that records persisted and deleted objects."""

    objects: dict[str, bytes] = field(default_factory=dict)
    deleted_keys: list[str] = field(default_factory=list)

    async def save(
        self,
        file_data: BinaryIO | bytes,
        filename: str,
    ) -> str:
        payload = file_data if isinstance(file_data, bytes) else file_data.read()
        key = f"fake/{filename}"
        self.objects[key] = payload
        return key

    async def load(self, key: str) -> bytes:
        try:
            return self.objects[key]
        except KeyError as exc:
            raise FileNotFoundError(key) from exc

    async def delete(self, key: str) -> None:
        self.deleted_keys.append(key)
        self.objects.pop(key, None)

    def public_url(self, key: str) -> str:
        return f"/fake-media/{key}"


@pytest.fixture
def sources_context() -> Generator[tuple[TestClient, object, FakeStorage], None, None]:
    """Build a minimal Sources app with isolated DB and storage dependencies."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)
    fake_storage = FakeStorage()

    def override_get_db() -> Generator[Session, None, None]:
        with Session(test_engine) as session:
            yield session

    test_app = FastAPI()
    test_app.include_router(sources.router)
    test_app.dependency_overrides[get_db] = override_get_db
    test_app.dependency_overrides[sources.get_storage_backend] = lambda: fake_storage

    with TestClient(test_app) as client:
        yield client, test_engine, fake_storage

    test_engine.dispose()


def test_sources_crud_creates_queued_job_and_deletes_storage(
    sources_context,
) -> None:
    """POST/GET/DELETE persists the expected DB state without running a worker."""
    client, test_engine, fake_storage = sources_context

    create_response = client.post(
        "/v1/sources",
        files={"file": ("meeting.wav", b"fake audio", "audio/wav")},
    )

    assert create_response.status_code == 201
    created = create_response.json()
    source_id = created["sourceId"]
    job_id = created["jobId"]
    assert created == {
        "sourceId": source_id,
        "jobId": job_id,
        "status": JobStatus.QUEUED,
    }

    with Session(test_engine) as session:
        source = session.get(Source, source_id)
        assert source is not None
        assert source.storage_path == "fake/meeting.wav"
        job = session.exec(select(Job).where(Job.source_id == source_id)).one()
        assert job.id == job_id
        assert job.status == JobStatus.QUEUED

    list_response = client.get("/v1/sources")
    assert list_response.status_code == 200
    assert [item["id"] for item in list_response.json()] == [source_id]

    detail_response = client.get(f"/v1/sources/{source_id}")
    assert detail_response.status_code == 200
    assert detail_response.json() == {
        "id": source_id,
        "title": "meeting.wav",
        "mediaType": MediaType.AUDIO,
        "durationMs": None,
        "createdAt": detail_response.json()["createdAt"],
        "status": "processing",
        "jobId": job_id,
        "docs": [],
        "documents": [],
    }

    delete_response = client.delete(f"/v1/sources/{source_id}")
    assert delete_response.status_code == 204
    assert fake_storage.deleted_keys == ["fake/meeting.wav"]
    assert "fake/meeting.wav" not in fake_storage.objects

    with Session(test_engine) as session:
        assert session.get(Source, source_id) is None
        assert session.get(Job, job_id) is None
