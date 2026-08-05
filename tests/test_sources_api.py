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

from meetasr.api.auth_deps import get_current_user
from meetasr.api.routes import sources
from meetasr.db.connection import get_db
from meetasr.db.models_phase2 import Job, JobStatus, MediaType, Source
from meetasr.db.user_model import User
from meetasr.storage.backend import StorageBackend

# ============================================================
# Fake storage
# ============================================================

@dataclass
class FakeStorage(StorageBackend):
    """
    Storage giả trong RAM.
    Không tạo file thật.
    """

    objects: dict[str, bytes] = field(default_factory=dict)
    deleted_keys: list[str] = field(default_factory=list)

    async def save(
        self,
        file_data: BinaryIO | bytes,
        filename: str,
    ) -> str:

        payload = (
            file_data
            if isinstance(file_data, bytes)
            else file_data.read()
        )

        key = f"fake/{filename}"
        self.objects[key] = payload

        return key


    async def load(self, key: str) -> bytes:
        if key not in self.objects:
            raise FileNotFoundError(key)

        return self.objects[key]


    async def delete(self, key: str) -> None:
        self.deleted_keys.append(key)
        self.objects.pop(key, None)


    def public_url(self, key: str) -> str:
        return f"/fake-media/{key}"



# ============================================================
# Fake authenticated user
# ============================================================

def fake_current_user() -> User:
    """
    Mock user.

    Đây không phải dữ liệu production.
    Chỉ để test dependency authentication.
    """

    return User(
        id="test-user-id",
        provider="test",
        provider_id="test-provider-id",
        email="test@example.com",
        name="Test User",
    )



# ============================================================
# Test application
# ============================================================

@pytest.fixture
def sources_context() -> Generator[
    tuple[TestClient, object, FakeStorage],
    None,
    None,
]:

    test_engine = create_engine(
        "sqlite://",
        connect_args={
            "check_same_thread": False
        },
        poolclass=StaticPool,
    )


    SQLModel.metadata.create_all(test_engine)


    fake_storage = FakeStorage()


    def override_get_db():
        with Session(test_engine) as session:
            yield session



    test_app = FastAPI()

    test_app.include_router(
        sources.router
    )


    # DB mock
    test_app.dependency_overrides[
        get_db
    ] = override_get_db


    # Storage mock
    test_app.dependency_overrides[
        sources.get_storage_backend
    ] = lambda: fake_storage


    # Auth mock
    test_app.dependency_overrides[
        get_current_user
    ] = fake_current_user



    with TestClient(test_app) as client:
        yield (
            client,
            test_engine,
            fake_storage,
        )


    test_engine.dispose()



# ============================================================
# CRUD test
# ============================================================

def test_sources_crud_creates_queued_job_and_deletes_storage(
    sources_context,
) -> None:

    """
    POST /v1/sources
    GET /v1/sources
    GET /v1/sources/{id}
    DELETE /v1/sources/{id}

    Không chạy worker.
    """

    client, test_engine, fake_storage = sources_context



    # -----------------------------
    # CREATE
    # -----------------------------

    create_response = client.post(
        "/v1/sources",
        files={
            "file": (
                "meeting.wav",
                b"fake audio bytes",
                "audio/wav",
            )
        },
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



    # -----------------------------
    # DB verify
    # -----------------------------

    with Session(test_engine) as session:

        source = session.get(
            Source,
            source_id,
        )


        assert source is not None

        assert source.user_id == "test-user-id"

        assert source.storage_path == (
            "fake/meeting.wav"
        )


        job = session.exec(
            select(Job)
            .where(
                Job.source_id == source_id
            )
        ).one()


        assert job.id == job_id
        assert job.status == JobStatus.QUEUED



    # -----------------------------
    # LIST
    # -----------------------------

    list_response = client.get(
        "/v1/sources"
    )


    assert list_response.status_code == 200


    items = list_response.json()


    assert [
        item["id"]
        for item in items
    ] == [
        source_id
    ]



    # -----------------------------
    # DETAIL
    # -----------------------------

    detail_response = client.get(
        f"/v1/sources/{source_id}"
    )


    assert detail_response.status_code == 200


    detail = detail_response.json()


    assert detail["id"] == source_id
    assert detail["title"] == "meeting.wav"
    assert detail["mediaType"] == MediaType.AUDIO
    assert detail["durationMs"] is None
    assert detail["status"] == "processing"
    assert detail["jobId"] == job_id
    assert detail["docs"] == []
    assert detail["documents"] == []



    # -----------------------------
    # DELETE
    # -----------------------------

    delete_response = client.delete(
        f"/v1/sources/{source_id}"
    )


    assert delete_response.status_code == 204


    assert fake_storage.deleted_keys == [
        "fake/meeting.wav"
    ]


    assert (
        "fake/meeting.wav"
        not in fake_storage.objects
    )



    # -----------------------------
    # DB removed
    # -----------------------------

    with Session(test_engine) as session:

        assert (
            session.get(
                Source,
                source_id,
            )
            is None
        )


        assert (
            session.get(
                Job,
                job_id,
            )
            is None
        )
