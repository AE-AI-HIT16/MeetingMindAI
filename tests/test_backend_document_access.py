"""Documents (view / finalize / export) are only accessible to the Source owner."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from meetasr.backend.api.auth_deps import get_current_user
from meetasr.backend.api.routes import documents_phase2
from meetasr.backend.db.connection import get_db
from meetasr.backend.db.models_phase2 import Document, MediaType, Source
from meetasr.backend.db.user_model import User


@pytest.fixture
def setup():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        owner = User(provider="google", provider_id="o", email="o@x.com", name="Owner")
        other = User(provider="google", provider_id="t", email="t@x.com", name="Other")
        db.add(owner)
        db.add(other)
        db.commit()
        owned = Source(filename="a.wav", media_type=MediaType.AUDIO, storage_path="k", user_id=owner.id)
        guest = Source(filename="g.wav", media_type=MediaType.AUDIO, storage_path="k")
        db.add(owned)
        db.add(guest)
        db.commit()
        owned_doc = Document(source_id=owned.id, mode="live", markdown="# A\n\nx")
        guest_doc = Document(source_id=guest.id, mode="live", markdown="# G\n\nx")
        db.add(owned_doc)
        db.add(guest_doc)
        db.commit()
        ids = {
            "owner": owner.id, "other": other.id,
            "owned_doc": owned_doc.id, "guest_doc": guest_doc.id,
        }

    app = FastAPI()
    app.include_router(documents_phase2.router)

    def _db():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_db] = _db
    state = {"user": None}

    def _user():
        if state["user"] is None:
            return None
        with Session(engine) as s:
            return s.get(User, state["user"])

    app.dependency_overrides[get_current_user] = _user
    return TestClient(app), state, ids


def _calls(doc_id):
    return [
        ("get", f"/v1/documents/{doc_id}"),
        ("get", f"/v1/documents/{doc_id}/export?format=md"),
        ("post", f"/v1/documents/{doc_id}/finalize"),
    ]


@pytest.mark.parametrize("who", [None, "other"])
def test_non_owner_is_forbidden(setup, who):
    client, state, ids = setup
    state["user"] = ids[who] if who else None
    for method, url in _calls(ids["owned_doc"]):
        response = getattr(client, method)(url, **({"json": {"mode": "summary"}} if method == "post" else {}))
        assert response.status_code == 403, (url, response.status_code)


def test_owner_can_view_and_export(setup):
    client, state, ids = setup
    state["user"] = ids["owner"]
    assert client.get(f"/v1/documents/{ids['owned_doc']}").status_code == 200
    assert client.get(f"/v1/documents/{ids['owned_doc']}/export?format=md").status_code == 200


def test_guest_documents_stay_accessible(setup):
    client, state, ids = setup
    assert client.get(f"/v1/documents/{ids['guest_doc']}").status_code == 200
    assert client.get(f"/v1/documents/{ids['guest_doc']}/export?format=md").status_code == 200


def test_unknown_document_is_still_404(setup):
    client, _, _ = setup
    assert client.get("/v1/documents/does-not-exist").status_code == 404
