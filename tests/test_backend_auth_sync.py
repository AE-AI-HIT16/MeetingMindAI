"""Login sync: a returning user (the common case) must get a backend token."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from meetasr.backend.api.routes import auth
from meetasr.backend.db.connection import get_db
from meetasr.backend.db.user_model import User


def test_sync_new_then_returning_user_returns_token():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    app = FastAPI()
    app.include_router(auth.router)

    def _db():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_db] = _db
    client = TestClient(app)
    body = {
        "provider": "google", "provider_id": "g-1", "email": "a@example.com",
        "name": "An", "avatar_url": None, "sync_secret": auth._JWT_SECRET,
    }

    first = client.post("/v1/auth/sync", json=body)
    second = client.post("/v1/auth/sync", json=body)  # returning user: used to 500

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json()["access_token"]
    with Session(engine) as db:
        user = db.exec(select(User)).one()
        assert user.last_login_at is not None
