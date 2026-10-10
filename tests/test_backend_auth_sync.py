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


def test_google_and_github_with_same_email_share_one_library():
    import meetasr.backend.db.models_phase2  # noqa: F401  -- sources table
    from datetime import datetime, timedelta, timezone

    from meetasr.backend.db.models_phase2 import Source

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

    # Existing data: a Google account and a GitHub duplicate, each with uploads.
    with Session(engine) as db:
        google = User(provider="google", provider_id="g-1", email="a@example.com", name="An",
                      created_at=datetime.now(timezone.utc) - timedelta(days=1))
        github = User(provider="github", provider_id="gh-1", email="A@example.com", name="An")
        db.add_all([google, github]); db.commit()
        db.add_all([
            Source(filename="g.mp3", media_type="audio", storage_path="g", user_id=google.id),
            Source(filename="h.mp3", media_type="audio", storage_path="h", user_id=github.id),
        ]); db.commit()
        google_id = google.id

    for provider, provider_id in (("github", "gh-1"), ("google", "g-1")):
        body = {
            "provider": provider, "provider_id": provider_id, "email": "a@example.com",
            "name": "An", "avatar_url": None, "sync_secret": auth._JWT_SECRET,
        }
        response = client.post("/v1/auth/sync", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["user"]["id"] == google_id

    with Session(engine) as db:
        owners = {s.filename: s.user_id for s in db.exec(select(Source))}
        assert owners == {"g.mp3": google_id, "h.mp3": google_id}

    # A brand-new GitHub login with the same email does not create a third account.
    body["provider"], body["provider_id"] = "github", "gh-2"
    assert client.post("/v1/auth/sync", json=body).json()["user"]["id"] == google_id
    with Session(engine) as db:
        assert len(db.exec(select(User)).all()) == 2
