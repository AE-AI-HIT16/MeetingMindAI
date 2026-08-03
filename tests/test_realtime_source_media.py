"""Tests for realtime Source ownership and playable media persistence."""

from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import jwt
import numpy as np
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from meetasr.api import auth_deps
from meetasr.api.routes.realtime import (
    _create_realtime_job,
    _enqueue_final_transcript,
)
from meetasr.db.models_phase2 import Source
from meetasr.db.user_model import User
from meetasr.streaming.audio_archive import ArchivedAudio, AudioArchive
from meetasr.streaming.coverage import RealtimeCoverageTracker
from meetasr.streaming.final_transcript_queue import FinalTranscriptQueue


def _engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


def test_realtime_job_keeps_authenticated_owner() -> None:
    engine = _engine()
    with Session(engine) as db:
        user = User(
            provider="google",
            provider_id="owner-provider-id",
            email="owner@example.com",
            name="Owner",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        token = jwt.encode(
            {"sub": user.id},
            auth_deps._JWT_SECRET,
            algorithm=auth_deps._JWT_ALGORITHM,
        )
        resolved = auth_deps.resolve_token_user(token, db)

        source, _ = _create_realtime_job(db, resolved.id)

        assert db.get(Source, source.id).user_id == user.id
    engine.dispose()


@pytest.mark.asyncio
async def test_realtime_audio_is_saved_as_playable_wav_before_enqueue() -> None:
    engine = _engine()
    queue = FinalTranscriptQueue(maxsize=1)
    websocket = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(final_transcript_queue=queue)
        )
    )
    archive = AudioArchive(spool_threshold_bytes=4)
    samples = np.array([0, 1000, -1000, 2000], dtype=np.int16)
    archive.append(samples.tobytes())
    coverage = RealtimeCoverageTracker()
    coverage.mark_flush_completed()

    class FakeStorage:
        def __init__(self):
            self.saved = b""

        async def save(self, stream, filename):
            assert filename.endswith(".wav")
            self.saved = stream.read()
            return "realtime/session.wav"

        async def delete(self, key):
            raise AssertionError(f"unexpected cleanup: {key}")

    storage = FakeStorage()
    with Session(engine) as db:
        source, job = _create_realtime_job(db, None)
        session = SimpleNamespace(
            source_id=source.id,
            job_id=job.id,
            audio_archive=archive,
            coverage=coverage,
        )

        assert await _enqueue_final_transcript(
            websocket,
            session,
            db=db,
            storage=storage,
        )
        assert db.get(Source, source.id).storage_path == "realtime/session.wav"

    with wave.open(io.BytesIO(storage.saved), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.readframes(4) == samples.tobytes()

    queued = await queue.get()
    assert isinstance(queued.audio, ArchivedAudio)
    queued.cleanup()
    queue.task_done()
    engine.dispose()
