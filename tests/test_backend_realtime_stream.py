"""Production realtime WS route (meetasr.backend): warmup, window size, tail flush."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from meetasr.backend.api.routes import realtime, sources
from meetasr.backend.realtime import job_worker
from meetasr.backend.db.connection import get_db
from meetasr.backend.db.models_phase2 import Job, JobStatus, Source
from meetasr.backend.services.asr_service import ASRServiceResult

SAMPLE_RATE = 16000


class FakeASR:
    def __init__(self):
        self.windows: list[float] = []
        self.warmups = 0

    async def warmup(self):
        self.warmups += 1

    async def transcribe(self, audio, *, offset_ms=0, key=None):
        self.windows.append(len(audio) / SAMPLE_RATE)
        return ASRServiceResult(segments=[], text="", duration_ms=int(len(audio) / SAMPLE_RATE * 1000))


class FakeStorage:
    def __init__(self):
        self.saved: list[bytes] = []

    async def save(self, data, filename):
        self.saved.append(data)
        return f"key/{filename}"


@pytest.fixture
def harness(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(realtime, "engine", engine)
    storage = FakeStorage()
    monkeypatch.setattr(sources, "get_storage_backend", lambda: storage)

    app = FastAPI()
    app.include_router(realtime.router)
    asr = FakeASR()
    app.state.realtime_asr_service = asr
    app.state.realtime_pipeline = SimpleNamespace(vad=SimpleNamespace(name="fixed-window"))

    def _db():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_db] = _db
    return SimpleNamespace(app=app, client=TestClient(app), asr=asr, storage=storage, engine=engine)


def _pcm(seconds: float) -> bytes:
    return (np.full(int(seconds * SAMPLE_RATE), 1000, dtype=np.int16)).tobytes()


def _stream(client, url: str, seconds: float) -> list[dict]:
    with client.websocket_connect(url) as ws:
        init = ws.receive_json()
        assert init["type"] == "session_init"
        audio = _pcm(seconds)
        for i in range(0, len(audio), 3200):
            ws.send_bytes(audio[i:i + 3200])
        ws.send_json({"type": "stop"})
        events = []
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg["type"] == "stream_stopped":
                return events


def test_stop_archives_audio_and_queues_offline_job(harness):
    _stream(harness.client, "/v1/realtime/stream", 3.0)

    assert len(harness.storage.saved) == 1
    # WAV header (44 B) + 3 s PCM16 mono
    assert len(harness.storage.saved[0]) == 44 + 3 * SAMPLE_RATE * 2
    with Session(harness.engine) as s:
        assert s.exec(select(Job)).one().status == JobStatus.QUEUED
        assert s.exec(select(Source)).one().storage_path.startswith("key/")


def test_warms_up_and_transcribes_tail_on_stop(harness):
    # 7 s < 30 s window: previously nothing was transcribed before stop.
    events = _stream(harness.client, "/v1/realtime/stream", 7.0)

    assert harness.asr.warmups == 1
    assert sum(harness.asr.windows) == pytest.approx(7.0, abs=0.01)
    confirmed = [e["end_ms"] for e in events if e["type"] == "transcript_confirmed"]
    assert confirmed == [7000]


def test_failed_window_is_skipped_and_later_windows_still_confirm(harness):
    calls = {"n": 0}
    original = harness.asr.transcribe

    async def flaky(audio, *, offset_ms=0, key=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("RunPod down")
        return await original(audio, offset_ms=offset_ms, key=key)

    harness.asr.transcribe = flaky
    events = _stream(harness.client, "/v1/realtime/stream?window=5", 12.0)

    confirmed = [e["end_ms"] for e in events if e["type"] == "transcript_confirmed"]
    # first 5 s window failed; timeline still advances for the rest
    assert confirmed[-1] == 12000
    assert 5000 not in confirmed


def test_window_param_controls_asr_window_length(harness):
    _stream(harness.client, "/v1/realtime/stream?window=5", 12.0)

    assert len(harness.asr.windows) >= 2
    assert max(harness.asr.windows) <= 5.0 + 1e-6
    assert sum(harness.asr.windows) == pytest.approx(12.0, abs=0.01)


def test_stop_starts_offline_job_immediately_when_asr_service_available(harness, monkeypatch):
    started: list[str] = []

    async def fake_run(job_id, asr_service, storage):
        started.append(job_id)

    monkeypatch.setattr(job_worker, "run_job_processing", fake_run)
    harness.app.state.asr_service = object()

    init = None
    with harness.client.websocket_connect("/v1/realtime/stream") as ws:
        init = ws.receive_json()
        ws.send_bytes(_pcm(1.0))
        ws.send_json({"type": "stop"})
        while ws.receive_json()["type"] != "stream_stopped":
            pass

    assert started == [init["job_id"]]
    with Session(harness.engine) as s:
        assert s.exec(select(Job)).one().status == JobStatus.PROCESSING
