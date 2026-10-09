"""Tests for lossless realtime stop and queue draining."""

from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import WebSocketDisconnect

from meetasr.streaming.coverage import RealtimeCoverageTracker
from meetasr.streaming.final_transcript_queue import (
    FinalTranscriptJob,
    FinalTranscriptQueue,
)
from meetasr.streaming.graceful_stop import drain_realtime_session
from meetasr.streaming.protocol import RealtimeAuthFrame, receive_stream_frame
from meetasr.streaming.window_builder import SegmentWindowBuilder
from meetasr.streaming.worker import AudioWorker


@pytest.mark.asyncio
async def test_audio_worker_flushes_subsecond_tail_to_confirmed_asr() -> None:
    """The last buffered audio must not be discarded when recording stops."""
    class SpeechPredictor:
        def predict(self, frame: np.ndarray) -> float:
            return 0.9

        def reset(self) -> None:
            return None

    class FakeVAD:
        threshold = 0.5
        neg_threshold = 0.35
        min_speech_duration_ms = 250

        def create_streaming_predictor(self) -> SpeechPredictor:
            return SpeechPredictor()

    tail = np.ones(8000, dtype=np.float32)
    session = SimpleNamespace(
        pending_audio=tail.copy(),
        ready_segments=deque(),
        asr_queue=asyncio.Queue(),
        temp_asr_queue=asyncio.Queue(),
        vad_state=None,
        last_partial_request_ms=0,
        coverage=RealtimeCoverageTracker(),
    )
    window_builder = SegmentWindowBuilder(session)
    pipeline = SimpleNamespace(vad=FakeVAD(), realtime_config={})
    worker = AudioWorker(session, pipeline, window_builder)

    await worker.flush()

    queued = session.asr_queue.get_nowait()
    session.asr_queue.task_done()
    np.testing.assert_array_equal(queued.audio, tail)
    assert (queued.start_ms, queued.end_ms) == (0, 500)
    assert session.pending_audio.size == 0
    snapshot = session.coverage.snapshot()
    assert snapshot.flush_completed is True
    assert [(item.start_ms, item.end_ms) for item in snapshot.speech_ranges] == [
        (0, 500)
    ]


@pytest.mark.asyncio
async def test_drain_waits_for_audio_tail_and_confirmed_asr_in_order() -> None:
    calls: list[str] = []

    class Receiver:
        async def flush(self) -> None:
            calls.append("receiver.flush")

    class Worker:
        async def flush(self) -> None:
            calls.append("worker.flush")

    class Queue:
        def __init__(self, name: str) -> None:
            self.name = name

        async def join(self) -> None:
            calls.append(f"{self.name}.join")

    session = SimpleNamespace(
        audio_queue=Queue("audio"),
        asr_queue=Queue("asr"),
        partial_cut_queue=Queue("partial_cut"),
    )

    await drain_realtime_session(
        session,
        Receiver(),
        Worker(),
        timeout_seconds=1.0,
    )

    assert calls == [
        "receiver.flush",
        "audio.join",
        "worker.flush",
        "asr.join",
        "partial_cut.join",
    ]


class FakeWebSocket:
    def __init__(self, message: dict) -> None:
        self.message = message

    async def receive(self) -> dict:
        return self.message


@pytest.mark.asyncio
async def test_protocol_accepts_binary_audio_and_json_stop() -> None:
    audio = await receive_stream_frame(
        FakeWebSocket({"type": "websocket.receive", "bytes": b"pcm"})
    )
    stop = await receive_stream_frame(
        FakeWebSocket(
            {"type": "websocket.receive", "text": '{"type":"stop"}'}
        )
    )

    assert audio == b"pcm"
    assert stop == "stop"


@pytest.mark.asyncio
async def test_protocol_accepts_auth_as_first_control_frame() -> None:
    frame = await receive_stream_frame(
        FakeWebSocket(
            {
                "type": "websocket.receive",
                "text": '{"type":"auth","token":"signed-token"}',
            }
        )
    )

    assert frame == RealtimeAuthFrame(token="signed-token")


@pytest.mark.asyncio
async def test_protocol_preserves_abrupt_disconnect_fallback() -> None:
    with pytest.raises(WebSocketDisconnect):
        await receive_stream_frame(
            FakeWebSocket({"type": "websocket.disconnect", "code": 1006})
        )


@pytest.mark.asyncio
async def test_full_finalization_queue_waits_instead_of_dropping() -> None:
    queue = FinalTranscriptQueue(maxsize=1)
    first = FinalTranscriptJob(
        job_id="job-1",
        audio=np.zeros(1, dtype=np.float32),
    )
    second = FinalTranscriptJob(
        job_id="job-2",
        audio=np.zeros(1, dtype=np.float32),
    )

    assert await queue.put(first) is True
    pending = asyncio.create_task(queue.put(second))
    await asyncio.sleep(0)
    assert pending.done() is False

    assert await queue.get() is first
    queue.task_done()
    assert await pending is True
    assert await queue.get() is second
    queue.task_done()
