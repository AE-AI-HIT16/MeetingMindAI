"""Latency and bounded-work tests for realtime partial transcription."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import numpy as np
import pytest

from meetasr.streaming import temp_asr_woker
from meetasr.streaming.streaming_processor import StreamingProcessor
from meetasr.streaming.temp_asr_woker import PartialASRRequest, TempASRWorker


class _AlwaysSpeech:
    def predict(self, frame: np.ndarray) -> float:
        del frame
        return 0.9

    def reset(self) -> None:
        return None


class _FakeVAD:
    threshold = 0.5
    neg_threshold = 0.35
    min_speech_duration_ms = 250

    def create_streaming_predictor(self) -> _AlwaysSpeech:
        return _AlwaysSpeech()


def _processor_session() -> SimpleNamespace:
    return SimpleNamespace(
        pending_audio=np.empty(0, dtype=np.float32),
        ready_segments=[],
        vad_state=None,
        temp_asr_queue=asyncio.Queue(maxsize=1),
        last_partial_request_ms=0,
    )


@pytest.mark.asyncio
async def test_partial_waits_for_minimum_audio_then_uses_rolling_window() -> None:
    session = _processor_session()
    pipeline = SimpleNamespace(
        vad=_FakeVAD(),
        realtime_config={
            "vad": {"pre_roll_ms": 0},
            "asr": {
                "partial_interval_ms": 800,
                "partial_min_audio_ms": 700,
                "partial_max_audio_ms": 5000,
            },
        },
    )
    processor = StreamingProcessor(session, pipeline)

    processor.detector.process(np.ones(640 * 16, dtype=np.float32))
    await processor.request_partial_if_due()
    assert session.temp_asr_queue.empty()

    processor.detector.process(np.ones(640 * 2, dtype=np.float32))
    await processor.request_partial_if_due()
    first = session.temp_asr_queue.get_nowait()
    session.temp_asr_queue.task_done()
    assert first.start_ms == 0
    assert 700 <= first.end_ms < 800

    processor.detector.process(np.ones(16000 * 6, dtype=np.float32))
    await processor.request_partial_if_due()
    latest = session.temp_asr_queue.get_nowait()
    session.temp_asr_queue.task_done()
    assert latest.end_ms - latest.start_ms == 5000


@pytest.mark.asyncio
async def test_waiting_partial_request_is_replaced_by_newest() -> None:
    session = _processor_session()
    pipeline = SimpleNamespace(
        vad=_FakeVAD(),
        realtime_config={
            "vad": {"pre_roll_ms": 0},
            "asr": {
                "partial_interval_ms": 800,
                "partial_min_audio_ms": 700,
                "partial_max_audio_ms": 5000,
            },
        },
    )
    processor = StreamingProcessor(session, pipeline)

    processor.detector.process(np.ones(16000, dtype=np.float32))
    await processor.request_partial_if_due()
    processor.detector.process(np.ones(16000, dtype=np.float32))
    await processor.request_partial_if_due()

    assert session.temp_asr_queue.qsize() == 1
    request = session.temp_asr_queue.get_nowait()
    session.temp_asr_queue.task_done()
    assert request.end_ms >= 1900


@pytest.mark.asyncio
async def test_partial_worker_crops_audio_and_forces_configured_language(
    monkeypatch,
    caplog,
) -> None:
    calls: list[tuple[int, str]] = []

    class FakeASR:
        def recognize(self, audio, *, language):
            calls.append((len(audio[0]), language))
            return [{"text": "xin chào"}]

    class FakeWebSocket:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        async def send_json(self, message: dict) -> None:
            self.messages.append(message)

    websocket = FakeWebSocket()
    session = SimpleNamespace(
        partial_buffer=np.ones(16000 * 7, dtype=np.float32),
        partial_buffer_start_ms=0,
        partial_buffer_lock=asyncio.Lock(),
        websocket=websocket,
        job_id="job-latency",
        first_audio_received_at=time.perf_counter() - 1.0,
        first_partial_emitted_at=None,
        last_partial_emitted_at=None,
        partial_emitted_count=0,
    )
    pipeline = SimpleNamespace(
        asr=FakeASR(),
        transcription_language="vi",
    )
    worker = TempASRWorker(session, pipeline)

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(temp_asr_woker.asyncio, "to_thread", run_inline)

    caplog.set_level(logging.INFO, logger="temp_asr")
    await worker.process_request(
        PartialASRRequest(
            start_ms=1000,
            end_ms=6000,
            utterance_start_ms=1000,
            requested_at=time.perf_counter() - 0.1,
        )
    )

    assert calls == [(16000 * 5, "vi")]
    assert websocket.messages[0]["type"] == "transcript_partial"
    assert websocket.messages[0]["segment"]["start_ms"] == 1000
    assert websocket.messages[0]["segment"]["end_ms"] == 6000
    assert session.partial_emitted_count == 1
    assert session.first_partial_emitted_at is not None
    assert "first_audio_to_partial_ms=" in caplog.text
    assert "utterance_to_partial_ms=" in caplog.text
    assert "audio_end_to_request_ms=" in caplog.text
    assert "audio_end_to_emit_ms=" in caplog.text
    assert "request_to_emit_ms=" in caplog.text
