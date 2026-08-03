"""Tests for absolute timeline handling in confirmed realtime ASR."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

from meetasr.api.schemas_phase2 import TranscriptSegmentPayload
from meetasr.pipeline_realtime import ASRPipeline
from meetasr.services import realtime_asr_service
from meetasr.services.asr_service import ASRServiceResult
from meetasr.services.realtime_asr_service import RealtimeASRService
from meetasr.streaming.asr_worker import ASRWorker
from meetasr.streaming.coverage import RealtimeCoverageTracker
from meetasr.streaming.window_builder import ASRWindow


@pytest.mark.asyncio
async def test_asr_window_uses_absolute_vad_offset_and_cut_position() -> None:
    calls: list[tuple[np.ndarray, int]] = []

    class FakeService:
        async def transcribe(
            self,
            audio: np.ndarray,
            *,
            offset_ms: int = 0,
        ) -> ASRServiceResult:
            calls.append((audio, offset_ms))
            return ASRServiceResult(segments=[], text="xin chào", duration_ms=900)

    session = SimpleNamespace(
        confirmed_end_ms=2000,
        partial_cut_queue=asyncio.Queue(),
    )
    worker = ASRWorker(session, FakeService())
    window = ASRWindow(
        audio=np.ones(16000, dtype=np.float32),
        start_ms=5000,
        end_ms=6000,
        reason="speech_end",
    )

    result = await worker._transcribe(window)
    await worker._publish_cut_event(window)

    assert result.text == "xin chào"
    assert calls[0][1] == 5000
    assert session.confirmed_end_ms == 6000
    assert session.partial_cut_queue.get_nowait() == 6000


@pytest.mark.asyncio
async def test_confirmed_window_does_not_run_offline_vad_again(
    monkeypatch,
) -> None:
    class FailingVAD:
        def detect(self, audio: np.ndarray) -> list[object]:
            pytest.fail("endpointed realtime audio must not run VAD twice")

    class FakeASR:
        def recognize(
            self,
            chunks: list[np.ndarray],
            **kwargs: object,
        ) -> list[dict[str, object]]:
            assert len(chunks) == 1
            assert kwargs["language"] == "vi"
            return [{"text": "đồng ý", "char_timestamps": []}]

    async def run_inline(
        function: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        return function(*args, **kwargs)

    monkeypatch.setattr(realtime_asr_service.asyncio, "to_thread", run_inline)
    service = RealtimeASRService(
        ASRPipeline(
            asr_model=FakeASR(),
            vad_model=FailingVAD(),
            transcription_language="vi",
        )
    )

    result = await service.transcribe(
        np.ones(16000, dtype=np.float32),
        offset_ms=7000,
    )

    assert result.text == "đồng ý"
    assert [(item.start_ms, item.end_ms) for item in result.segments] == [
        (7000, 8000)
    ]


@pytest.mark.asyncio
async def test_asr_failure_marks_incomplete_coverage_and_worker_continues() -> None:
    calls = 0

    class FakeASRService:
        async def transcribe(self, audio, *, offset_ms=0):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary ASR failure")
            return ASRServiceResult(
                segments=[
                    TranscriptSegmentPayload(
                        start_ms=offset_ms,
                        end_ms=offset_ms + 1000,
                        text="Câu thứ hai.",
                    )
                ],
                text="Câu thứ hai.",
                duration_ms=1000,
            )

    class FakeTranscriptService:
        async def persist_and_publish(self, job_id, segment, websocket):
            return segment.model_copy(update={"id": 10})

    session = SimpleNamespace(
        job_id="job-coverage",
        websocket=SimpleNamespace(),
        asr_queue=asyncio.Queue(),
        partial_cut_queue=asyncio.Queue(),
        confirmed_end_ms=0,
        coverage=RealtimeCoverageTracker(),
    )
    for start_ms, end_ms in [(0, 1000), (1500, 2500)]:
        session.coverage.record_speech(start_ms, end_ms)
        await session.asr_queue.put(
            ASRWindow(
                audio=np.ones(16000, dtype=np.float32),
                start_ms=start_ms,
                end_ms=end_ms,
                reason="speech_end",
            )
        )
    worker = ASRWorker(
        session,
        FakeASRService(),
        transcript_service=FakeTranscriptService(),
    )
    task = asyncio.create_task(worker.run())

    await asyncio.wait_for(session.asr_queue.join(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    snapshot = session.coverage.snapshot()
    assert calls == 2
    assert snapshot.asr_failure_count == 1
    assert [(item.start_ms, item.end_ms) for item in snapshot.missing_ranges] == [
        (0, 1000)
    ]
    assert session.partial_cut_queue.get_nowait() == 2500
