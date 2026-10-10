"""Batched speaker-turn transcription (one RunPod job per batch)."""

from __future__ import annotations

import numpy as np
import pytest

from meetasr.backend.api.schemas_phase2 import Segment
from meetasr.backend.services.asr_service import ASRService, PreparedTranscription


def _service() -> ASRService:
    return ASRService(runpod_url="https://api.runpod.ai/v2/test", api_key="k")


def _prepared(seconds: float = 10.0) -> PreparedTranscription:
    audio = np.zeros(int(seconds * 16000), dtype=np.float32)
    return PreparedTranscription(audio, [], int(seconds * 1000))


@pytest.mark.asyncio
async def test_one_runpod_job_for_all_turns_with_timeline_offsets(monkeypatch):
    calls = []

    async def fake_call(action, payload, timeout=600.0):
        calls.append((action, len(payload["segments"])))
        return [
            [{"text": f"turn{i}", "start": 0.5, "end": 1.0}]
            for i in range(len(payload["segments"]))
        ]

    service = _service()
    monkeypatch.setattr(service, "_call_serverless", fake_call)
    segments = [Segment(0, 2000), Segment(2000, 2000), Segment(5000, 9000)]

    results = await service.transcribe_segments(_prepared(), segments)

    assert calls == [("transcribe_segments", 2)]  # empty turn not sent
    assert results[1] == []
    assert [s.text for s in results[0]] == ["turn0"]
    assert results[2][0].text == "turn1"
    assert results[2][0].start == pytest.approx(5.5)
    assert results[2][0].end == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_falls_back_to_per_turn_calls_on_old_runpod_image(monkeypatch):
    actions = []

    async def fake_call(action, payload, timeout=600.0):
        actions.append(action)
        if action == "transcribe_segments":
            raise RuntimeError("RunPod job failed: Unknown action: transcribe_segments")
        return [{"text": "x", "start": 0.0, "end": 0.5}]

    service = _service()
    monkeypatch.setattr(service, "_call_serverless", fake_call)

    results = await service.transcribe_segments(
        _prepared(), [Segment(0, 1000), Segment(1000, 2000)]
    )

    assert actions == ["transcribe_segments", "transcribe_segment", "transcribe_segment"]
    assert [r[0].start for r in results] == [0.0, 1.0]


@pytest.mark.asyncio
async def test_other_runpod_errors_are_not_swallowed(monkeypatch):
    async def fake_call(action, payload, timeout=600.0):
        raise RuntimeError("RunPod job failed: CUDA out of memory")

    service = _service()
    monkeypatch.setattr(service, "_call_serverless", fake_call)

    with pytest.raises(RuntimeError, match="CUDA"):
        await service.transcribe_segments(_prepared(), [Segment(0, 1000)])
