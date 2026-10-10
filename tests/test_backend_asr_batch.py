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


@pytest.mark.asyncio
async def test_overlapped_turn_carries_relative_overlaps_and_voice_profile(monkeypatch):
    from meetasr.backend.api.schemas_phase2 import SpeakerTurn

    sent = {}

    async def fake_call(action, payload, timeout=600.0):
        sent.update(payload)
        return [[] for _ in payload["segments"]]

    service = _service()
    monkeypatch.setattr(service, "_call_serverless", fake_call)
    prepared = _prepared()
    prepared.speaker_profiles = {1: [0.1, 0.2]}
    turns = [
        SpeakerTurn(0, 2000, 0),
        SpeakerTurn(3000, 6000, 1, overlaps=[(3500, 4200)]),
    ]

    await service.transcribe_segments(
        prepared, [t.to_segment() for t in turns], context="OKR", turns=turns
    )

    assert sent["context"] == "OKR"
    assert isinstance(sent["segments"][0], str)  # no overlap: plain WAV
    assert sent["segments"][1]["overlaps"] == [[500, 1200]]
    assert sent["segments"][1]["speaker_embedding"] == [0.1, 0.2]


def test_diarization_audio_is_lossless_flac_when_it_fits():
    from meetasr.backend.services.asr_service import numpy_to_diarization_audio_bytes

    audio = (np.random.default_rng(0).normal(0, 0.05, 16000 * 60)).astype(np.float32)
    assert numpy_to_diarization_audio_bytes(audio)[:4] == b"fLaC"


def test_long_diarization_audio_uses_opus_within_runpod_limit(monkeypatch):
    from meetasr.backend.services import asr_service

    # Pretend FLAC is too big so the Opus path is exercised on a short clip.
    monkeypatch.setattr(asr_service, "MAX_PREPARE_AUDIO_BYTES", 200_000)
    audio = (np.random.default_rng(0).normal(0, 0.05, 16000 * 40)).astype(np.float32)
    encoded = asr_service.numpy_to_diarization_audio_bytes(audio)
    assert b"OpusHead" in encoded[:200]
    assert len(encoded) <= 200_000
