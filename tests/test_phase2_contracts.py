import asyncio
import httpx
import numpy as np
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from meetasr.api.schemas_phase2 import (
    DoneEvent,
    SpeakerAssignment,
    SpeakerUpdateEvent,
    StatusEvent,
    TranscriptDeltaEvent,
    TranscriptSegmentPayload,
)
from meetasr.db.models_phase2 import TranscriptSegment
from meetasr.realtime.events import EventBus
from meetasr.schemas import Segment, SentenceInfo, SpeakerTurn, TranscriptResult
from meetasr.services import asr_service
from meetasr.services.asr_service import ASRService, ASRServiceResult
from meetasr.streaming.asr_worker import ASRWorker


def test_transcript_segment_payload_from_db() -> None:
    segment = TranscriptSegment(
        id=15,
        job_id="job-456",
        start_ms=12000,
        end_ms=15800,
        speaker=None,
        text="Hôm nay chúng ta sẽ bàn về dự án.",
    )

    payload = TranscriptSegmentPayload.from_db(segment)

    assert payload.model_dump() == {
        "id": 15,
        "start_ms": 12000,
        "end_ms": 15800,
        "speaker": None,
        "text": "Hôm nay chúng ta sẽ bàn về dự án.",
    }


def test_transcript_segment_rejects_invalid_time_range() -> None:
    with pytest.raises(
        ValidationError,
        match="end_ms must be greater",
    ):
        TranscriptSegmentPayload(
            start_ms=10000,
            end_ms=5000,
            speaker=None,
            text="Sai timestamp",
        )


def test_transcript_segment_rejects_negative_speaker() -> None:
    with pytest.raises(ValidationError):
        TranscriptSegmentPayload(
            start_ms=0,
            end_ms=1000,
            speaker=-1,
            text="Sai speaker",
        )


def test_job_event_contracts_serialize_expected_discriminators() -> None:
    segment = TranscriptSegmentPayload(
        start_ms=0,
        end_ms=1000,
        speaker=None,
        text="Xin chào",
    )
    assert StatusEvent(stage="transcribing", progress=0.5).model_dump() == {
        "type": "status",
        "stage": "transcribing",
        "progress": 0.5,
    }
    assert TranscriptDeltaEvent(segment=segment).model_dump()["type"] == (
        "transcript_delta"
    )
    assert DoneEvent(duration_ms=1000, num_segments=1).model_dump()["type"] == (
        "done"
    )
    assert SpeakerUpdateEvent(
        updates=[SpeakerAssignment(segment_id=15, speaker=1)]
    ).model_dump() == {
        "type": "speaker_update",
        "updates": [{"segment_id": 15, "speaker": 1}],
    }


@pytest.mark.asyncio
async def test_event_bus_fans_out_and_keeps_queue_bounded() -> None:
    bus = EventBus(max_queue_size=1)
    first = await bus.subscribe("job-1")
    second = await bus.subscribe("job-1")

    await bus.publish("job-1", {"type": "status", "progress": 0.1})
    await bus.publish("job-1", {"type": "status", "progress": 0.2})

    assert (await first.get())["progress"] == 0.2
    assert (await second.get())["progress"] == 0.2
    await bus.unsubscribe("job-1", first)
    await bus.unsubscribe("job-1", second)


@pytest.mark.asyncio
async def test_live_mic_sends_canonical_transcript_delta_segment() -> None:
    sent: list[dict] = []

    class FakeWebSocket:
        async def send_json(self, payload):
            sent.append(payload)

    worker = ASRWorker(
        SimpleNamespace(websocket=FakeWebSocket()),
        asr_service=None,
    )
    result = ASRServiceResult(
        segments=[
            TranscriptSegmentPayload(
                start_ms=1000,
                end_ms=2500,
                speaker=None,
                text="Xin chào",
            )
        ],
        text="Xin chào",
        duration_ms=1500,
    )

    await worker._send_result(result)

    assert sent == [
        {
            "type": "transcript_delta",
            "segment": {
                "id": None,
                "start_ms": 1000,
                "end_ms": 2500,
                "speaker": None,
                "text": "Xin chào",
            },
        }
    ]


def test_asr_service_normalizes_pipeline_result(monkeypatch) -> None:
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    # This environment cannot shut down asyncio's default thread executor.
    # Keep this unit test deterministic while still testing normalization.
    monkeypatch.setattr(asr_service.asyncio, "to_thread", run_inline)

    class StubPipeline:
        def transcribe(self, audio, *, key=None):
            assert audio == b"audio"
            assert key == "meeting.wav"
            return TranscriptResult(
                key=key,
                text="Xin chào",
                duration=2.5,
                sentence_info=[
                    SentenceInfo(
                        text="Xin chào",
                        start=0.25,
                        end=1.5,
                        speaker=1,
                    )
                ],
            )

    result = asyncio.run(
        ASRService(StubPipeline()).transcribe(
            b"audio",
            offset_ms=10_000,
            key="meeting.wav",
        )
    )

    assert result.duration_ms == 2500
    assert result.segments[0].model_dump() == {
        "id": None,
        "start_ms": 10250,
        "end_ms": 11500,
        "speaker": 1,
        "text": "Xin chào",
    }


def test_asr_service_exposes_incremental_pipeline_steps(monkeypatch) -> None:
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asr_service.asyncio, "to_thread", run_inline)

    class StubPipeline:
        def prepare_incremental_transcription(self, audio_source):
            assert audio_source == "meeting.wav"
            return "decoded-audio", [Segment(1000, 2000)], 3000

        def transcribe_vad_segment(
            self,
            audio,
            segment,
            language="auto",
        ):
            assert audio == "decoded-audio"
            assert segment == Segment(1000, 2000)
            assert language == "vi"
            return [
                SentenceInfo(
                    text="Xin chào",
                    start=1.0,
                    end=2.0,
                )
            ]

        def finalize_incremental_transcript(
            self,
            audio,
            sentences,
            vad_segments,
        ):
            assert audio == "decoded-audio"
            assert vad_segments == [Segment(1000, 2000)]
            sentences[0].speaker = 0
            return sentences

    async def run_flow():
        service = ASRService(StubPipeline())
        prepared = await service.prepare_incremental("meeting.wav")
        sentences = await service.transcribe_segment(
            prepared,
            prepared.vad_segments[0],
            language="vi",
        )
        finalized = await service.finalize_incremental(prepared, sentences)
        return prepared, finalized

    prepared, finalized = asyncio.run(run_flow())

    assert prepared.duration_ms == 3000
    assert finalized[0].speaker == 0


def test_asr_service_uses_diarization_first_turns_and_default_language(
    monkeypatch,
) -> None:
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asr_service.asyncio, "to_thread", run_inline)

    class StubPipeline:
        diarization_first = True
        transcription_language = "vi"

        def prepare_diarization_first_transcription(self, audio_source):
            assert audio_source == "meeting.wav"
            return (
                "decoded-audio",
                [Segment(0, 4000)],
                [SpeakerTurn(500, 2500, 1)],
                4000,
            )

        def transcribe_vad_segment(
            self,
            audio,
            segment,
            language,
            **kwargs,
        ):
            assert audio == "decoded-audio"
            assert segment == Segment(500, 2500)
            assert language == "vi"
            return [SentenceInfo(text="Xin chào", start=0.5, end=2.5)]

        def finalize_preassigned_transcript(self, sentences):
            assert sentences[0].speaker == 1
            return sentences

    async def run_flow():
        service = ASRService(StubPipeline())
        prepared = await service.prepare_incremental("meeting.wav")
        assert prepared.speaker_turns is not None
        turn = prepared.speaker_turns[0]
        sentences = await service.transcribe_segment(
            prepared,
            turn.to_segment(),
        )
        sentences[0].speaker = turn.speaker
        return await service.finalize_incremental(prepared, sentences)

    finalized = asyncio.run(run_flow())

    assert finalized[0].speaker == 1


def test_backend_asr_service_retries_on_503(monkeypatch) -> None:
    from meetasr.backend.services.asr_service import ASRService as BackendASRService

    service = BackendASRService(runpod_url="http://localhost:8001")
    service._ensure_local_port_8001 = lambda: None  # No-op in test

    calls = 0

    class MockResponse:
        def __init__(self, status_code: int, data: dict):
            self.status_code = status_code
            self._data = data
            self.request = httpx.Request("POST", "http://localhost:8001/v1/prepare_incremental")

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "http://localhost:8001/v1/prepare_incremental")
                response = httpx.Response(self.status_code, json=self._data, request=request)
                raise httpx.HTTPStatusError("503 error", request=request, response=response)

        def json(self):
            return self._data

    async def mock_post(self_client, url, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return MockResponse(503, {"detail": "Pipeline not loaded"})
        return MockResponse(200, {
            "vad_segments": [{"start_ms": 0, "end_ms": 2000}],
            "duration_ms": 2000,
            "speaker_turns": None,
        })

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    async def run():
        audio = np.zeros(2 * 16000, dtype=np.float32)
        return await service.prepare_incremental(audio)

    prepared = asyncio.run(run())
    assert calls == 2
    assert prepared.duration_ms == 2000
    assert len(prepared.vad_segments) == 1


