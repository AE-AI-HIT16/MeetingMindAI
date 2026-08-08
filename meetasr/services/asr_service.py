
from __future__ import annotations


import asyncio
import numpy as np
from typing import Any

from meetasr.api.schemas_phase2 import TranscriptSegmentPayload, SentenceInfo, Segment, SpeakerTurn
from dataclasses import dataclass
from meetasr.backend.services.runpod_client import RunPodClient

@dataclass(frozen=True)
class ASRServiceResult:
    """Normalized result consumed by both live and upload workers."""

    segments: list[TranscriptSegmentPayload]
    text: str
    duration_ms: int


@dataclass(frozen=True)
class PreparedTranscription:
    """Audio decoded once and its full-file VAD timeline."""

    audio: np.ndarray
    vad_segments: list[Segment]
    duration_ms: int
    speaker_turns: list[SpeakerTurn] | None = None


class ASRService:
    """Run one shared pipeline off the event loop and normalize its result."""

    def __init__(self, pipeline: Any = None) -> None:
        self.client = pipeline if pipeline is not None else RunPodClient()
        self._transcribe_lock = asyncio.Lock()

    async def _call_client(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if asyncio.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        res = fn(*args, **kwargs)
        if asyncio.iscoroutine(res):
            return await res
        return res

    async def transcribe(
        self,
        audio_source: Any,
        *,
        offset_ms: int = 0,
        key: str | None = None,
    ) -> ASRServiceResult:
        """Upload jobs and live‑mic sessions share this service. Serialize access because the underlying pipeline/GPU is not safe to run concurrently."""
        async with self._transcribe_lock:
            result = await self._call_client(
                self.client.transcribe,
                audio_source,
                key=key,
            )

        segments = [
            TranscriptSegmentPayload(
                start_ms=offset_ms + int(sentence.start * 1000),
                end_ms=offset_ms + int(sentence.end * 1000),
                speaker=sentence.speaker,
                text=sentence.text,
            )
            for sentence in result.sentence_info
            if sentence.text.strip()
        ]

        duration_ms = max(0, int(result.duration * 1000))
        if not segments and result.text.strip():
            segments.append(
                TranscriptSegmentPayload(
                    start_ms=offset_ms,
                    end_ms=offset_ms + duration_ms,
                    speaker=None,
                    text=result.text.strip(),
                )
            )

        return ASRServiceResult(
            segments=segments,
            text=result.text,
            duration_ms=duration_ms,
        )

    async def prepare_incremental(
        self,
        audio_source: Any,
    ) -> PreparedTranscription:
        """Decode audio and optionally build diarization‑first ASR turns."""
        async with self._transcribe_lock:
            if getattr(self.client, "diarization_first", False):
                method = getattr(
                    self.client,
                    "prepare_diarization_first_transcription",
                    getattr(self.client, "prepare_incremental", None),
                )
                (
                    audio,
                    vad_segments,
                    speaker_turns,
                    duration_ms,
                ) = await self._call_client(method, audio_source)
            else:
                method = getattr(
                    self.client,
                    "prepare_incremental_transcription",
                    getattr(self.client, "prepare_incremental", None),
                )
                res = await self._call_client(method, audio_source)
                if len(res) == 4:
                    audio, vad_segments, speaker_turns, duration_ms = res
                else:
                    audio, vad_segments, duration_ms = res
                    speaker_turns = None
        return PreparedTranscription(
            audio=audio,
            vad_segments=vad_segments,
            duration_ms=duration_ms,
            speaker_turns=speaker_turns,
        )

    async def transcribe_segment(
        self,
        prepared: PreparedTranscription,
        segment: Segment,
        *,
        language: str = "auto",
        key: str | None = None,
    ) -> list[SentenceInfo]:
        """Run ASR for one VAD segment while preserving global timestamps."""
        kwargs = {"key": key} if key is not None else {}
        effective_language = (
            getattr(self.client, "transcription_language", "auto")
            if language == "auto"
            else language
        )
        method = getattr(
            self.client,
            "transcribe_vad_segment",
            getattr(self.client, "transcribe_segment", None),
        )
        async with self._transcribe_lock:
            return await self._call_client(
                method,
                prepared.audio,
                segment,
                effective_language,
                **kwargs,
            )

    async def finalize_incremental(
        self,
        prepared: PreparedTranscription,
        sentences: list[SentenceInfo],
    ) -> list[SentenceInfo]:
        """Finalize preassigned turns or run legacy ASR‑first diarization."""
        async with self._transcribe_lock:
            if getattr(self.client, "finalize_preassigned_transcript", None):
                return await self._call_client(
                    self.client.finalize_preassigned_transcript,
                    sentences,
                )
            if getattr(self.client, "finalize_incremental_transcript", None):
                return await self._call_client(
                    self.client.finalize_incremental_transcript,
                    prepared.audio,
                    sentences,
                    prepared.vad_segments,
                )
            return await self._call_client(
                self.client.finalize_incremental,
                prepared,
                sentences,
            )


