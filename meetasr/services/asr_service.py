"""Async adapter around the synchronous MeetPipeline transcription API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np

from meetasr.api.schemas_phase2 import TranscriptSegmentPayload
from meetasr.schemas import Segment, SentenceInfo, SpeakerTurn


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

    def __init__(self, pipeline: Any) -> None:
        if pipeline is None:
            raise ValueError("pipeline is required")
        self.pipeline = pipeline
        self._transcribe_lock = asyncio.Lock()

    async def transcribe(
        self,
        audio_source: Any,
        *,
        offset_ms: int = 0,
        key: str | None = None,
    ) -> ASRServiceResult:
        # Upload jobs and live-mic sessions share this service. Serialize access
        # because the underlying pipeline/GPU is not safe to run concurrently.
        async with self._transcribe_lock:
            result = await asyncio.to_thread(
                self.pipeline.transcribe,
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
        """Decode audio and optionally build diarization-first ASR turns."""
        async with self._transcribe_lock:
            if getattr(self.pipeline, "diarization_first", False):
                (
                    audio,
                    vad_segments,
                    speaker_turns,
                    duration_ms,
                ) = await asyncio.to_thread(
                    self.pipeline.prepare_diarization_first_transcription,
                    audio_source,
                )
            else:
                audio, vad_segments, duration_ms = await asyncio.to_thread(
                    self.pipeline.prepare_incremental_transcription,
                    audio_source,
                )
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
            getattr(self.pipeline, "transcription_language", "auto")
            if language == "auto"
            else language
        )
        async with self._transcribe_lock:
            return await asyncio.to_thread(
                self.pipeline.transcribe_vad_segment,
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
        """Finalize preassigned turns or run legacy ASR-first diarization."""
        async with self._transcribe_lock:
            if prepared.speaker_turns is not None:
                return await asyncio.to_thread(
                    self.pipeline.finalize_preassigned_transcript,
                    sentences,
                )
            return await asyncio.to_thread(
                self.pipeline.finalize_incremental_transcript,
                prepared.audio,
                sentences,
                prepared.vad_segments,
            )
