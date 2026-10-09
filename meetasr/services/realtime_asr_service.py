from __future__ import annotations

import asyncio
from typing import Any

from meetasr.api.schemas_phase2 import TranscriptSegmentPayload
from meetasr.services.asr_service import ASRServiceResult
from meetasr.services.inference_coordinator import (
    InferenceCoordinator,
    InferenceKind,
)


class RealtimeASRService:
    """
    Adapter cho realtime ASRPipeline.

    ASRPipeline và MeetPipeline dùng chung output contract:
        TranscriptResult
            |
            v
        SentenceInfo
            |
            v
        TranscriptSegmentPayload
    """


    def __init__(
        self,
        pipeline: Any,
        coordinator: InferenceCoordinator | None = None,
    ):
        if pipeline is None:
            raise ValueError(
                "pipeline is required"
            )

        self.pipeline = pipeline
        self.coordinator = coordinator

        self._lock = asyncio.Lock()



    async def transcribe(
        self,
        audio,
        *,
        offset_ms: int = 0,
        key: str | None = None,
    ) -> ASRServiceResult:


        if self.coordinator is not None:
            result = await self.coordinator.submit(
                InferenceKind.CONFIRMED,
                self.pipeline.transcribe,
                audio,
                key=key,
                language=getattr(
                    self.pipeline,
                    "transcription_language",
                    "auto",
                ),
                skip_vad=True,
            )
        else:
            async with self._lock:
                result = await asyncio.to_thread(
                    self.pipeline.transcribe,
                    audio,
                    key=key,
                    language=getattr(
                        self.pipeline,
                        "transcription_language",
                        "auto",
                    ),
                    skip_vad=True,
                )

        segments = []

        for sentence in result.sentence_info:

            if not sentence.text.strip():
                continue


            segments.append(
                TranscriptSegmentPayload(
                    start_ms=(
                        offset_ms
                        + int(sentence.start * 1000)
                    ),

                    end_ms=(
                        offset_ms
                        + int(sentence.end * 1000)
                    ),

                    speaker=sentence.speaker,

                    text=sentence.text.strip(),
                )
            )

        duration_ms = max(
            0,
            int(result.duration * 1000),
        )


        return ASRServiceResult(
            segments=segments,

            text=result.text,

            duration_ms=duration_ms,
        )
