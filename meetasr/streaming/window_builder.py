"""Bridge endpointed VAD utterances into the confirmed ASR queue."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meetasr.streaming.streaming_vad_types import SpeechUtterance, UtteranceReason


@dataclass(frozen=True, slots=True)
class ASRWindow:
    """An ASR input that retains its original session coordinates."""

    audio: np.ndarray
    start_ms: int
    end_ms: int
    reason: UtteranceReason


class SegmentWindowBuilder:
    """Enqueue each endpointed utterance immediately for confirmed ASR."""

    def __init__(self, session: object) -> None:
        self.session = session

    async def process(self) -> None:
        while self.session.ready_segments:
            utterance: SpeechUtterance = self.session.ready_segments.popleft()
            coverage = getattr(self.session, "coverage", None)
            if coverage is not None:
                coverage.record_speech(
                    utterance.start_ms,
                    utterance.end_ms,
                )
            await self.session.asr_queue.put(
                ASRWindow(
                    audio=utterance.audio,
                    start_ms=utterance.start_ms,
                    end_ms=utterance.end_ms,
                    reason=utterance.reason,
                )
            )

    async def flush(self, *, force: bool = False) -> None:
        """Compatibility no-op: the VAD already emits complete windows."""
        del force
