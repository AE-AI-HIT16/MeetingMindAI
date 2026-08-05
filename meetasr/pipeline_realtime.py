"""Realtime ASR pipeline: VAD + ASR only."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from meetasr.schemas import Segment, SentenceInfo, TranscriptResult
from meetasr.utils.audio import load_audio

VAD_PADDING_MS = 100
SAMPLE_RATE = 16000


class ASRPipeline:
    """
    Realtime pipeline.

    Pipeline:
        1. VAD
        2. ASR

    Rule:
        1 VAD segment = 1 SentenceInfo
    """


    def __init__(
        self,
        asr_model,
        vad_model=None,
        device: str = "cpu",
    ):
        self.asr = asr_model
        self.vad = vad_model
        self.device = device


    # -------------------------------------------------------------
    # Full transcription API
    # -------------------------------------------------------------

    def transcribe(
        self,
        audio_source,
        key: Optional[str] = None,
        language: str = "auto",
        **kwargs,
    ) -> TranscriptResult:

        audio = load_audio(audio_source)

        duration = len(audio) / SAMPLE_RATE

        segments = self._run_vad(audio)

        sentence_info = []

        for idx, segment in enumerate(segments):
            logging.warning(
                "DEBUG PROCESS VAD[%s] start=%s end=%s duration=%sms",
                idx,
                segment.start_ms,
                segment.end_ms,
                segment.end_ms - segment.start_ms,
            )

            sentences = self.transcribe_vad_segment(
                audio,
                segment,
                language=language,
                **kwargs,
            )

            logging.warning(
                "DEBUG ASR OUTPUT FOR VAD[%s]: %s",
                idx,
                [
                    {
                        "text": s.text,
                        "start": s.start,
                        "end": s.end,
                    }
                    for s in sentences
                ],
            )

            sentence_info.extend(sentences)


        text = " ".join(
            s.text
            for s in sentence_info
        )

        logging.warning(
            "DEBUG FINAL SENTENCE_INFO count=%s data=%s",
            len(sentence_info),
            [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                }
                for s in sentence_info
            ],
        )


        return TranscriptResult(
            key=key or "realtime",
            text=text,
            duration=duration,
            sentence_info=sentence_info,
        )


    # -------------------------------------------------------------
    # Incremental API
    # -------------------------------------------------------------

    def prepare_incremental_transcription(
        self,
        audio_source,
    ):

        audio = load_audio(audio_source)

        duration_ms = int(
            len(audio) / SAMPLE_RATE * 1000
        )

        return (
            audio,
            self._run_vad(audio),
            duration_ms,
        )


    def transcribe_vad_segment(
        self,
        audio: np.ndarray,
        segment: Segment,
        language: str = "auto",
        **kwargs,
    ) -> list[SentenceInfo]:

        total_ms = int(
            len(audio) / SAMPLE_RATE * 1000
        )


        start_ms = max(
            0,
            segment.start_ms - VAD_PADDING_MS,
        )

        end_ms = min(
            total_ms,
            segment.end_ms + VAD_PADDING_MS,
        )


        start = int(
            start_ms / 1000 * SAMPLE_RATE
        )

        end = int(
            end_ms / 1000 * SAMPLE_RATE
        )


        chunk = audio[start:end]


        if len(chunk) == 0:
            return []


        # IMPORTANT:
        # Không dùng recognize_long_form
        # vì ASR sẽ tự chia câu.
        results = self.asr.recognize(
            [chunk],
            language=language,
            **kwargs,
        )


        if not results:
            return []


        result = results[0]

        return [
            SentenceInfo(
                text=result.get(
                    "text",
                    "",
                ).strip(),

                start=segment.start_ms / 1000,

                end=segment.end_ms / 1000,

                speaker=None,

                char_timestamps=result.get(
                    "char_timestamps",
                    [],
                ),
            )
        ]


    def finalize_incremental_transcript(
        self,
        audio: np.ndarray,
        sentence_info: list[SentenceInfo],
        vad_segments: list[Segment],
    ):
        return sentence_info



    # -------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------

    def _run_vad(
        self,
        audio: np.ndarray,
    ) -> list[Segment]:

        if self.vad is None:

            duration_ms = int(
                len(audio) / SAMPLE_RATE * 1000
            )

            return [
                Segment(
                    0,
                    duration_ms,
                )
            ]


        t0 = time.perf_counter()


        segments = self.vad.detect(
            audio
        )

        logging.warning(
            "DEBUG VAD RESULT count=%s segments=%s",
            len(segments),
            [
                {
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "duration_ms": s.end_ms - s.start_ms,
                }
                for s in segments
            ],
        )

        segments = self.vad.detect(audio)

        logging.warning(
            "RAW VAD=%s",
            [
                {
                    "start": s.start_ms,
                    "end": s.end_ms,
                }
                for s in segments
            ],
        )

        # segments = merge_vad_segments(segments)

        logging.warning(
            "MERGED VAD=%s",
            [
                {
                    "start": s.start_ms,
                    "end": s.end_ms,
                }
                for s in segments
            ],
        )


        logging.info(
            "Realtime VAD: %s segments %.2fs",
            len(segments),
            time.perf_counter() - t0,
        )


        return segments
