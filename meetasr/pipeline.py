"""MeetPipeline — the core ASR + Speaker + LLM orchestrator."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np

from meetasr.schemas import TranscriptResult, MeetingReport, SentenceInfo, Segment
from meetasr.utils.audio import load_audio
from meetasr.utils.timestamp import merge_vad_segments, build_sentence_info
from meetasr.utils.download import download_model
from meetasr.utils.misc import deep_update


class MeetPipeline:
    """End-to-end meeting processing pipeline.

    Pipeline order:
        1. VAD   — detect speech segments
        2. ASR   — transcribe each segment
        3. Punc  — restore punctuation (optional)
        4. SPK   — speaker diarization (optional)
        5. LLM   — summarize, extract topics/actions/decisions (optional)

    Use AutoPipeline.from_config() for easy construction from a config dict.
    """

    def __init__(
        self,
        asr_model,
        vad_model=None,
        punc_model=None,
        spk_model=None,
        llm_summarizer=None,
        device: str = "cpu",
    ):
        """Initialize MeetPipeline with pre-built model instances.

        Args:
            asr_model: AbsASR instance (required).
            vad_model: AbsVAD instance. If None, treats entire audio as one segment.
            punc_model: AbsPunc instance. If None, skips punctuation step.
            spk_model: AbsSpk instance. If None, skips speaker diarization.
            llm_summarizer: MeetingSummarizer instance. If None, skips LLM step.
            device: Torch device string.
        """
        self.asr = asr_model
        self.vad = vad_model
        self.punc = punc_model
        self.spk = spk_model
        self.summarizer = llm_summarizer
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_source,
        key: Optional[str] = None,
        language: str = "auto",
        **kwargs,
    ) -> TranscriptResult:
        """Transcribe audio to text with speaker labels and timestamps.

        Args:
            audio_source: File path (str), URL, bytes, or np.ndarray.
            key: Identifier for this audio (default: filename stem).
            language: Language hint ("auto", "vi", "zh", "en", etc.)
            **kwargs: Extra params passed to ASR model.

        Returns:
            TranscriptResult with text, sentence_info, duration.
        """
        if key is None:
            key = _derive_key(audio_source)

        audio = load_audio(audio_source)
        duration = len(audio) / 16000.0

        # Step 1: VAD
        segments = self._run_vad(audio)

        # Step 2: ASR per segment
        asr_results = self._run_asr(audio, segments, language=language, **kwargs)

        # Step 3: Build sentence_info
        sentence_info = build_sentence_info(asr_results, segments)

        # Step 4: Punctuation
        if self.punc is not None:
            sentence_info = self._run_punc(sentence_info)

        # Step 5: Speaker diarization
        if self.spk is not None:
            sentence_info = self._run_spk(audio, sentence_info, segments)

        full_text = " ".join(s.text for s in sentence_info)

        return TranscriptResult(
            key=key,
            text=full_text,
            duration=duration,
            sentence_info=sentence_info,
        )

    def summarize_meeting(
        self,
        audio_source,
        key: Optional[str] = None,
        language: str = "vi",
        **kwargs,
    ) -> MeetingReport:
        """Full pipeline: transcribe + LLM summarization.

        Args:
            audio_source: File path, URL, bytes, or np.ndarray.
            key: Identifier for this audio.
            language: Output language for LLM ("vi" or "en").
            **kwargs: Passed to transcribe().

        Returns:
            MeetingReport with transcript + LLM analysis.

        Raises:
            RuntimeError: If no LLM summarizer is configured.
        """
        if self.summarizer is None:
            raise RuntimeError(
                "No LLM summarizer configured. "
                "Pass llm_summarizer to MeetPipeline or configure 'llm' in config."
            )
        transcript = self.transcribe(audio_source, key=key, **kwargs)
        return self.summarizer.summarize(transcript)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _run_vad(self, audio: np.ndarray) -> list[Segment]:
        """Run VAD or return single full-audio segment if no VAD model."""
        if self.vad is None:
            duration_ms = int(len(audio) / 16000.0 * 1000)
            return [Segment(0, duration_ms)]

        t0 = time.perf_counter()
        segments = self.vad.detect(audio)
        segments = merge_vad_segments(segments)
        logging.info(
            f"VAD: {len(segments)} segments detected "
            f"({time.perf_counter() - t0:.2f}s)"
        )
        if not segments:
            logging.warning("VAD found no speech segments — audio may be silent.")
        return segments

    def _run_asr(
        self,
        audio: np.ndarray,
        segments: list[Segment],
        **kwargs,
    ) -> list[dict]:
        """Run ASR on each VAD segment."""
        if not segments:
            return []

        t0 = time.perf_counter()
        # Slice audio for each segment
        chunks = []
        for seg in segments:
            start = int(seg.start_ms / 1000.0 * 16000)
            end = int(seg.end_ms / 1000.0 * 16000)
            chunk = audio[start:end]
            if len(chunk) > 0:
                chunks.append(chunk)

        results = self.asr.recognize(chunks, **kwargs)
        logging.info(
            f"ASR: {len(results)} results "
            f"({time.perf_counter() - t0:.2f}s)"
        )
        return results

    def _run_punc(self, sentence_info: list[SentenceInfo]) -> list[SentenceInfo]:
        """Restore punctuation for each sentence."""
        t0 = time.perf_counter()
        for s in sentence_info:
            try:
                s.text = self.punc.restore(s.text)
            except Exception as e:
                logging.warning(f"Punc failed for '{s.text[:30]}...': {e}")
        logging.info(f"Punc: done ({time.perf_counter() - t0:.2f}s)")
        return sentence_info

    def _run_spk(
        self,
        audio: np.ndarray,
        sentence_info: list[SentenceInfo],
        segments: list[Segment],
    ) -> list[SentenceInfo]:
        """Assign speaker labels via embedding + clustering."""
        import torch
        t0 = time.perf_counter()
        try:
            embeddings = []
            for seg in segments:
                start = int(seg.start_ms / 1000.0 * 16000)
                end = int(seg.end_ms / 1000.0 * 16000)
                chunk = audio[start:end]
                emb = self.spk.embed(chunk)
                embeddings.append(emb)

            all_embs = torch.cat(embeddings, dim=0)
            labels = self.spk.cluster(all_embs)

            for s, label in zip(sentence_info, labels):
                s.speaker = label

            n_speakers = len(set(labels))
            logging.info(
                f"SPK: {n_speakers} speaker(s) detected "
                f"({time.perf_counter() - t0:.2f}s)"
            )
        except Exception as e:
            logging.warning(f"Speaker diarization failed: {e}. Proceeding without speaker labels.")

        return sentence_info


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _derive_key(source) -> str:
    """Derive a human-readable key from the audio source."""
    if isinstance(source, str):
        return os.path.splitext(os.path.basename(source))[0]
    return f"audio_{int(time.time())}"
