"""MeetPipeline — the core ASR + Speaker + LLM orchestrator."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional
import torch

import numpy as np

from meetasr.schemas import TranscriptResult, MeetingReport, SentenceInfo, Segment
from meetasr.utils.audio import load_audio
from meetasr.utils.timestamp import merge_vad_segments, build_sentence_info
from meetasr.utils.download import download_model
from meetasr.utils.misc import deep_update
from meetasr.utils.diarization import chunk_segment, circle_pad, assign_speakers_by_overlap, compressed_seg, map_chars_to_speakers
from meetasr.utils.diarization import (chunk_segment, circle_pad, assign_speakers_by_overlap, compressed_seg,map_chars_to_speakers, split_at_speaker_turns,)

VAD_PADDING_MS = 100
SAMPLE_RATE = 16000


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
        doc_planner=None,
        device: str = "cpu",
    ):
        """Initialize MeetPipeline with pre-built model instances.

        Args:
            asr_model: AbsASR instance (required).
            vad_model: AbsVAD instance. If None, treats entire audio as one segment.
            punc_model: AbsPunc instance. If None, skips punctuation step.
            spk_model: AbsSpk instance. If None, skips speaker diarization.
            llm_summarizer: MeetingSummarizer instance. If None, skips LLM step.
            doc_planner: DocumentPlanner instance for Phase 2 structured documents.
            device: Torch device string.
        """
        self.asr = asr_model
        self.vad = vad_model
        self.punc = punc_model
        self.spk = spk_model
        self.summarizer = llm_summarizer
        self.doc_planner = doc_planner
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
        duration = len(audio) / SAMPLE_RATE

        # Step 1: VAD
        segments = self._run_vad(audio)

        # Step 2: ASR per segment
        asr_results = self._run_asr(audio, segments, language=language, **kwargs)

        # Step 3: Build sentence_info
        sentence_info = build_sentence_info(asr_results, segments)

        # Step 4: Speaker diarization
        # Must run BEFORE punctuation because punctuation changes text length,
        # which would break char_timestamps alignment used for splitting speakers.
        if self.spk is not None:
            sentence_info = self._run_spk(audio, sentence_info, segments)

        # Step 5: Punctuation
        if self.punc is not None:
            sentence_info = self._run_punc(sentence_info)

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
            duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
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
        total_ms = int(len(audio) / SAMPLE_RATE * 1000)
        for seg in segments:
            start_ms = max(0, seg.start_ms - VAD_PADDING_MS)
            end_ms = min(total_ms, seg.end_ms + VAD_PADDING_MS)
            start = int(start_ms / 1000.0 * SAMPLE_RATE)
            end = int(end_ms / 1000.0 * SAMPLE_RATE)
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
        """Restore punctuation for each sentence and align timestamps."""
        from meetasr.utils.timestamp import align_punctuated_timestamps
        t0 = time.perf_counter()
        for s in sentence_info:
            try:
                raw_text = s.text
                punc_text = self.punc.restore(raw_text)
                if punc_text != raw_text:
                    s.char_timestamps = align_punctuated_timestamps(
                        raw_text, punc_text, s.char_timestamps
                    )
                    s.text = punc_text
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
        """Assign speaker labels via embedding + clustering.

        Implements 3D-Speaker diarization pipeline (T1 + T3 + T4):
            T1: Sub-segmentation — each VAD segment → 1.5s chunks (0.75s step)
            T3: Post-processing  — compressed_seg merges adjacent same-speaker chunks
            T4: Alignment        — assign speaker per sentence by overlap duration
        """
        t0 = time.perf_counter()
        try:
            sample_rate = 16000
            target_len = int(1.5 * sample_rate)  # 24,000 frames

            # T1: Sub-segmentation — collect all chunks across all VAD segments
            all_chunks = []
            for seg in segments:
                all_chunks.extend(chunk_segment(seg.start_s, seg.end_s, dur=1.5, step=0.75))

            if not all_chunks:
                logging.warning("SPK: no chunks produced — skipping diarization.")
                return sentence_info

            # T1: Extract embedding per chunk (sequential, memory-safe)
            embeddings = []
            for st, ed in all_chunks:
                chunk_np = audio[int(st * sample_rate):int(ed * sample_rate)]
                if len(chunk_np) < target_len:
                    t = torch.from_numpy(chunk_np).float()
                    chunk_np = circle_pad(t, target_len).numpy()
                emb = self.spk.embed(chunk_np.astype(np.float32))
                embeddings.append(emb)

            # Cluster all chunk embeddings
            all_embs = torch.cat(embeddings, dim=0)
            labels = self.spk.cluster(all_embs)

            # T3: Build diar segments + merge adjacent same-speaker chunks
            diar_segs = [[c[0], c[1], int(l)] for c, l in zip(all_chunks, labels)]
            diar_segs = compressed_seg(diar_segs)

            # T4: Word-level speaker attribution 
            new_sentence_info = []
            for sent in sentence_info:
                if sent.char_timestamps:
                    char_speakers = map_chars_to_speakers(
                        sent.char_timestamps, diar_segs,
                    )
                    sub_sents = split_at_speaker_turns(sent, char_speakers)
                    new_sentence_info.extend(sub_sents)
                else:
                    # Fallback: sentence-level (models không có timestamps, vd Zipformer)
                    assign_speakers_by_overlap([sent], diar_segs)
                    new_sentence_info.append(sent)

            # Remap speaker IDs chronologically based on their first appearance
            spk_mapping = {}
            for sent in new_sentence_info:
                if sent.speaker is not None:
                    if sent.speaker not in spk_mapping:
                        spk_mapping[sent.speaker] = len(spk_mapping)
                    sent.speaker = spk_mapping[sent.speaker]

            sentence_info = new_sentence_info

            n_speakers = len(spk_mapping)
            logging.info(
                f"SPK: {n_speakers} speaker(s), {len(all_chunks)} chunks "
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
