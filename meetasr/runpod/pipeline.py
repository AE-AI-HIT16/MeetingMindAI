"""MeetPipeline — the core ASR + Speaker + LLM orchestrator."""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Optional

import numpy as np
import torch

from meetasr.runpod.schemas import (
    MeetingReport,
    Segment,
    SentenceInfo,
    SpeakerTurn,
    TranscriptResult,
)
from meetasr.runpod.utils.audio import load_audio
from meetasr.runpod.utils.diarization import (
    assign_speakers_by_overlap,
    build_speaker_turns,
    chunk_segment,
    circle_pad,
    compressed_seg,
    map_chars_to_speakers,
    split_at_speaker_turns,
)
from meetasr.runpod.utils.timestamp import (
    build_sentence_info,
    clip_sentence_to_range,
    find_speech_gaps,
    merge_rescued_sentences,
    merge_vad_segments,
    split_punctuated_sentence_info,
)

VAD_PADDING_MS = 100
SAMPLE_RATE = 16000
GAP_RESCUE_MIN_GAP_MS = 2000
GAP_RESCUE_RIGHT_CONTEXT_MS = 1000


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
        enable_gap_rescue: bool = False,
        diarization_first: bool = False,
        speaker_turn_max_chunk_ms: int = 15000,
        speaker_turn_boundary_search_ms: int = 2000,
        speaker_turn_min_chunk_ms: int = 1000,
        transcription_language: str = "auto",
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
            enable_gap_rescue: Re-decode VAD-confirmed speech gaps for ASR
                models with native long-form timestamps. Disabled by default.
            diarization_first: Build speaker turns before ASR for uploaded media.
            speaker_turn_max_chunk_ms: Maximum Qwen input duration per turn.
            speaker_turn_boundary_search_ms: Backward low-energy search window.
            speaker_turn_min_chunk_ms: Minimum tail duration when splitting turns.
            transcription_language: Default language used by upload jobs.
        """
        if speaker_turn_max_chunk_ms <= 0:
            raise ValueError("speaker_turn_max_chunk_ms must be positive")
        if speaker_turn_boundary_search_ms < 0:
            raise ValueError("speaker_turn_boundary_search_ms must be non-negative")
        if (
            speaker_turn_min_chunk_ms <= 0
            or speaker_turn_min_chunk_ms >= speaker_turn_max_chunk_ms
        ):
            raise ValueError(
                "speaker_turn_min_chunk_ms must be positive and smaller than "
                "speaker_turn_max_chunk_ms"
            )
        self.asr = asr_model
        self.vad = vad_model
        self.punc = punc_model
        self.spk = spk_model
        self.summarizer = llm_summarizer
        self.doc_planner = doc_planner
        self.device = device
        self.enable_gap_rescue = enable_gap_rescue
        self.diarization_first = diarization_first
        self.speaker_turn_max_chunk_ms = speaker_turn_max_chunk_ms
        self.speaker_turn_boundary_search_ms = speaker_turn_boundary_search_ms
        self.speaker_turn_min_chunk_ms = speaker_turn_min_chunk_ms
        self.transcription_language = transcription_language

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
        asr_results, asr_segments, timestamp_offsets_ms = self._run_asr(
            audio, segments, language=language, **kwargs
        )

        # Step 3: Build sentence_info
        sentence_info = build_sentence_info(
            asr_results,
            asr_segments,
            timestamp_offsets_ms=timestamp_offsets_ms,
        )

        if (
            self.enable_gap_rescue
            and self.vad is not None
            and getattr(self.asr, "uses_internal_vad", False)
        ):
            sentence_info = self._rescue_speech_gaps(
                audio,
                sentence_info,
                segments,
                duration_ms=int(duration * 1000),
                language=language,
                **kwargs,
            )

        # Step 4: Speaker diarization
        # Must run BEFORE punctuation because punctuation changes text length,
        # which would break char_timestamps alignment used for splitting speakers.
        if self.spk is not None:
            sentence_info = self._run_spk(audio, sentence_info, segments)

        # Step 5: Punctuation. Whisper already emits punctuation, so applying a
        # second punctuation model would alter its native transcript and timing.
        if (
            self.punc is not None
            and not getattr(self.asr, "has_native_punctuation", False)
        ):
            sentence_info = self._run_punc(sentence_info)
            sentence_info = split_punctuated_sentence_info(sentence_info)
        elif self.punc is not None:
            logging.info("Punc: skipped because ASR provides native punctuation")

        full_text = " ".join(s.text for s in sentence_info)

        return TranscriptResult(
            key=key,
            text=full_text,
            duration=duration,
            sentence_info=sentence_info,
        )

    def prepare_incremental_transcription(
        self,
        audio_source,
    ) -> tuple[np.ndarray, list[Segment], int]:
        """Decode audio once and find the VAD segments used by an upload Job."""
        audio = load_audio(audio_source)
        duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
        return audio, self._run_vad(audio), duration_ms

    def prepare_diarization_first_transcription(
        self,
        audio_source,
    ) -> tuple[np.ndarray, list[Segment], list[SpeakerTurn] | None, int]:
        """Decode audio, run VAD + diarization, and build ASR-ready turns.

        ``None`` speaker turns means the caller must use the legacy ASR-first
        fallback. This preserves transcript availability when diarization is
        disabled, produces no usable chunks, or fails.
        """
        audio = load_audio(audio_source)
        duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
        vad_segments = self._run_vad(audio)
        if not self.diarization_first or self.spk is None or not vad_segments:
            return audio, vad_segments, None, duration_ms

        try:
            diar_segments = self._diarize_segments(audio, vad_segments)
            if not diar_segments:
                logging.warning(
                    "SPK-first: no diarization segments; using ASR-first fallback."
                )
                return audio, vad_segments, None, duration_ms
            speaker_turns = build_speaker_turns(
                audio,
                diar_segments,
                max_chunk_ms=self.speaker_turn_max_chunk_ms,
                boundary_search_ms=self.speaker_turn_boundary_search_ms,
                min_chunk_ms=self.speaker_turn_min_chunk_ms,
                sample_rate=SAMPLE_RATE,
            )
            if not speaker_turns:
                logging.warning(
                    "SPK-first: no speaker turns; using ASR-first fallback."
                )
                return audio, vad_segments, None, duration_ms
            logging.info(
                "SPK-first: built %s ASR turn(s) from %s diarization segment(s).",
                len(speaker_turns),
                len(diar_segments),
            )
            return audio, vad_segments, speaker_turns, duration_ms
        except Exception as exc:
            logging.warning(
                "SPK-first preparation failed: %s. Using ASR-first fallback.",
                exc,
            )
            return audio, vad_segments, None, duration_ms

    def transcribe_vad_segment(
        self,
        audio: np.ndarray,
        segment: Segment,
        language: str = "auto",
        **kwargs,
    ) -> list[SentenceInfo]:
        """Transcribe one VAD segment and return timestamps on the full timeline.

        Models with native long-form VAD receive only this padded VAD chunk.
        Other wrappers use the normal batched ``recognize`` API with one chunk.
        Speaker diarization and external punctuation are intentionally deferred
        until all chunks have been transcribed.
        """
        if getattr(self.asr, "uses_internal_vad", False):
            total_ms = int(len(audio) / SAMPLE_RATE * 1000)
            start_ms = max(0, segment.start_ms - VAD_PADDING_MS)
            end_ms = min(total_ms, segment.end_ms + VAD_PADDING_MS)
            start = int(start_ms / 1000.0 * SAMPLE_RATE)
            end = int(end_ms / 1000.0 * SAMPLE_RATE)
            chunk = audio[start:end]
            if len(chunk) == 0:
                return []

            results = self.asr.recognize_long_form(
                chunk,
                language=language,
                **kwargs,
            )
            return build_sentence_info(
                results,
                [segment] * len(results),
                timestamp_offsets_ms=[start_ms] * len(results),
            )

        results, result_segments, offsets = self._run_asr(
            audio,
            [segment],
            language=language,
            **kwargs,
        )
        return build_sentence_info(
            results,
            result_segments,
            timestamp_offsets_ms=offsets,
        )

    def finalize_incremental_transcript(
        self,
        audio: np.ndarray,
        sentence_info: list[SentenceInfo],
        vad_segments: list[Segment],
    ) -> list[SentenceInfo]:
        """Assign global speakers and punctuation without changing segment IDs.

        The ordinary full-file pipeline may split a sentence at speaker turns.
        An incremental upload has already persisted each sentence, so this path
        projects the final diarization back onto those stable sentence records.
        """
        finalized = copy.deepcopy(sentence_info)

        if self.spk is not None and finalized:
            diarized = self._run_spk(
                audio,
                copy.deepcopy(finalized),
                vad_segments,
            )
            for sentence in finalized:
                candidates = [
                    (
                        min(sentence.end, candidate.end)
                        - max(sentence.start, candidate.start),
                        candidate.speaker,
                    )
                    for candidate in diarized
                    if candidate.speaker is not None
                    and min(sentence.end, candidate.end)
                    > max(sentence.start, candidate.start)
                ]
                if candidates:
                    sentence.speaker = max(candidates, key=lambda item: item[0])[1]

        if (
            self.punc is not None
            and not getattr(self.asr, "has_native_punctuation", False)
        ):
            finalized = self._run_punc(finalized)

        return finalized

    def finalize_preassigned_transcript(
        self,
        sentence_info: list[SentenceInfo],
    ) -> list[SentenceInfo]:
        """Restore punctuation without changing preassigned speaker turns."""
        finalized = copy.deepcopy(sentence_info)
        if self.punc is not None and (
            not getattr(self.asr, "has_native_punctuation", False)
            or _needs_external_punctuation(finalized)
        ):
            finalized = self._run_punc(finalized)
        return finalized

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
        max_segment_ms = getattr(self.vad, "max_segment_ms", 60000)
        segments = merge_vad_segments(segments, max_segment_ms=max_segment_ms)
        min_segment_ms = getattr(self.vad, "min_segment_ms", 200)
        segments = _limit_segment_duration(
            segments,
            max_segment_ms=max_segment_ms,
            min_segment_ms=min_segment_ms,
        )
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
    ) -> tuple[list[dict], list[Segment], list[int]]:
        """Run ASR on its preferred input shape and retain global offsets."""
        if getattr(self.asr, "uses_internal_vad", False):
            t0 = time.perf_counter()
            results = self.asr.recognize_long_form(audio, **kwargs)
            duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
            logging.info(
                "ASR long-form: %s result(s) (%0.2fs)",
                len(results),
                time.perf_counter() - t0,
            )
            return (
                results,
                [Segment(0, duration_ms)] * len(results),
                [0] * len(results),
            )

        if not segments:
            return [], [], []

        t0 = time.perf_counter()
        # Slice audio for each segment
        chunks = []
        chunk_segments = []
        timestamp_offsets_ms = []
        total_ms = int(len(audio) / SAMPLE_RATE * 1000)
        for seg in segments:
            start_ms = max(0, seg.start_ms - VAD_PADDING_MS)
            end_ms = min(total_ms, seg.end_ms + VAD_PADDING_MS)
            start = int(start_ms / 1000.0 * SAMPLE_RATE)
            end = int(end_ms / 1000.0 * SAMPLE_RATE)
            chunk = audio[start:end]
            if len(chunk) > 0:
                chunks.append(chunk)
                chunk_segments.append(seg)
                timestamp_offsets_ms.append(start_ms)

        results = self.asr.recognize(chunks, **kwargs)
        if len(results) != len(chunk_segments):
            logging.warning(
                "ASR returned %s results for %s chunks; truncating timestamp "
                "alignment to the available pairs.",
                len(results),
                len(chunk_segments),
            )
        pair_count = min(len(results), len(chunk_segments))
        logging.info(
            f"ASR: {len(results)} results "
            f"({time.perf_counter() - t0:.2f}s)"
        )
        return (
            results[:pair_count],
            chunk_segments[:pair_count],
            timestamp_offsets_ms[:pair_count],
        )

    def _rescue_speech_gaps(
        self,
        audio: np.ndarray,
        sentences: list[SentenceInfo],
        vad_segments: list[Segment],
        duration_ms: int,
        language: str,
        **kwargs,
    ) -> list[SentenceInfo]:
        """Re-decode only VAD-confirmed gaps in a long-form ASR timeline."""
        gaps = find_speech_gaps(
            sentences,
            vad_segments,
            duration_ms=duration_ms,
            min_gap_ms=GAP_RESCUE_MIN_GAP_MS,
        )
        rescued_sentences = []
        for index, gap in enumerate(gaps):
            input_end_ms = min(
                duration_ms,
                gap.end_ms + GAP_RESCUE_RIGHT_CONTEXT_MS,
            )
            audio_chunk = audio[
                int(gap.start_ms / 1000 * SAMPLE_RATE):
                int(input_end_ms / 1000 * SAMPLE_RATE)
            ]
            rescue_results = self.asr.recognize_long_form(
                audio_chunk,
                language=language,
                key=f"gap_rescue_{index}",
                **kwargs,
            )
            rescue_segments = [
                Segment(gap.start_ms, input_end_ms)
                for _ in rescue_results
            ]
            rescue_sentences = build_sentence_info(
                rescue_results,
                rescue_segments,
                timestamp_offsets_ms=[gap.start_ms] * len(rescue_results),
            )
            for sentence in rescue_sentences:
                clipped = clip_sentence_to_range(
                    sentence,
                    gap.start_ms,
                    gap.end_ms,
                )
                if clipped is not None:
                    rescued_sentences.append(clipped)

        if rescued_sentences:
            logging.info(
                "Gap rescue: inserted %s sentence(s) from %s candidate gap(s)",
                len(rescued_sentences),
                len(gaps),
            )
        return merge_rescued_sentences(sentences, rescued_sentences)

    def _run_punc(self, sentence_info: list[SentenceInfo]) -> list[SentenceInfo]:
        """Restore punctuation for each sentence and align timestamps."""
        from meetasr.runpod.utils.timestamp import align_punctuated_timestamps
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
            diar_segs = self._diarize_segments(audio, segments)
            if not diar_segs:
                logging.warning("SPK: no chunks produced — skipping diarization.")
                return sentence_info

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
                f"SPK: {n_speakers} speaker(s), {len(diar_segs)} turn(s) "
                f"({time.perf_counter() - t0:.2f}s)"
            )
        except Exception as e:
            logging.warning(f"Speaker diarization failed: {e}. Proceeding without speaker labels.")

        return sentence_info

    def _diarize_segments(
        self,
        audio: np.ndarray,
        segments: list[Segment],
    ) -> list[list]:
        """Return compressed ``[start_s, end_s, speaker_id]`` segments."""
        if self.spk is None or not segments:
            return []

        sample_rate = SAMPLE_RATE
        target_len = int(1.5 * sample_rate)
        all_chunks = []
        for seg in segments:
            all_chunks.extend(
                chunk_segment(seg.start_s, seg.end_s, dur=1.5, step=0.75)
            )
        if not all_chunks:
            return []

        embedded_chunks = []
        embeddings = []
        for st, ed in all_chunks:
            chunk_np = audio[int(st * sample_rate):int(ed * sample_rate)]
            if len(chunk_np) == 0:
                continue
            if len(chunk_np) < target_len:
                tensor = torch.from_numpy(chunk_np).float()
                chunk_np = circle_pad(tensor, target_len).numpy()
            embeddings.append(self.spk.embed(chunk_np.astype(np.float32)))
            embedded_chunks.append([st, ed])

        if not embeddings:
            return []
        all_embs = torch.cat(embeddings, dim=0)
        labels = self.spk.cluster(all_embs)
        usable_count = min(len(embedded_chunks), len(labels))
        if usable_count != len(embedded_chunks):
            logging.warning(
                "SPK returned %s labels for %s chunks; truncating.",
                len(labels),
                len(embedded_chunks),
            )
        diar_segs = [
            [chunk[0], chunk[1], int(label)]
            for chunk, label in zip(
                embedded_chunks[:usable_count],
                labels[:usable_count],
            )
        ]
        return compressed_seg(diar_segs)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _limit_segment_duration(
    segments: list[Segment],
    max_segment_ms: int,
    min_segment_ms: int,
) -> list[Segment]:
    """Split VAD output so ASR never receives an unexpectedly long chunk."""
    if max_segment_ms <= 0:
        return segments

    limited = []
    for segment in segments:
        start_ms = segment.start_ms
        while segment.end_ms - start_ms > max_segment_ms:
            limited.append(Segment(start_ms, start_ms + max_segment_ms))
            start_ms += max_segment_ms

        if segment.end_ms - start_ms < min_segment_ms and limited:
            previous = limited[-1]
            if previous.end_ms == start_ms:
                limited[-1] = Segment(previous.start_ms, segment.end_ms)
                continue
        limited.append(Segment(start_ms, segment.end_ms))
    return limited


def _derive_key(source) -> str:
    """Derive a human-readable key from the audio source."""
    if isinstance(source, str):
        return os.path.splitext(os.path.basename(source))[0]
    return f"audio_{int(time.time())}"


def _needs_external_punctuation(
    sentences: list[SentenceInfo],
    *,
    min_text_chars: int = 80,
    max_chars_per_terminal_mark: int = 240,
) -> bool:
    """Detect long ASR text whose claimed native punctuation is unusable."""
    text = " ".join(sentence.text.strip() for sentence in sentences if sentence.text.strip())
    if len(text) < min_text_chars:
        return False
    terminal_marks = sum(text.count(mark) for mark in ".!?。！？")
    return (
        terminal_marks == 0
        or len(text) / terminal_marks > max_chars_per_terminal_mark
    )
