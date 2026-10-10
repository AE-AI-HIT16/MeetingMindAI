"""Pipeline steps for non-speech and overlapped speech.

Kept apart from ``MeetPipeline`` so the orchestration file stays readable:
gate VAD segments without a voice, re-check who speaks in interjections and
separate a speaker's voice inside overlapped regions before ASR.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from meetasr.runpod.models.abs_models import AbsSegmenter, AbsSeparator, AbsSpk
from meetasr.runpod.schemas import Segment, SpeakerTurn
from meetasr.runpod.utils.segmentation import SAMPLE_RATE, LocalSegmentation, speech_ratio

logger = logging.getLogger(__name__)


def gate_non_speech(
    segmenter: AbsSegmenter,
    audio: np.ndarray,
    vad_segments: list[Segment],
) -> tuple[LocalSegmentation, list[Segment]]:
    """Drop VAD segments the segmentation model hears as non-speech.

    Silero fires on coughs, laughter, clapping or typing; those segments made
    Qwen write words ("Applause", "Đi.") and gave clustering extra "speakers".

    Args:
        segmenter: Segmentation model with a ``min_speech_ratio`` threshold.
        audio: Float32 mono audio at 16kHz.
        vad_segments: Speech segments from the VAD.

    Returns:
        The segmentation (reused by diarization) and the kept segments.
    """
    segmentation = segmenter(audio)
    kept = [
        segment for segment in vad_segments
        if speech_ratio(segmentation, segment.start_s, segment.end_s)
        >= segmenter.min_speech_ratio
    ]
    if len(kept) != len(vad_segments):
        logger.info(
            "Speech gate: dropped %d of %d VAD segment(s) without voice.",
            len(vad_segments) - len(kept), len(vad_segments),
        )
    return segmentation, kept


def verify_overlap_speakers(
    separator: AbsSeparator,
    spk: AbsSpk,
    audio: np.ndarray,
    turns: list[SpeakerTurn],
    profiles: dict[int, list[float]],
    *,
    min_overlap_ratio: float = 0.5,
    margin: float = 0.05,
) -> None:
    """Re-check who speaks in mostly-overlapped turns using separated voices.

    Segmentation guesses the second voice's identity from a 10 s window
    and is often wrong for short interjections. The turn is separated
    into two streams; the stream closest to a concurrent speaker is that
    speaker, and the other stream is matched against every voice profile.

    Args:
        separator: Speech separation model.
        spk: Speaker model used for voice embeddings.
        audio: Float32 mono audio at 16kHz.
        turns: Speaker turns with ``overlaps``; relabelled in place.
        profiles: Mean voice embedding per turn speaker.
        min_overlap_ratio: Only turns overlapped at least this much.
        margin: Cosine gain needed to change a turn's speaker.
    """
    if len(profiles) < 2:
        return
    speakers = sorted(profiles)
    matrix = np.asarray([profiles[k] for k in speakers], dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    changed = 0
    t0 = time.perf_counter()
    for turn in turns:
        covered = sum(b - a for a, b in turn.overlaps)
        if turn.duration_ms <= 0 or covered / turn.duration_ms < min_overlap_ratio:
            continue
        concurrent = {
            other.speaker for other in turns
            if other.speaker != turn.speaker
            and other.start_ms < turn.end_ms and other.end_ms > turn.start_ms
        }
        if not concurrent:
            continue
        chunk = audio[int(turn.start_ms * SAMPLE_RATE / 1000):int(turn.end_ms * SAMPLE_RATE / 1000)]
        try:
            streams = separator.separate(chunk)
            embeddings = spk.embed_batch(streams)
        except Exception as exc:
            logger.warning("Speaker verification failed for a turn: %s", exc)
            continue
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().float().cpu().numpy()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
        similarity = embeddings @ matrix.T  # [streams, speakers]
        columns = [speakers.index(k) for k in concurrent if k in speakers]
        if not columns:
            continue
        # The stream that sounds most like someone already talking is them.
        other_stream = int(np.argmax(similarity[:, columns].max(axis=1)))
        own = similarity[1 - other_stream].copy()
        own[columns] = -np.inf
        best = speakers[int(np.argmax(own))]
        current = speakers.index(turn.speaker) if turn.speaker in speakers else None
        if best != turn.speaker and (
            current is None or own[speakers.index(best)] - own[current] > margin
        ):
            turn.speaker = best
            changed += 1
    if changed:
        logger.info(
            "Overlap speaker check: relabelled %d turn(s) (%.2fs)",
            changed, time.perf_counter() - t0,
        )


def separate_overlaps(
    separator: AbsSeparator,
    spk: AbsSpk,
    chunks: list[np.ndarray],
    overlaps: list[list[tuple[int, int]]],
    speaker_embeddings: list[list[float] | None],
) -> list[np.ndarray]:
    """Replace overlapped regions of each chunk with that chunk's speaker.

    Args:
        separator: Speech separation model.
        spk: Speaker model whose ``embed_batch`` picks the target stream.
        chunks: Turn audio; modified copies are returned.
        overlaps: Per chunk, overlapped ``(start_ms, end_ms)`` ranges.
        speaker_embeddings: Per chunk, the speaker's voice profile.

    Returns:
        Chunks with the target voice in overlapped regions.
    """
    t0 = time.perf_counter()
    separated = 0
    for index, (regions, embedding) in enumerate(zip(overlaps, speaker_embeddings)):
        if not regions or embedding is None:
            continue
        try:
            chunks[index] = separator.extract(
                chunks[index], regions, np.asarray(embedding), spk.embed_batch
            )
            separated += 1
        except Exception as exc:
            logger.warning("Overlap separation failed for chunk %s: %s", index, exc)
    if separated:
        logger.info(
            "Separated overlapped speech in %d chunk(s) (%.2fs)",
            separated, time.perf_counter() - t0,
        )
    return chunks
