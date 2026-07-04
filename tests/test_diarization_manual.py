"""Test diarization-only: audio file → speaker timeline.

Cách chạy:
    conda activate meetasr
    cd /home/anhtu/workspace/HIT/MeetingMindAI
    python tests/test_diarization_manual.py path/to/audio.wav

Đầu ra mẫu:
    [00:00.000 → 00:03.250]  Speaker 0
    [00:03.250 → 00:07.100]  Speaker 1
    [00:07.100 → 00:12.500]  Speaker 0
    ...
"""

from __future__ import annotations

import sys
import logging
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ──────────────────────────────────────────────────────────────────
# CONFIG — chỉnh ở đây nếu cần
# ──────────────────────────────────────────────────────────────────
VAD_MODEL    = "fsmn-vad"
SPK_MODEL    = "cam++"
DEVICE       = "cpu"        # đổi sang "cuda" nếu có GPU
HUB          = "ms"         # ModelScope
SAMPLE_RATE  = 16000
# ──────────────────────────────────────────────────────────────────


def fmt_time(seconds: float) -> str:
    """Format seconds → MM:SS.mmm"""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:06.3f}"


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file → float32 mono numpy array at target sample rate."""
    try:
        import soundfile as sf
        audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    except Exception:
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def run_vad(vad_model, audio: np.ndarray):
    """Run VAD, return list of Segment."""
    from meetasr.utils.timestamp import merge_vad_segments
    segments = vad_model.detect(audio)
    segments = merge_vad_segments(segments)
    return segments


def run_diarization(spk_model, audio: np.ndarray, segments) -> list[list]:
    """Sub-segment → embed → cluster → compress → return diar_segs."""
    from meetasr.utils.diarization import chunk_segment, circle_pad, compressed_seg

    target_len = int(1.5 * SAMPLE_RATE)

    # T1: chunk all VAD segments
    all_chunks = []
    for seg in segments:
        all_chunks.extend(chunk_segment(seg.start_s, seg.end_s, dur=1.5, step=0.75))

    if not all_chunks:
        logging.warning("No chunks produced — audio may be too short or silent.")
        return []

    logging.info(f"Embedding {len(all_chunks)} chunks...")

    # T1: embed each chunk
    embeddings = []
    for st, ed in all_chunks:
        chunk_np = audio[int(st * SAMPLE_RATE):int(ed * SAMPLE_RATE)]
        if len(chunk_np) < target_len:
            t = torch.from_numpy(chunk_np).float()
            chunk_np = circle_pad(t, target_len).numpy()
        emb = spk_model.embed(chunk_np.astype(np.float32))
        embeddings.append(emb)

    # T2: cluster
    all_embs = torch.cat(embeddings, dim=0)
    logging.info(f"Clustering {all_embs.shape[0]} embeddings...")
    labels = spk_model.cluster(all_embs)

    # T3: build + compress diar segments
    diar_segs = [[c[0], c[1], int(l)] for c, l in zip(all_chunks, labels)]
    diar_segs = compressed_seg(diar_segs)

    return diar_segs


def print_timeline(diar_segs: list[list], audio_duration: float) -> None:
    """Pretty-print speaker timeline."""
    print()
    print("=" * 55)
    print("  SPEAKER DIARIZATION TIMELINE")
    print("=" * 55)
    if not diar_segs:
        print("  (no speakers detected)")
        return

    speakers = sorted({s[2] for s in diar_segs})
    print(f"  Detected speakers : {len(speakers)}  →  {speakers}")
    print(f"  Audio duration    : {fmt_time(audio_duration)}")
    print("-" * 55)

    for st, ed, spk in diar_segs:
        duration = ed - st
        print(f"  [{fmt_time(st)} → {fmt_time(ed)}]  "
              f"Speaker {spk}  ({duration:.2f}s)")

    print("=" * 55)

    # Per-speaker speaking time
    print("\n  Speaking time summary:")
    from collections import defaultdict
    spk_time: dict = defaultdict(float)
    for st, ed, spk in diar_segs:
        spk_time[spk] += ed - st
    for spk in sorted(spk_time):
        pct = spk_time[spk] / audio_duration * 100
        print(f"    Speaker {spk}: {spk_time[spk]:.1f}s  ({pct:.1f}%)")
    print()


def main(audio_path: str) -> None:
    t_start = time.perf_counter()

    # ── Load audio ──
    logging.info(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    logging.info(f"Duration: {duration:.1f}s  |  Samples: {len(audio)}")

    # ── Load models ──
    from meetasr.auto.auto_model import AutoModel

    logging.info("Loading VAD model...")
    vad = AutoModel(model=VAD_MODEL, hub=HUB, device=DEVICE)

    logging.info("Loading speaker model...")
    spk = AutoModel(model=SPK_MODEL, hub=HUB, device=DEVICE)

    # ── VAD ──
    logging.info("Running VAD...")
    segments = run_vad(vad, audio)
    logging.info(f"VAD: {len(segments)} speech segments detected")
    for i, seg in enumerate(segments):
        logging.info(f"  Segment {i}: {fmt_time(seg.start_s)} → {fmt_time(seg.end_s)}")

    # ── Diarization ──
    logging.info("Running diarization (embed + cluster + compress)...")
    diar_segs = run_diarization(spk, audio, segments)

    # ── Output ──
    print_timeline(diar_segs, duration)
    logging.info(f"Total time: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/test_diarization_manual.py <audio_file>")
        print("Example: python tests/test_diarization_manual.py test_audio.wav")
        sys.exit(1)

    main(sys.argv[1])
