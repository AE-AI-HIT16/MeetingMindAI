"""Time the offline (RunPod) pipeline stage by stage on one audio file.

Usage:
    python scripts/eval/benchmark.py meeting.wav
    python scripts/eval/benchmark.py meeting.wav --set asr.model_size=Qwen/Qwen3-ASR-1.7B
    # previous pipeline (no overlap detection / separation):
    python scripts/eval/benchmark.py meeting.wav \
        --set pipeline.overlap_detection.enabled=false \
        --set pipeline.overlap_separation.enabled=false

Run each configuration in its own process so GPU memory is measured cleanly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import BATCH_SIZE, SR, build_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/runpod_gpu.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--language", default="vi")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="turns per RunPod call (backend job worker)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import torch

    from meetasr.runpod.utils.audio import load_audio

    t0 = time.perf_counter()
    pipeline = build_pipeline(args.config, args.overrides)
    for component in (pipeline.segmenter, pipeline.separator):
        if component is not None:
            component._ensure_loaded()
    pipeline.asr._ensure_loaded()
    load_s = time.perf_counter() - t0

    audio = load_audio(str(args.audio))
    minutes = len(audio) / SR / 60
    # Warm-up so CUDA kernel setup is not billed to the first stage.
    pipeline.transcribe_chunks([audio[: SR * 5]], language=args.language)
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    prepared = pipeline.prepare_diarization_first_transcription(audio)
    prepare_s = time.perf_counter() - t0
    turns, profiles = prepared[2] or [], (prepared[4] if len(prepared) > 4 else {})

    t0 = time.perf_counter()
    words = 0
    for index in range(0, len(turns), args.batch_size):
        batch = turns[index:index + args.batch_size]
        chunks = [audio[int(t.start_ms * SR / 1000):int(t.end_ms * SR / 1000)] for t in batch]
        results = pipeline.transcribe_chunks(
            chunks,
            language=args.language,
            overlaps=[[(a - t.start_ms, b - t.start_ms) for a, b in getattr(t, "overlaps", [])] for t in batch],
            speaker_embeddings=[profiles.get(t.speaker) for t in batch],
        )
        words += sum(len(s.text.split()) for sentences in results for s in sentences)
    asr_s = time.perf_counter() - t0

    total = prepare_s + asr_s
    print(f"audio {minutes:.1f} min | turns {len(turns)} "
          f"({sum(1 for t in turns if getattr(t, 'overlaps', None))} overlapped) | words {words}")
    print(f"model load {load_s:.1f}s | prepare (VAD+diarization) {prepare_s:.1f}s | "
          f"ASR {asr_s:.1f}s | total {total:.1f}s = {total / minutes:.1f}s per audio minute | "
          f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
