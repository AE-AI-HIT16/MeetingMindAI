"""Run the offline (RunPod) pipeline on synthetic meetings and score it.

Mirrors production: ``prepare_incremental`` (VAD + diarization -> speaker
turns) on the worker, the backend slicing turns out of the full audio, and
``transcribe_segments`` in batches of 8.

Usage:
    python scripts/eval/evaluate.py --data data/eval/synthetic \
        --config configs/runpod_gpu.yaml --out data/eval/results/baseline.json
    # diarization only (fast, no ASR):
    python scripts/eval/evaluate.py ... --diar-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import aggregate, score_meeting  # noqa: E402

from meetasr.backend.utils.overlap_echo import is_overlap_echo, overlap_ratio  # noqa: E402

SR = 16000
BATCH_SIZE = 8


def build_pipeline(config_path: Path, overrides: list[str]):
    import meetasr.runpod as runpod_pkg
    from meetasr.runpod.auto.auto_pipeline import AutoPipeline

    config = yaml.safe_load(config_path.read_text())
    for override in overrides:
        dotted, _, raw = override.partition("=")
        node = config
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    config.pop("llm", None)
    runpod_pkg._register_all_models()
    return AutoPipeline.from_config(config)


def run_meeting(
    pipeline, audio, *, diar_only: bool, language: str, context: str,
    pad_ms: int = 0, drop_echo: bool = True,
):
    prepared = pipeline.prepare_diarization_first_transcription(audio)
    turns = prepared[2]
    # Older pipelines return no speaker profiles (and turns without overlaps).
    profiles = prepared[4] if len(prepared) > 4 else {}
    if turns is None:
        raise RuntimeError("diarization-first returned no speaker turns")
    if diar_only:
        return [
            {"start": t.start_ms / 1000, "end": t.end_ms / 1000, "speaker": t.speaker, "text": ""}
            for t in turns
        ]

    hypothesis = []
    transcribed = []
    for index in range(0, len(turns), BATCH_SIZE):
        batch = turns[index:index + BATCH_SIZE]
        starts = [max(0, t.start_ms - pad_ms) for t in batch]
        chunks = [
            audio[int(start * SR / 1000):int((t.end_ms + pad_ms) * SR / 1000)]
            for start, t in zip(starts, batch)
        ]
        kwargs = {"context": context} if context else {}
        results = pipeline.transcribe_chunks(
            chunks,
            language=language,
            overlaps=[
                [(a - start, b - start) for a, b in getattr(t, "overlaps", [])]
                for start, t in zip(starts, batch)
            ],
            speaker_embeddings=[profiles.get(t.speaker) for t in batch],
            **kwargs,
        )
        for start, turn, sentences in zip(starts, batch, results):
            transcribed.append((turn, " ".join(s.text for s in sentences)))
            for sentence in sentences:
                hypothesis.append({
                    "start": sentence.start + start / 1000,
                    "end": sentence.end + start / 1000,
                    "speaker": turn.speaker,
                    "text": sentence.text,
                    "turn": id(turn),
                })
    if drop_echo:
        echoes = set()
        for turn, text in transcribed:
            concurrent = [
                other_text for other, other_text in transcribed
                if other.speaker != turn.speaker
                and other.start_ms < turn.end_ms and other.end_ms > turn.start_ms
            ]
            ratio = overlap_ratio(turn.start_ms, turn.end_ms, getattr(turn, "overlaps", []))
            if is_overlap_echo(text, ratio, concurrent):
                echoes.add(id(turn))
        hypothesis = [h for h in hypothesis if h["turn"] not in echoes]
    for h in hypothesis:
        h.pop("turn")
    return hypothesis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/runpod_gpu.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="config override, e.g. --set asr.model_size=Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--diar-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--language", default="vi")
    parser.add_argument("--context", default="")
    parser.add_argument("--keep-echo", action="store_true",
                        help="keep overlapped turns that repeat another speaker's words")
    parser.add_argument("--pad-ms", type=int, default=0,
                        help="audio added on both sides of each turn before ASR")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from meetasr.runpod.utils.audio import load_audio

    pipeline = build_pipeline(args.config, args.overrides)
    meetings = sorted(args.data.glob("*.wav"))
    if args.limit:
        meetings = meetings[:args.limit]

    results, hypotheses = {}, {}
    started = time.perf_counter()
    for wav in meetings:
        reference = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        audio = load_audio(str(wav))
        t0 = time.perf_counter()
        hypothesis = run_meeting(
            pipeline, audio, diar_only=args.diar_only,
            language=args.language, context=args.context, pad_ms=args.pad_ms,
            drop_echo=not args.keep_echo,
        )
        score = score_meeting(reference, hypothesis)
        score["seconds"] = round(time.perf_counter() - t0, 1)
        results[wav.stem] = score
        hypotheses[wav.stem] = hypothesis
        print(
            f"{wav.stem}: DER {score['der']:.3f} (miss {score['miss']:.3f} fa {score['false_alarm']:.3f} "
            f"conf {score['confusion']:.3f}) spk {score['n_hyp_speakers']}/{score['n_ref_speakers']}"
            + ("" if args.diar_only else
               f" WER {score['wer']:.3f} cpWER {score['cpwer']:.3f} noise_words {score['noise_words']}"
               f" interj {score['interjections_found']}/{score['interjections']}")
            + f" [{score['seconds']}s]",
            flush=True,
        )

    summary = aggregate(results)
    summary["seconds_total"] = round(time.perf_counter() - started, 1)
    summary["overrides"] = args.overrides
    summary["diar_only"] = args.diar_only
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()},
                     ensure_ascii=False, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"summary": summary, "meetings": results, "hypotheses": hypotheses},
        ensure_ascii=False, indent=1,
    ), encoding="utf-8")


if __name__ == "__main__":
    main()
