"""Build synthetic Vietnamese meetings with exact ground truth.

Mixes VIVOS utterances (19 speakers, read speech with transcripts) into
multi-speaker conversations with the hard cases the pipeline must handle:

* turn-taking with short gaps and partial overlaps (two people talking at once)
* short interjections fully inside another speaker's turn ("dạ vâng"-like)
* non-speech events (cough, laughter, clapping, typing...) from ESC-50 in pauses
* a low noise floor and per-speaker loudness differences

Usage:
    python scripts/eval/build_synthetic_meetings.py \
        --vivos data/eval/raw/vivos/test --noise data/eval/raw/noise \
        --out data/eval/synthetic --count 12

Each meeting produces ``<id>.wav`` and ``<id>.json`` (utterances, noise events).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
INTERJECTION_MAX_S = 2.5


def _load(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)


def _trim(audio: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
    """Trim leading/trailing silence so reference times match real speech."""
    frame = int(0.02 * SR)
    if len(audio) < frame:
        return audio
    n = len(audio) // frame
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    active = np.where(20 * np.log10(rms / (rms.max() + 1e-12)) > threshold_db)[0]
    if active.size == 0:
        return audio
    return audio[max(0, (active[0] - 1) * frame): min(len(audio), (active[-1] + 2) * frame)]


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2) + 1e-12))


def _room_response(rng: random.Random) -> np.ndarray:
    """Synthetic room impulse response (direct path + exponential tail)."""
    rt60 = rng.uniform(0.3, 0.7)
    length = int(rt60 * SR)
    t = np.arange(length) / SR
    noise = np.random.default_rng(rng.randrange(2**31)).normal(0, 1, length)
    tail = noise * np.exp(-6.9 * t / rt60) * 10 ** (-rng.uniform(6, 12) / 20)
    tail[: int(0.005 * SR)] = 0
    tail[0] = 1.0
    return (tail / np.sqrt(np.sum(tail ** 2))).astype(np.float32)


def _reverberate(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    wet = fftconvolve(audio, rir)[: len(audio)].astype(np.float32)
    return wet / max(_rms(wet), 1e-6) * _rms(audio)


def load_vivos(root: Path) -> dict[str, list[tuple[np.ndarray, str]]]:
    prompts = {}
    for line in (root / "prompts.txt").read_text(encoding="utf-8").splitlines():
        key, _, text = line.partition(" ")
        prompts[key] = text.strip().lower()
    speakers: dict[str, list[tuple[np.ndarray, str]]] = {}
    for spk_dir in sorted((root / "waves").iterdir()):
        items = []
        for wav in sorted(spk_dir.glob("*.wav")):
            audio = _trim(_load(wav))
            items.append((audio / max(_rms(audio), 1e-4) * 0.05, prompts[wav.stem]))
        speakers[spk_dir.name] = items
    return speakers


def load_noise(root: Path) -> list[tuple[np.ndarray, str]]:
    clips = []
    for wav in sorted(root.glob("*.wav")):
        audio = _trim(_load(wav), threshold_db=-30.0)
        if len(audio) < int(0.3 * SR):
            continue
        clips.append((audio, wav.name.split("__")[0]))
    return clips


def build_meeting(
    rng: random.Random,
    speakers: dict[str, list[tuple[np.ndarray, str]]],
    noise: list[tuple[np.ndarray, str]],
    *,
    n_speakers: int,
    target_s: float,
    hard: bool = False,
    genders: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict]:
    candidates = sorted(speakers)
    if hard and genders and rng.random() < 0.5:
        # Same-gender groups are the hardest to tell apart.
        gender = rng.choice(["m", "f"])
        same = [name for name in candidates if genders.get(name) == gender]
        if len(same) >= n_speakers:
            candidates = same
    names = rng.sample(candidates, n_speakers)
    pools = {name: rng.sample(speakers[name], len(speakers[name])) for name in names}
    gains = {name: 10 ** (rng.uniform(-6, 3) / 20) for name in names}
    rooms = {name: _room_response(rng) for name in names} if hard else {}
    events: list[tuple[int, np.ndarray]] = []
    utterances: list[dict] = []
    noise_events: list[dict] = []

    def take(name: str, max_s: float | None = None):
        pool = pools[name]
        for index, (audio, text) in enumerate(pool):
            if max_s is None or len(audio) <= max_s * SR:
                audio, text = pool.pop(index)
                if name in rooms:
                    audio = _reverberate(audio, rooms[name])
                return audio, text
        return None

    cursor = int(0.5 * SR)
    previous = None
    while cursor < target_s * SR:
        candidates = [n for n in names if n != previous and pools[n]]
        if not candidates:
            break
        name = rng.choice(candidates)
        turn_start = cursor
        turn_pieces = []
        for _ in range(rng.choice([1, 1, 2, 2, 3])):
            item = take(name)
            if item is None:
                break
            audio, text = item
            audio = audio * gains[name]
            events.append((cursor, audio))
            turn_pieces.append((cursor, cursor + len(audio)))
            utterances.append({
                "start": cursor / SR, "end": (cursor + len(audio)) / SR,
                "speaker": name, "text": text, "kind": "normal",
            })
            cursor += len(audio) + int(rng.uniform(0.15, 0.4) * SR)
        if not turn_pieces:
            break
        turn_end = turn_pieces[-1][1]

        # Interjection by someone else fully inside this turn ("dạ", "vâng").
        others = [n for n in names if n != name and pools[n]]
        if others and turn_end - turn_start > 3 * SR and rng.random() < 0.5:
            other = rng.choice(others)
            item = take(other, max_s=INTERJECTION_MAX_S)
            if item is not None:
                audio, text = item
                audio = audio * gains[other] * 10 ** (rng.uniform(-3, 3) / 20)
                start = rng.randint(turn_start + SR, max(turn_start + SR, turn_end - len(audio) - SR // 2))
                events.append((start, audio))
                utterances.append({
                    "start": start / SR, "end": (start + len(audio)) / SR,
                    "speaker": other, "text": text, "kind": "interjection",
                })

        roll = rng.random()
        if roll < 0.22:
            # Next speaker starts before this one finishes.
            cursor = turn_end - int(rng.uniform(0.3, 1.2) * SR)
        elif roll < 0.42 and noise:
            # A pause with a non-speech event in it.
            gap_start = turn_end + int(rng.uniform(0.3, 0.8) * SR)
            clip, category = rng.choice(noise)
            clip = clip[: int(rng.uniform(1.0, 3.0) * SR)]
            level = 0.05 * rng.uniform(0.3, 1.2)
            clip = clip / max(_rms(clip), 1e-4) * level
            events.append((gap_start, clip))
            noise_events.append({
                "start": gap_start / SR, "end": (gap_start + len(clip)) / SR,
                "category": category,
            })
            cursor = gap_start + len(clip) + int(rng.uniform(0.3, 0.8) * SR)
        else:
            cursor = turn_end + int(rng.uniform(0.2, 1.2) * SR)
        previous = name

    length = max(start + len(audio) for start, audio in events) + SR // 2
    mix = np.zeros(length, dtype=np.float32)
    for start, audio in events:
        mix[start:start + len(audio)] += audio
    floor = np.random.default_rng(rng.randrange(2**31)).normal(0, 1, length).astype(np.float32)
    if hard:
        # Brown-ish noise (fans, air conditioning) at 12-20 dB SNR.
        floor = np.cumsum(floor)
        floor -= np.convolve(floor, np.ones(SR // 10) / (SR // 10), mode="same")
        floor /= max(_rms(floor), 1e-6)
        snr_db = rng.uniform(12, 20)
    else:
        snr_db = rng.uniform(25, 35)
    mix += floor * 0.05 * 10 ** (-snr_db / 20)
    peak = float(np.max(np.abs(mix)))
    if peak > 0.95:
        mix *= 0.95 / peak

    utterances.sort(key=lambda item: item["start"])
    reference = {
        "speakers": names,
        "duration": length / SR,
        "utterances": utterances,
        "noise_events": noise_events,
    }
    return mix, reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vivos", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--minutes", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hard", action="store_true",
                        help="reverb, louder noise floor, same-gender groups, 3-6 speakers")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    speakers = load_vivos(args.vivos)
    noise = load_noise(args.noise)
    genders = dict(
        line.split() for line in (args.vivos / "genders.txt").read_text().splitlines() if line.strip()
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        n_speakers = ([3, 4, 5, 6] if args.hard else [2, 3, 4])[index % (4 if args.hard else 3)]
        mix, reference = build_meeting(
            rng, speakers, noise, n_speakers=n_speakers, target_s=args.minutes * 60,
            hard=args.hard, genders=genders,
        )
        meeting_id = f"meet{index:02d}_{n_speakers}spk"
        sf.write(args.out / f"{meeting_id}.wav", mix, SR)
        (args.out / f"{meeting_id}.json").write_text(
            json.dumps(reference, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        kinds = [u["kind"] for u in reference["utterances"]]
        print(
            f"{meeting_id}: {reference['duration']:.0f}s, {len(kinds)} utt, "
            f"{kinds.count('interjection')} interjections, "
            f"{len(reference['noise_events'])} noise events"
        )


if __name__ == "__main__":
    main()
