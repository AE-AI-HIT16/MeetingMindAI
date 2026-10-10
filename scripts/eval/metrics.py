"""Scoring for synthetic meetings: DER, WER/cpWER, speaker count, noise
hallucination and interjection recall.

Hypothesis format: list of dicts ``{"start", "end", "speaker", "text"}``
(seconds), i.e. what the transcript UI shows.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from scipy.optimize import linear_sum_assignment

FRAME_S = 0.01


def normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_word != hyp_word),
            )
        previous = current
    return previous[-1]


def _activity(items, labels, n_frames, collar_s=0.0):
    matrix = np.zeros((n_frames, len(labels)), dtype=bool)
    index = {label: i for i, label in enumerate(labels)}
    for item in items:
        a = max(0, int(round(item["start"] / FRAME_S)))
        b = min(n_frames, int(round(item["end"] / FRAME_S)))
        if b > a:
            matrix[a:b, index[item["speaker"]]] = True
    return matrix


def speaker_mapping(reference: list[dict], hypothesis: list[dict], duration: float):
    """Hungarian map hyp speaker -> ref speaker by overlapping time."""
    ref_labels = sorted({u["speaker"] for u in reference})
    hyp_labels = sorted({h["speaker"] for h in hypothesis if h["speaker"] is not None})
    n_frames = int(duration / FRAME_S) + 1
    ref = _activity(reference, ref_labels, n_frames)
    hyp = _activity([h for h in hypothesis if h["speaker"] is not None], hyp_labels, n_frames)
    if not hyp_labels:
        return {}, ref_labels, hyp_labels, ref, hyp
    overlap = ref.T.astype(np.int64) @ hyp.astype(np.int64)
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {hyp_labels[c]: ref_labels[r] for r, c in zip(rows, cols) if overlap[r, c] > 0}
    return mapping, ref_labels, hyp_labels, ref, hyp


def der(reference, hypothesis, duration, collar_s=0.25):
    """Overlap-aware DER with a no-score collar around reference boundaries."""
    mapping, ref_labels, hyp_labels, ref, hyp = speaker_mapping(reference, hypothesis, duration)
    n_frames = ref.shape[0]
    scored = np.ones(n_frames, dtype=bool)
    collar = int(collar_s / FRAME_S)
    for u in reference:
        for edge in (u["start"], u["end"]):
            f = int(round(edge / FRAME_S))
            scored[max(0, f - collar): f + collar] = False

    mapped = np.zeros_like(ref)
    for h_index, label in enumerate(hyp_labels):
        if label in mapping:
            mapped[:, ref_labels.index(mapping[label])] |= hyp[:, h_index]
    n_ref = ref.sum(1)
    n_hyp = hyp.sum(1)
    n_correct = (ref & mapped).sum(1)
    miss = np.maximum(n_ref - n_hyp, 0)
    fa = np.maximum(n_hyp - n_ref, 0)
    confusion = np.minimum(n_ref, n_hyp) - n_correct
    total = n_ref[scored].sum()
    overlap_frames = scored & (n_ref >= 2)
    return {
        "der": float((miss + fa + confusion)[scored].sum() / max(total, 1)),
        "miss": float(miss[scored].sum() / max(total, 1)),
        "false_alarm": float(fa[scored].sum() / max(total, 1)),
        "confusion": float(confusion[scored].sum() / max(total, 1)),
        # Of the time 2+ people really talk at once, how much do we show 2+?
        "overlap_detected": float((n_hyp[overlap_frames] >= 2).mean()) if overlap_frames.any() else None,
        "mapping": mapping,
    }


def score_meeting(reference: dict, hypothesis: list[dict]) -> dict:
    utterances = reference["utterances"]
    duration = reference["duration"]
    diar = der(utterances, hypothesis, duration)
    mapping = diar.pop("mapping")

    ref_words = [w for u in utterances for w in normalize_words(u["text"])]
    hyp_sorted = sorted(hypothesis, key=lambda h: h["start"])
    hyp_words = [w for h in hyp_sorted for w in normalize_words(h["text"])]
    wer_errors = edit_distance(ref_words, hyp_words)

    # cpWER: per reference speaker, compare with the mapped hyp speaker's text.
    cp_errors = 0
    ref_speakers = sorted({u["speaker"] for u in utterances})
    for speaker in ref_speakers:
        r = [w for u in utterances if u["speaker"] == speaker for w in normalize_words(u["text"])]
        h = [
            w for item in hyp_sorted
            if mapping.get(item["speaker"]) == speaker
            for w in normalize_words(item["text"])
        ]
        cp_errors += edit_distance(r, h)
    unmapped = [
        w for item in hyp_sorted
        if item["speaker"] is None or item["speaker"] not in mapping
        for w in normalize_words(item["text"])
    ]
    cp_errors += len(unmapped)

    # Words emitted while only a non-speech event was audible.
    noise_words = 0
    for event in reference["noise_events"]:
        speech_nearby = any(
            u["start"] < event["end"] + 0.2 and u["end"] > event["start"] - 0.2
            for u in utterances
        )
        if speech_nearby:
            continue
        for item in hypothesis:
            mid = (item["start"] + item["end"]) / 2
            if event["start"] - 0.3 <= mid <= event["end"] + 0.3:
                noise_words += len(normalize_words(item["text"]))

    # Interjections: words found in a hyp segment of the right speaker.
    found = 0
    interjections = [u for u in utterances if u["kind"] == "interjection"]
    for u in interjections:
        words = normalize_words(u["text"])
        best = 0.0
        for item in hypothesis:
            if mapping.get(item["speaker"]) != u["speaker"]:
                continue
            if item["start"] > u["end"] + 0.5 or item["end"] < u["start"] - 0.5:
                continue
            hyp_set = normalize_words(item["text"])
            hits = sum(1 for w in words if w in hyp_set)
            best = max(best, hits / max(len(words), 1))
        found += best >= 0.5

    hyp_speakers = {h["speaker"] for h in hypothesis if h["speaker"] is not None}
    return {
        **diar,
        "wer": wer_errors / max(len(ref_words), 1),
        "cpwer": cp_errors / max(len(ref_words), 1),
        "ref_words": len(ref_words),
        "n_ref_speakers": len(ref_speakers),
        "n_hyp_speakers": len(hyp_speakers),
        "noise_words": noise_words,
        "interjections": len(interjections),
        "interjections_found": found,
    }


def aggregate(results: dict[str, dict]) -> dict:
    rows = list(results.values())
    total_words = sum(r["ref_words"] for r in rows)

    def weighted(key):
        values = [(r[key], r["ref_words"]) for r in rows if r.get(key) is not None]
        return sum(v * w for v, w in values) / max(sum(w for _, w in values), 1)

    interjections = sum(r["interjections"] for r in rows)
    return {
        "meetings": len(rows),
        "der": weighted("der"),
        "miss": weighted("miss"),
        "false_alarm": weighted("false_alarm"),
        "confusion": weighted("confusion"),
        "overlap_detected": weighted("overlap_detected"),
        "wer": weighted("wer"),
        "cpwer": weighted("cpwer"),
        "speaker_count_exact": sum(r["n_hyp_speakers"] == r["n_ref_speakers"] for r in rows) / len(rows),
        "speaker_count_abs_err": sum(abs(r["n_hyp_speakers"] - r["n_ref_speakers"]) for r in rows) / len(rows),
        "noise_words": sum(r["noise_words"] for r in rows),
        "interjection_recall": sum(r["interjections_found"] for r in rows) / max(interjections, 1),
        "ref_words": total_words,
    }
