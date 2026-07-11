"""Diarization utility functions — ported from 3D-Speaker (Apache 2.0).

Source references:
    chunk_segment    : speakerlab/bin/infer_diarization.py:234-241
    circle_pad       : speakerlab/utils/utils.py:232-238
    compressed_seg   : speakerlab/bin/infer_diarization.py:402-419
    assign_speakers_by_overlap : adapted from egs/.../out_transcription.py:60-109
"""

from __future__ import annotations

import numpy as np
import torch



def chunk_segment(
    start_s: float,
    end_s: float,
    dur: float = 1.5,
    step: float = 0.75,
) -> list[list[float]]:
    """Split a VAD segment into overlapping sub-segments.

    Adapted from 3D-Speaker infer_diarization.py:234-241.

    Args:
        start_s: Segment start in seconds.
        end_s: Segment end in seconds.
        dur: Sub-segment duration in seconds (default 1.5).
        step: Sliding step in seconds (default 0.75, 50% overlap).

    Returns:
        List of [sub_start, sub_end] pairs in seconds.
    """
    chunks = []
    subseg_st = round(start_s, 3)
    while round(subseg_st+dur,3) <= round(end_s+step,3):
        subseg_ed = round(min(subseg_st+dur, end_s), 3)
        chunks.append([subseg_st, subseg_ed])
        subseg_st = round(subseg_st+step,3)
    return chunks


def circle_pad(x: "torch.Tensor", target_len: int, dim: int = 0) -> "torch.Tensor":
    """Pad a short audio tensor by repeating it (circular padding).

    Adapted from 3D-Speaker utils.py:232-238.
    Preserves speech characteristics better than zero-padding.

    Args:
        x: 1-D audio tensor.
        target_len: Target number of samples.
        dim: Dimension to pad along (default 0).

    Returns:
        Tensor of length target_len.
    """

    xlen = x.shape[dim]
    if xlen >= target_len:
        return x
    n = int(np.ceil(target_len / xlen))
    xcat = torch.cat([x for _ in range(n)], dim=dim)
    return torch.narrow(xcat, dim, 0, target_len)


def compressed_seg(seg_list: list[list]) -> list[list]:
    """Merge adjacent same-speaker segments, handling overlap boundaries.

    Adapted from 3D-Speaker infer_diarization.py:402-419.

    Args:
        seg_list: [[start_s, end_s, speaker_id], ...] sorted by start.

    Returns:
        Merged segment list with same format.
    """
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed, cluster_id = seg
        if i == 0:
            new_seg_list.append([seg_st, seg_ed, cluster_id])
        elif cluster_id == new_seg_list[-1][2]:
            # Same speaker: extend or append if gap exists
            if seg_st > new_seg_list[-1][1]:
                new_seg_list.append([seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][1] = seg_ed
        else:
            # Different speaker: split at midpoint if overlapping
            if seg_st < new_seg_list[-1][1]:
                p = (new_seg_list[-1][1] + seg_st) / 2
                new_seg_list[-1][1] = p
                seg_st = p
            new_seg_list.append([seg_st, seg_ed, cluster_id])
    return new_seg_list


def assign_speakers_by_overlap(
    sentence_info: list,
    diar_segs: list[list],
) -> list:
    """Assign speaker to each sentence based on maximum overlap duration.

    Adapted from 3D-Speaker out_transcription.py:60-109.
    Replaces the old zip() logic (1 speaker per VAD segment) with per-sentence overlap matching.

    Args:
        sentence_info: List of SentenceInfo objects with .start and .end (seconds).
        diar_segs: [[start_s, end_s, speaker_id], ...] from compressed_seg.

    Returns:
        sentence_info with .speaker filled in for each sentence.
    """
    last_spk = 0
    for s in sentence_info:
        overlap_per_spk: dict[int, float] = {}
        for seg_st, seg_ed, spk_id in diar_segs:
            overlap = min(s.end, seg_ed) - max(s.start, seg_st)
            if overlap > 0:
                overlap_per_spk[spk_id] = overlap_per_spk.get(spk_id, 0.0) + overlap

        if overlap_per_spk:
            s.speaker = max(overlap_per_spk, key=overlap_per_spk.get)
            last_spk = s.speaker
        else:
            # No overlap found: inherit previous speaker
            s.speaker = last_spk

    return sentence_info

def map_chars_to_speakers(
        char_timestamps: list[list[int]],
        diar_segs: list[list],
) -> list[int | None]:
    """Map character-level timestamps to speaker IDs.

    Args:
        char_timestamps: List of character start and end times in ms.
        diar_segs: Diarization segments [[start_s, end_s, speaker_id], ...].

    Returns:
        List of speaker IDs corresponding to each character.
    """
    char_speakers: list[int | None] = []
    for ts in char_timestamps:
        char_start_s = ts[0] / 1000.0
        char_end_s = ts[1] / 1000.0
        best_spk = None
        best_overlap = 0.0
        for seg_st, seg_ed, spk_id in diar_segs:
            overlap=min(char_end_s, seg_ed) - max(char_start_s, seg_st)
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = spk_id
        char_speakers.append(best_spk)
    return char_speakers


def _fill_none_gaps(char_speakers: list[int | None]) ->list[int | None]:
    """Fill None gaps in char_speakers with the last known speaker."""
    result = list(char_speakers)
    last_valid = None
    for i, spk in enumerate(result):
        if spk is not None:
            last_valid = spk
        else:
            result[i] = last_valid
    last_valid = None
    for i in range(len(result) -1, -1 , -1):
        if result[i] is not None:
            last_valid = result[i]
        else:
            result[i] = last_valid
    return result

def _merge_short_groups(
        groups: list[tuple[int,int,int]],
        min_chars: int =3,
) -> list[tuple[int,int,int]]:
    if len(groups) <2:
        return groups
    merged = [groups[0]]
    for start_idx, end_idx, spk in groups[1:]:
        length = end_idx - start_idx
        if length < min_chars:
            merged[-1] = (merged[-1][0], end_idx, merged[-1][2])
        else:
            merged.append((start_idx, end_idx, spk))
    return merged
def split_at_speaker_turns(
    sentence: "SentenceInfo",
    char_speakers: list[int | None],
    min_chars: int = 3,
) -> list["SentenceInfo"]:
    """Split a sentence into multiple sub-sentences at speaker turns.

    Args:
        sentence: SentenceInfo object containing text and character timestamps.
        char_speakers: List of speaker IDs for each character.
        min_chars: Minimum number of characters for a valid speaker segment.

    Returns:
        List of new SentenceInfo objects divided by speaker turns.
    """
    from meetasr.schemas import SentenceInfo
    if not char_speakers or not sentence.char_timestamps:
        return [sentence]
    char_speakers = _fill_none_gaps(char_speakers)
    if all(s is None for s in char_speakers):
        return [sentence]
    groups: list[tuple[int, int, int]] = []
    current_spk = char_speakers[0]
    start_idx = 0
    for i in range(1, len(char_speakers)):
        if char_speakers[i] != current_spk:
            groups.append((start_idx,i,current_spk))
            current_spk = char_speakers[i]
            start_idx =i
    groups.append((start_idx,len(char_speakers),current_spk))
    groups = _merge_short_groups(groups, min_chars)
    text = sentence.text
    sub_sentences: list["SentenceInfo"] = []
    for grp_start, grp_end, spk in groups:
        sub_text = text[grp_start:grp_end].strip()
        if not sub_text:
            continue
        sub_ts = sentence.char_timestamps[grp_start: grp_end]
        if not sub_ts:
            continue
        sub_stent = SentenceInfo(
            text=sub_text,
            start = sub_ts[0][0]/ 1000.0,
            end = sub_ts[-1][1]/ 1000.0,
            speaker = spk,
            char_timestamps = sub_ts,
        )
        sub_sentences.append(sub_stent)
    return sub_sentences if sub_sentences else [sentence]
