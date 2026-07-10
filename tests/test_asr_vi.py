"""Test end-to-end pipeline: ASR + VAD + Speaker Diarization.

Dùng để test cả 2 hướng ASR:
  - Hướng 1: faster-whisper  (Vietnamese + word timestamps)
  - Hướng 2: zipformer-vi    (Vietnamese + token timestamps)

Cách chạy:
    cd /home/anhtu/workspace/HIT/MeetingMindAI

    # Hướng 2: Zipformer-VI (model đã có sẵn, không cần cài thêm)
    python tests/test_asr_vi.py --asr zipformer-vi --audio tests/data/test1.wav

    # Hướng 1: faster-whisper (cần: pip install faster-whisper)
    python tests/test_asr_vi.py --asr faster-whisper --audio tests/data/test1.wav

Kết quả kiểm tra:
  ✅ text tiếng Việt đúng
  ✅ char_timestamps có giá trị (không rỗng)
  ✅ speaker được gán cho từng câu
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SAMPLE_RATE = 16000


def run(asr_model_name: str, audio_path: str) -> None:
    import meetasr  # noqa: F401 — trigger registration of all models

    from meetasr.auto.auto_pipeline import AutoPipeline
    from meetasr.auto.auto_model import AutoModel
    from meetasr.utils.audio import load_audio
    from meetasr.utils.timestamp import merge_vad_segments, build_sentence_info

    # ── Build config cho từng hướng ──
    if asr_model_name == "faster-whisper":
        config = {
            "asr": {
                "model": "faster-whisper",
                "hub": "none",
                "model_size": "medium",     # Đổi thành "large-v3" nếu muốn chất lượng tốt hơn
                "device": "cpu",
                "compute_type": "int8",
            },
            "vad": {"model": "fsmn-vad", "hub": "ms", "max_single_segment_time": 15000},
            "punc": {"model": "vibert-capu", "hub": "hf"},
            "spk": {"model": "cam++", "hub": "ms"},
        }
    elif asr_model_name == "zipformer-vi":
        config = {
            "asr": {
                "model": "zipformer-vi",
                "hub": "hf",
                "device": "cpu",
            },
            "vad": {"model": "fsmn-vad", "hub": "ms", "max_single_segment_time": 15000},
            "punc": {"model": "vibert-capu", "hub": "hf"},
            "spk": {"model": "cam++", "hub": "ms"},
        }
    else:
        print(f"❌ Unknown ASR model: {asr_model_name}")
        print("   Chọn: faster-whisper | zipformer-vi")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  TEST: {asr_model_name.upper()}")
    print(f"  Audio: {audio_path}")
    print(f"{'='*70}\n")

    # ── Load pipeline ──
    logging.info("Building pipeline...")
    pipeline = AutoPipeline.from_config(config)

    # ──────────────────────────────────────────────────────
    # PHASE 1: Raw ASR (trước diarization) để debug
    # ──────────────────────────────────────────────────────
    logging.info(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE

    logging.info("Running VAD...")
    segments = pipeline._run_vad(audio)

    print(f"  VAD: {len(segments)} segments")
    for i, seg in enumerate(segments):
        dur = (seg.end_ms - seg.start_ms) / 1000
        print(f"    [{i}] {seg.start_ms/1000:.1f}s → {seg.end_ms/1000:.1f}s ({dur:.1f}s)")
    print()

    logging.info("Running ASR...")
    asr_results = pipeline._run_asr(audio, segments, language="vi")

    print(f"  ASR: {len(asr_results)} results (raw, trước diarization)")
    total_chars_raw = 0
    for i, r in enumerate(asr_results):
        text = r.get("text", "")
        ts_count = len(r.get("timestamp", []))
        total_chars_raw += len(text)
        seg = segments[i] if i < len(segments) else None
        seg_info = f" (VAD: {seg.start_ms/1000:.1f}-{seg.end_ms/1000:.1f}s)" if seg else ""
        print(f"    [{i}]{seg_info}")
        print(f"      Text ({len(text)} chars, {ts_count} ts): {text}")
        # Check alignment
        if ts_count > 0 and ts_count != len(text):
            print(f"      ⚠️  MISMATCH: {len(text)} chars ≠ {ts_count} timestamps!")
    print()
    print(f"  Tổng raw text: {total_chars_raw} ký tự")
    print()

    # ──────────────────────────────────────────────────────
    # PHASE 2: Full pipeline (ASR + Punc + Speaker)
    # ──────────────────────────────────────────────────────
    logging.info("Running full pipeline (transcribe)...")
    result = pipeline.transcribe(audio_path, language="vi")

    print(f"\n{'='*70}")
    print(f"  KẾT QUẢ SAU DIARIZATION ({len(result.sentence_info)} câu, {result.duration:.1f}s)")
    print(f"{'='*70}\n")

    ts_count_total = 0
    total_chars_final = 0
    speakers_found = set()

    for i, s in enumerate(result.sentence_info):
        ts_count = len(s.char_timestamps)
        ts_count_total += ts_count
        total_chars_final += len(s.text)
        spk_label = f"[Speaker {s.speaker}]" if s.speaker is not None else "[No speaker]"
        if s.speaker is not None:
            speakers_found.add(s.speaker)

        print(f"  [{s.start:6.2f}s → {s.end:6.2f}s] {spk_label}")
        print(f"    Text ({len(s.text)} chars): {s.text}")
        if ts_count > 0 and ts_count != len(s.text):
            print(f"    ⚠️  MISMATCH: {len(s.text)} chars ≠ {ts_count} timestamps!")
        print()

    # ── Tổng kết ──
    print(f"{'='*70}")
    print(f"  TỔNG KẾT")
    print(f"{'='*70}")
    print(f"  Số câu          : {len(result.sentence_info)}")
    print(f"  Tổng ký tự (raw): {total_chars_raw}")
    print(f"  Tổng ký tự (final): {total_chars_final}")
    if total_chars_raw > 0:
        pct = total_chars_final / total_chars_raw * 100
        print(f"  Bảo toàn text   : {pct:.0f}%  {'✅' if pct > 90 else '⚠️ MẤT TEXT!'}")
    print(f"  Tổng timestamps : {ts_count_total}")
    print(f"  Số speakers     : {len(speakers_found)} {sorted(speakers_found)}")

    # ── Kiểm tra tự động ──
    print(f"\n  CHECK:")

    ok_text = len(result.sentence_info) > 0
    ok_ts = ts_count_total > 0
    ok_spk = len(speakers_found) > 0
    ok_preserve = total_chars_raw > 0 and total_chars_final / total_chars_raw > 0.9

    print(f"  {'✅' if ok_text else '❌'} ASR ra text      : {len(result.sentence_info)} câu")
    print(f"  {'✅' if ok_ts else '❌'} Có timestamps    : {ts_count_total} entries")
    print(f"  {'✅' if ok_spk else '❌'} Có speaker labels: {sorted(speakers_found)}")
    print(f"  {'✅' if ok_preserve else '❌'} Text bảo toàn    : {total_chars_final}/{total_chars_raw} chars")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asr",
        required=True,
        choices=["faster-whisper", "zipformer-vi"],
        help="ASR model để test",
    )
    parser.add_argument(
        "--audio",
        default="tests/data/test1.wav",
        help="Đường dẫn tới file audio (default: tests/data/test1.wav)",
    )
    args = parser.parse_args()
    run(args.asr, args.audio)
