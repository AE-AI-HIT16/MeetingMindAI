#!/usr/bin/env python3
"""Manual end-to-end smoke test for DocumentPlanner's multi-pass path.

Usage:
    python3 tests/test_planner_e2e.py

Requirements:
    - Groq API key in meeting_config.yaml (or set GROQ_API_KEY env)
    - No ASR/VAD models needed — transcript is simulated

What it does:
    1. Builds a representative Vietnamese meeting transcript
    2. Creates a Groq LLM client directly (no AutoPipeline)
    3. Runs DocumentPlanner.plan_and_write() — the full long path
    4. Prints report as Markdown + saves JSON to scripts/output/
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meetasr.schemas import TranscriptResult, SentenceInfo
from meetasr.schemas_doc import DocumentReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("e2e_test")


# ------------------------------------------------------------------
# 1. Simulated 60-min Vietnamese meeting transcript
# ------------------------------------------------------------------

def build_long_path_transcript() -> TranscriptResult:
    """Create a representative Vietnamese meeting transcript.

    Returns:
        TranscriptResult used by the manually forced multi-pass path.
    """
    blocks = [
        # Block 1: Opening + Backend status (0-10 min)
        [
            (0, "Xin chào mọi người, cảm ơn đã tham gia cuộc họp hôm nay."),
            (1, "Hôm nay chúng ta sẽ review tiến độ dự án MeetingMind và bàn kế hoạch sprint tới."),
            (0, "Trước tiên, mời anh Khương báo cáo tiến độ backend."),
            (2, "Dạ vâng. Tuần qua team backend đã hoàn thành module ASR integration."),
            (2, "Chúng tôi đã tích hợp SenseVoice small model và test trên 50 file audio."),
            (2, "Độ chính xác trung bình đạt 92 phần trăm trên tiếng Việt."),
            (2, "Có một vấn đề là model bị chậm khi xử lý file trên 30 phút."),
            (0, "Cụ thể là chậm bao nhiêu?"),
            (2, "File 60 phút mất khoảng 8 phút để xử lý trên CPU."),
            (2, "Nếu dùng GPU thì giảm xuống còn 2 phút."),
            (1, "Vậy chúng ta cần deploy trên server có GPU."),
            (0, "OK ghi nhận. Tiếp theo là phần frontend."),
        ],
        # Block 2: Frontend status (10-20 min)
        [
            (3, "Dạ, em là Hà, phụ trách frontend."),
            (3, "Tuần qua em đã hoàn thành giao diện upload audio và hiển thị transcript."),
            (3, "Tính năng real-time preview đang ở mức 80 phần trăm."),
            (3, "Em gặp khó khăn với việc đồng bộ timestamp giữa audio player và transcript."),
            (0, "Vấn đề cụ thể là gì?"),
            (3, "Khi user click vào một câu trong transcript, audio player cần nhảy đến timestamp tương ứng."),
            (3, "Nhưng do delay network nên có độ trễ khoảng 500 mili giây."),
            (1, "Có thể pre-buffer audio để giảm delay không?"),
            (3, "Em đang thử phương án đó. Dự kiến xong trong sprint này."),
            (0, "Tốt. Deadline cho tính năng này là thứ Sáu tuần sau nhé."),
            (3, "Vâng ạ, em sẽ hoàn thành trước thứ Sáu."),
            (0, "Anh Khương hỗ trợ Hà nếu cần API endpoint mới nhé."),
            (2, "Dạ được ạ."),
        ],
        # Block 3: Budget discussion (20-30 min)
        [
            (1, "Bây giờ chúng ta bàn về ngân sách quý 3."),
            (1, "Tổng budget được duyệt là 500 triệu đồng."),
            (1, "Trong đó infrastructure chiếm 200 triệu, nhân sự 250 triệu, còn lại là chi phí vận hành."),
            (0, "Quyết định: chúng ta sẽ dùng RunPod cho GPU infrastructure."),
            (1, "Ghi nhận. Anh Khương setup RunPod account trong tuần này nhé."),
        ],
    ]

    sentences: list[SentenceInfo] = []
    t = 0.0
    for block in blocks:
        for speaker, text in block:
            duration = len(text) * 0.05 + 2.0
            sentences.append(SentenceInfo(
                text=text,
                start=round(t, 1),
                end=round(t + duration, 1),
                speaker=speaker,
            ))
            t += duration + 0.5

    # Optional filler can be added to exercise more map-reduce chunks.
    filler_topics = []

    for i, filler in enumerate(filler_topics):
        speaker = i % 4
        duration = len(filler) * 0.05 + 2.0
        sentences.append(SentenceInfo(
            text=filler,
            start=round(t, 1),
            end=round(t + duration, 1),
            speaker=speaker,
        ))
        t += duration + 0.5

    full_text_lines = [
        f"[{s.start:.1f}s] Speaker {s.speaker}: {s.text}"
        for s in sentences
    ]
    full_text = "\n".join(full_text_lines)

    logger.info(
        "Built transcript: %d sentences, %d chars, ~%.0f min",
        len(sentences), len(full_text), t / 60,
    )

    return TranscriptResult(
        key="e2e_test_60min",
        text=full_text,
        duration=t,
        sentence_info=sentences,
    )


# ------------------------------------------------------------------
# 2. Build LLM client
# ------------------------------------------------------------------

def build_groq_client():
    """Build Groq LLM client from config or env."""
    api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "meeting_config.yaml"
        )
        if os.path.exists(config_path):
            with open(config_path) as f:
                for line in f:
                    if "api_key" in line and ":" in line and not line.strip().startswith("#"):
                        api_key = line.split(":", 1)[1].strip().strip('"').strip("'")
                        break
            logger.info("API key loaded from meeting_config.yaml")

    if not api_key:
        logger.error(
            "No API key. Set GROQ_API_KEY env or add to meeting_config.yaml"
        )
        sys.exit(1)

    from meetasr.llm.groq_client import GroqClient
    return GroqClient(api_key=api_key, model="llama-3.1-8b-instant")


# ------------------------------------------------------------------
# 3. Main
# ------------------------------------------------------------------

def main() -> None:
    """Run end-to-end DocumentPlanner test."""
    logger.info("=" * 60)
    logger.info("E2E Test: DocumentPlanner — multi-pass path")
    logger.info("=" * 60)

    # Step 1: Build transcript
    transcript = build_long_path_transcript()

    # Step 2: Build LLM client
    client = build_groq_client()
    logger.info("LLM: %s (%s)", type(client).__name__, client.model)

    # Step 3: Build DocumentPlanner
    import meetasr.llm.planner
    # Force multiple chunks for this manual smoke test.
    meetasr.llm.planner.MAX_CHARS_PER_CHUNK = 800

    planner = meetasr.llm.planner.DocumentPlanner(
        client=client, temperature=0.3, max_tokens=4096,
    )

    # Step 4: Check chunk count
    full_text = planner._format_transcript(transcript)
    chunk_count = len(
        planner._chunk(
            full_text,
            planner._write_data_budget(
                {"heading": "Tóm tắt", "kind": "summary"},
                "Tài liệu",
            ),
        )
    )
    logger.info("Text: %d chars → %d chunks", len(full_text), chunk_count)

    # Step 5: Run
    logger.info("Running plan_and_write()...")
    report = planner.plan_and_write(transcript)

    # Step 6: Results
    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)
    print(f"Content kind: {report.content_kind}")
    print(f"Sections: {len(report.sections)}")
    print(f"Processing time: {report.processing_time:.2f}s")
    print(f"LLM: {report.llm_model}")

    print("\n" + "=" * 60)
    print("MARKDOWN")
    print("=" * 60)
    print(report.to_markdown())

    # Save output
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "e2e_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())

    md_path = os.path.join(output_dir, "e2e_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())

    logger.info("Saved: %s, %s", json_path, md_path)

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    checks = [
        ("Has content_kind", bool(report.content_kind)),
        ("Has sections", len(report.sections) > 0),
        ("All sections have markdown", all(s.markdown.strip() for s in report.sections)),
        ("Processing time > 0", report.processing_time > 0),
        ("Has LLM model", bool(report.llm_model)),
    ]
    ok = True
    for name, passed in checks:
        s = "✅" if passed else "❌"
        print(f"  {s} {name}")
        if not passed:
            ok = False

    print(f"\n{'✅ ALL PASS' if ok else '❌ SOME FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
