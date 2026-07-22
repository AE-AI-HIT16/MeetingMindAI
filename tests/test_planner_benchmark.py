"""Deterministic call-count benchmark for long DocumentPlanner workloads.

The benchmark models 30, 60, and 90 minutes of timestamped speech without a
real LLM. It verifies how work scales with transcript length and guards the
planner's N extraction + 1 aggregate + K reduce call budget.
"""

from __future__ import annotations

import json

import pytest

from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.llm.planner import MAX_CHARS_PER_CHUNK, DocumentPlanner
from meetasr.llm.planner_chunk import OVERLAP_LINES, chunk_with_overlap
from meetasr.schemas import SentenceInfo, TranscriptResult


class BenchmarkLLMClient(AbsLLMClient):
    """Return valid stage-specific responses without exposing call_count."""

    def chat(self, prompt: str, **kwargs) -> str:
        if "Từ đoạn transcript sau" in prompt:
            return json.dumps(
                {
                    "summary": (
                        "Nhóm cập nhật tiến độ và thống nhất công việc tiếp theo."
                    ),
                    "action_items": [
                        {
                            "task": "Hoàn thiện hạng mục",
                            "who": "Nhóm",
                            "deadline": "Thứ Sáu",
                        }
                    ],
                    "decisions": [
                        {"content": "Giữ kế hoạch hiện tại", "made_by": "Nhóm"}
                    ],
                    "key_points": ["Tiến độ đang đúng kế hoạch"],
                    "quotes": [],
                },
                ensure_ascii=False,
            )
        if "Dưới đây là tóm tắt cấu trúc" in prompt:
            return json.dumps(
                {
                    "content_kind": "Cuộc họp dự án",
                    "outline": [
                        {
                            "id": "s1",
                            "heading": "Tóm tắt",
                            "kind": "summary",
                            "source_keys": ["summary"],
                        },
                        {
                            "id": "s2",
                            "heading": "Công việc",
                            "kind": "action_items",
                            "source_keys": ["action_items"],
                        },
                        {
                            "id": "s3",
                            "heading": "Quyết định",
                            "kind": "decisions",
                            "source_keys": ["decisions"],
                        },
                    ],
                },
                ensure_ascii=False,
            )
        if "chuyên gia tổng hợp tài liệu" in prompt:
            return "Nội dung được tổng hợp từ dữ liệu có bằng chứng."
        raise AssertionError("Benchmark received an unknown planner prompt")


def _timed_transcript(duration_minutes: int) -> TranscriptResult:
    """Build one timestamped sentence per 10 seconds for the requested duration."""
    segment_seconds = 10
    segment_count = duration_minutes * 60 // segment_seconds
    sentence_text = (
        "Nhóm cập nhật tiến độ hạng mục, xác nhận dữ liệu kiểm thử "
        "và phân công "
        "người phụ trách hoàn thành công việc trước thứ Sáu."
    )
    sentences = [
        SentenceInfo(
            text=f"{sentence_text} Mốc nội dung {index + 1}.",
            start=float(index * segment_seconds),
            end=float((index + 1) * segment_seconds),
            speaker=index % 3,
        )
        for index in range(segment_count)
    ]
    return TranscriptResult(
        key=f"benchmark_{duration_minutes}m",
        text=" ".join(sentence.text for sentence in sentences),
        duration=float(duration_minutes * 60),
        sentence_info=sentences,
    )


@pytest.mark.parametrize("duration_minutes", [30, 60, 90])
def test_long_transcript_call_budget(duration_minutes: int) -> None:
    """A 30-90 minute run reports its own exact stage-level call budget."""
    planner = DocumentPlanner(client=BenchmarkLLMClient())
    transcript = _timed_transcript(duration_minutes)

    report = planner.plan_and_write(transcript)
    metrics = planner.last_run_metrics

    assert metrics is not None
    formatted = planner._format_transcript(transcript)
    expected_chunks = len(
        chunk_with_overlap(
            formatted,
            max_chars=MAX_CHARS_PER_CHUNK,
            overlap_lines=OVERLAP_LINES,
        )
    )
    expected_sections = 3

    assert transcript.duration == duration_minutes * 60
    assert metrics.path == "long"
    assert metrics.input_chars == len(formatted)
    assert metrics.chunk_count == expected_chunks
    assert metrics.section_count == expected_sections
    assert metrics.calls_by_stage == {
        "extraction": expected_chunks,
        "aggregate": 1,
        "reduce": expected_sections,
    }
    assert metrics.llm_calls == expected_chunks + 1 + expected_sections
    assert metrics.llm_failures == 0
    assert len(report.sections) == expected_sections
