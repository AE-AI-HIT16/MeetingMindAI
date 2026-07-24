"""Deterministic call-count benchmark for Plan → Write workloads."""

from __future__ import annotations

import json

import pytest

from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.llm.planner import DocumentPlanner
from meetasr.schemas import SentenceInfo, TranscriptResult


class BenchmarkLLMClient(AbsLLMClient):
    """Return deterministic plan, map, and reduce responses."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        if "ĐỀ XUẤT tiêu đề cùng" in prompt:
            return json.dumps(
                {
                    "content_kind": "Cuộc họp dự án",
                    "outline": [
                        {"id": "s1", "heading": "Tổng quan", "kind": "summary"},
                        {
                            "id": "s2",
                            "heading": "Tiến độ triển khai",
                            "kind": "topic",
                        },
                        {
                            "id": "s3",
                            "heading": "Dữ liệu kiểm thử",
                            "kind": "topic",
                        },
                        {
                            "id": "s4",
                            "heading": "Phân công trước thứ Sáu",
                            "kind": "topic",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        if "các bản nháp rời rạc cho cùng một mục" in prompt:
            return "Nội dung cuối đã gộp và loại bỏ ý trùng."
        return "Bản nháp có căn cứ từ một phần transcript."


def _timed_transcript(duration_minutes: int) -> TranscriptResult:
    """Build one timestamped sentence per ten seconds."""
    segment_count = duration_minutes * 6
    sentence_text = (
        "Nhóm cập nhật tiến độ, xác nhận dữ liệu kiểm thử và phân công "
        "người phụ trách hoàn thành công việc trước thứ Sáu."
    )
    sentences = [
        SentenceInfo(
            text=f"{sentence_text} Mốc {index + 1}.",
            start=float(index * 10),
            end=float((index + 1) * 10),
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
def test_long_transcript_uses_real_map_reduce(duration_minutes: int) -> None:
    """Call count scales with chunks and outline sections, not one giant call."""
    client = BenchmarkLLMClient()
    planner = DocumentPlanner(client=client)
    transcript = _timed_transcript(duration_minutes)
    formatted = planner._format_transcript(transcript)
    section = {"heading": "Tóm tắt", "kind": "summary"}
    chunk_count = len(
        planner._chunk(
            formatted,
            planner._write_data_budget(section, "Cuộc họp dự án"),
        )
    )
    section_count = 4

    report = planner.plan_and_write(transcript)

    expected_calls = 1 + section_count * (chunk_count + 1)
    assert chunk_count > 1
    assert len(client.calls) == expected_calls
    assert len(report.sections) == section_count
    assert all(len(prompt) <= 8000 for prompt in client.calls)
