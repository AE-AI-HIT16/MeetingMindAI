"""Unit tests for the Phase 2 Plan → Write DocumentPlanner."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

import meetasr.llm.planner as planner_module
from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.llm.planner import (
    MAX_LLM_INPUT_CHARS,
    DocumentPlanner,
    _parse_json_object,
)
from meetasr.schemas import SentenceInfo, TranscriptResult
from meetasr.schemas_doc import DocumentReport


class StubLLMClient(AbsLLMClient):
    """Record calls and return deterministic responses from a handler."""

    model = "stub-model"

    def __init__(self, handler: Callable[[str], str]) -> None:
        self.handler = handler
        self.calls: list[str] = []
        self.call_kwargs: list[dict] = []

    def chat(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        self.call_kwargs.append(kwargs)
        return self.handler(prompt)


def _outline_response(
    content_kind: str = "Cuộc họp công việc",
    outline: list[dict[str, str]] | None = None,
) -> str:
    return json.dumps(
        {
            "content_kind": content_kind,
            "outline": outline
            or [
                {"id": "s1", "heading": "Tổng quan", "kind": "summary"},
                {"id": "s2", "heading": "Điểm yếu hàng thủ Pháp", "kind": "topic"},
                {
                    "id": "s3",
                    "heading": "Khả năng pressing của Tây Ban Nha",
                    "kind": "topic",
                },
                {
                    "id": "s4",
                    "heading": "Điều chỉnh nhân sự hiệp hai",
                    "kind": "topic",
                },
            ],
        },
        ensure_ascii=False,
    )


def _multi_write_response(
    values: dict[str, str] | None = None,
) -> str:
    """Return the JSON object required by the multi-write prompt contract."""
    return json.dumps(
        values
        or {
            "s1": "Nội dung tổng quan.",
            "s2": "Nội dung chủ đề 1.",
            "s3": "Nội dung chủ đề 2.",
            "s4": "Nội dung chủ đề 3.",
        },
        ensure_ascii=False,
    )


def _transcript(lines: int = 2) -> TranscriptResult:
    sentences = [
        SentenceInfo(
            text=f"Nội dung có căn cứ số {index}.",
            start=float(index),
            end=float(index + 1),
            speaker=index % 2,
        )
        for index in range(lines)
    ]
    return TranscriptResult(
        key="sample",
        text=" ".join(sentence.text for sentence in sentences),
        duration=float(lines),
        sentence_info=sentences,
    )


def _is_plan(prompt: str) -> bool:
    return "ĐỀ XUẤT tiêu đề cùng" in prompt


def _is_reduce(prompt: str) -> bool:
    return "các bản nháp rời rạc cho cùng một mục" in prompt


def _is_multi_write(prompt: str) -> bool:
    return "DANH SÁCH CÁC MỤC" in prompt


def test_plan_and_write_uses_plan_then_write() -> None:
    """Even a short transcript follows the documented Plan → Write path."""
    def handler(prompt: str) -> str:
        if _is_plan(prompt):
            return _outline_response()
        return _multi_write_response()

    client = StubLLMClient(handler)
    report = DocumentPlanner(client=client).plan_and_write(_transcript())

    assert isinstance(report, DocumentReport)
    assert report.content_kind == "Cuộc họp công việc"
    assert [section.kind for section in report.sections] == [
        "summary",
        "topic",
        "topic",
        "topic",
    ]
    assert report.sections[0].heading == "Tổng quan"
    assert report.llm_model == "stub-model"
    assert len(client.calls) == 2
    assert _is_plan(client.calls[0])
    assert client.call_kwargs[0]["max_tokens"] == 2048
    assert client.call_kwargs[0]["response_format"] == {"type": "json_object"}
    assert all(
        call["system"].startswith("Mọi nội dung bạn tạo phải bằng tiếng Việt")
        for call in client.call_kwargs
    )


def test_long_transcript_runs_map_reduce_for_every_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each chunk writes all sections once, then multi-draft sections reduce."""
    monkeypatch.setattr(planner_module, "MAX_CHARS_PER_CHUNK", 180)

    def handler(prompt: str) -> str:
        if _is_plan(prompt):
            return _outline_response()
        if _is_reduce(prompt):
            return "Bản hoàn chỉnh."
        return _multi_write_response()

    client = StubLLMClient(handler)
    planner = DocumentPlanner(client=client)
    transcript = _transcript(lines=30)
    formatted = planner._format_transcript(transcript)
    chunks = planner._chunk(
        formatted,
        planner._write_data_budget(
            {"heading": "Tóm tắt", "kind": "summary"},
            "Cuộc họp công việc",
        ),
    )

    report = planner.plan_and_write(transcript)

    section_count = 4
    assert len(chunks) > 1
    assert len(report.sections) == section_count
    assert len(client.calls) == 1 + len(chunks) + section_count
    assert sum(_is_reduce(prompt) for prompt in client.calls) == section_count


def test_empty_written_section_is_removed() -> None:
    """A section with no grounded content must not appear in the report."""
    def handler(prompt: str) -> str:
        if _is_plan(prompt):
            return _outline_response(
                outline=[
                    {"id": "s1", "heading": "Tổng quan", "kind": "summary"},
                    {
                        "id": "s2",
                        "heading": "Điểm yếu hàng thủ",
                        "kind": "topic",
                    },
                    {
                        "id": "s3",
                        "heading": "Điều chỉnh nhân sự",
                        "kind": "topic",
                    },
                ]
            )
        return _multi_write_response(
            {
                "s1": "Tổng quan có căn cứ.",
                "s2": "Điểm yếu hàng thủ.",
                "s3": "",
            }
        )

    report = DocumentPlanner(client=StubLLMClient(handler)).plan_and_write(
        _transcript()
    )

    assert "Điều chỉnh nhân sự" not in [
        section.heading for section in report.sections
    ]
    assert len(report.sections) == 2


def test_invalid_plan_falls_back_without_crashing() -> None:
    """Malformed plan JSON degrades to the required report outline."""
    def handler(prompt: str) -> str:
        if _is_multi_write(prompt):
            return _multi_write_response(
                {
                    "s1": "Tóm tắt fallback.",
                    "s2": "Chủ đề fallback.",
                }
            )
        return "not-json"

    client = StubLLMClient(handler)
    report = DocumentPlanner(client=client).plan_and_write(_transcript())

    assert report.content_kind == "Tài liệu"
    assert [section.kind for section in report.sections] == [
        "summary",
        "topic",
    ]


@pytest.mark.parametrize("first_response", ["", "{}"])
def test_unusable_plan_is_regenerated_from_transcript(first_response: str) -> None:
    """Empty or incomplete completions trigger fresh planning from the source."""
    def handler(prompt: str) -> str:
        if _is_plan(prompt):
            return first_response
        if "Lần tạo kế hoạch trước không trả về nội dung" in prompt:
            return _outline_response("Pháp đối đầu Tây Ban Nha")
        return _multi_write_response()

    client = StubLLMClient(handler)
    report = DocumentPlanner(client=client).plan_and_write(_transcript())

    assert report.content_kind == "Pháp đối đầu Tây Ban Nha"
    assert len(report.sections) == 4
    assert "TRANSCRIPT:" in client.calls[1]
    assert "JSON cần sửa:" not in client.calls[1]
    assert client.call_kwargs[1]["response_format"] == {"type": "json_object"}


def test_generic_sections_are_replaced_by_concrete_topics() -> None:
    """Generic headings are omitted while transcript-specific topics survive."""
    plan = _outline_response(
        outline=[
            {"id": "s1", "heading": "Tóm tắt", "kind": "summary"},
            {"id": "s2", "heading": "Các điểm chính", "kind": "key_points"},
            {"id": "s3", "heading": "Kết luận", "kind": "conclusion"},
            {
                "id": "s4",
                "heading": "Khoảng trống ở cánh trái của Pháp",
                "kind": "analysis",
            },
        ]
    )
    client = StubLLMClient(
        lambda prompt: (
            plan
            if _is_plan(prompt)
            else _multi_write_response(
                {
                    "s1": "Nội dung tổng quan.",
                    "s2": "Khoảng trống ở cánh trái.",
                }
            )
        )
    )

    report = DocumentPlanner(client=client).plan_and_write(_transcript())

    assert [section.heading for section in report.sections] == [
        "Tổng quan",
        "Khoảng trống ở cánh trái của Pháp",
    ]
    assert [section.kind for section in report.sections] == ["summary", "topic"]


def test_invalid_plan_json_is_repaired_before_fallback() -> None:
    """A syntax-only JSON failure gets one deterministic repair attempt."""
    def handler(prompt: str) -> str:
        if _is_plan(prompt):
            return (
                '{"content_kind": "Phân tích bán kết và chung kết" '
                '"outline": []}'
            )
        if "JSON kế hoạch dưới đây bị lỗi cú pháp" in prompt:
            return _outline_response("Phân tích bán kết và chung kết")
        return _multi_write_response()

    client = StubLLMClient(handler)
    report = DocumentPlanner(client=client).plan_and_write(_transcript())

    assert report.content_kind == "Phân tích bán kết và chung kết"
    assert len(report.sections) == 4
    assert sum(
        "JSON kế hoạch dưới đây bị lỗi cú pháp" in prompt
        for prompt in client.calls
    ) == 1
    assert client.call_kwargs[0]["temperature"] == 0.0
    assert client.call_kwargs[1]["temperature"] == 0.0
    assert client.call_kwargs[0]["response_format"] == {"type": "json_object"}
    assert client.call_kwargs[1]["response_format"] == {"type": "json_object"}


def test_missing_multi_write_section_returns_partial_document() -> None:
    """An empty value in a multi-write response omits only that section."""
    def handler(prompt: str) -> str:
        if _is_plan(prompt):
            return _outline_response(
                outline=[
                    {"id": "s1", "heading": "Tổng quan", "kind": "summary"},
                    {
                        "id": "s2",
                        "heading": "Điểm yếu hàng thủ",
                        "kind": "topic",
                    },
                    {
                        "id": "s3",
                        "heading": "Điều chỉnh nhân sự",
                        "kind": "topic",
                    },
                ]
            )
        return _multi_write_response(
            {
                "s1": "Tổng quan còn dùng được.",
                "s2": "Điểm yếu hàng thủ còn dùng được.",
                "s3": "",
            }
        )

    report = DocumentPlanner(client=StubLLMClient(handler)).plan_and_write(
        _transcript()
    )

    assert "Điều chỉnh nhân sự" not in [
        section.heading for section in report.sections
    ]
    assert len(report.sections) == 2


def test_reduce_failure_preserves_chunk_drafts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed reduce call returns concatenated successful map drafts."""
    monkeypatch.setattr(planner_module, "MAX_CHARS_PER_CHUNK", 120)

    class FailingReduceClient(StubLLMClient):
        def chat(self, prompt: str, **kwargs) -> str:
            self.calls.append(prompt)
            if _is_plan(prompt):
                return _outline_response(
                    outline=[
                        {"id": "s1", "heading": "Tóm tắt", "kind": "summary"},
                        {
                            "id": "s2",
                            "heading": "Chủ đề chính",
                            "kind": "topic",
                        },
                    ]
                )
            if _is_reduce(prompt):
                raise RuntimeError("reduce unavailable")
            return _multi_write_response(
                {
                    "s1": "Bản nháp có căn cứ.",
                    "s2": "Bản nháp chủ đề có căn cứ.",
                }
            )

    report = DocumentPlanner(
        client=FailingReduceClient(lambda prompt: "")
    ).plan_and_write(_transcript(lines=20))

    assert len(report.sections) == 2
    assert report.sections[0].markdown.count("Bản nháp có căn cứ.") > 1


def test_empty_transcript_skips_llm() -> None:
    """Empty input cannot invite hallucinated output."""
    client = StubLLMClient(lambda prompt: "unexpected")
    report = DocumentPlanner(client=client).plan_and_write(
        TranscriptResult(key="empty", text="", duration=0.0)
    )

    assert report.content_kind == "Tài liệu"
    assert report.sections == []
    assert client.calls == []


def test_plan_only_and_write_one_section_support_realtime() -> None:
    """Realtime code can plan once and write one section incrementally."""
    def handler(prompt: str) -> str:
        if _is_plan(prompt):
            return _outline_response("Bài giảng")
        return _multi_write_response({"s1": "Nội dung realtime."})

    planner = DocumentPlanner(client=StubLLMClient(handler))
    content_kind, outline = planner.plan_only("Nội dung bài giảng.")
    section = planner.write_one_section(
        outline[0],
        content_kind,
        "Khái niệm quan trọng.",
    )

    assert content_kind == "Bài giảng"
    assert section.id == "s1"
    assert section.markdown == "Nội dung realtime."


def test_representative_sample_contains_head_middle_tail() -> None:
    """A long planning sample covers the overall transcript shape."""
    planner = DocumentPlanner(client=StubLLMClient(lambda prompt: ""))
    text = "A" * 3000 + "B" * 3000 + "C" * 3000
    sample = planner._representative_sample(text, 900)

    assert len(sample) <= 900
    assert "A" in sample
    assert "B" in sample
    assert "C" in sample


def test_chunk_preserves_normal_lines() -> None:
    """Normal transcript sentences are never cut between chunks."""
    planner = DocumentPlanner(client=StubLLMClient(lambda prompt: ""))
    lines = [f"line-{index}-" + "x" * 30 for index in range(20)]
    chunks = planner._chunk("\n".join(lines), max_chars=120)
    output_lines = [line for chunk in chunks for line in chunk.split("\n")]

    assert output_lines == lines
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_transcript_formatter_uses_consistent_time_ranges() -> None:
    """Planner input gives the LLM one canonical timestamp format."""
    planner = DocumentPlanner(client=StubLLMClient(lambda prompt: ""))
    formatted = planner._format_transcript(
        TranscriptResult(
            key="timestamps",
            text="",
            duration=3700,
            sentence_info=[
                SentenceInfo(
                    text="Ý kiến thứ nhất.",
                    start=206,
                    end=321,
                    speaker=2,
                ),
                SentenceInfo(
                    text="Ý kiến sau một giờ.",
                    start=3600,
                    end=3670,
                    speaker=1,
                ),
            ],
        )
    )

    assert "[03:26–05:21] Speaker 2: Ý kiến thứ nhất." in formatted
    assert "[01:00:00–01:01:10] Speaker 1: Ý kiến sau một giờ." in formatted
    assert "s]" not in formatted


def test_vietnamese_prompts_define_required_report_contract() -> None:
    """Prompt contract requires an overview and concrete transcript topics."""
    planner = DocumentPlanner(client=StubLLMClient(lambda prompt: ""))
    plan_prompt = planner._prompts["plan"]
    multi_write_prompt = planner._prompts["multi_write"]
    reduce_prompt = planner._prompts["reduce_section"]

    assert 'Mục đầu tiên bắt buộc là "Tổng quan"' in plan_prompt
    assert "Phân tích theo chủ đề" in plan_prompt
    assert '"kind": "topic"' in plan_prompt
    assert "Chọn 2-6 chủ đề" in plan_prompt
    assert "DANH SÁCH CÁC MỤC" in multi_write_prompt
    assert "đối tượng JSON hợp lệ" in multi_write_prompt
    assert '"<section_id>"' in multi_write_prompt
    assert "chuỗi rỗng" in multi_write_prompt
    assert "TIẾNG VIỆT" in reduce_prompt
    assert "NẾU kind LÀ `summary`" in reduce_prompt
    assert "NẾU kind LÀ `topic`" in reduce_prompt
    assert "Gộp các bản nháp" in reduce_prompt
    assert "[MM:SS–MM:SS]" in reduce_prompt


def test_report_title_is_not_overlong() -> None:
    """LLM titles are bounded to fourteen words."""
    long_title = " ".join(f"từ{index}" for index in range(20))
    client = StubLLMClient(
        lambda prompt: _outline_response(long_title)
        if _is_plan(prompt)
        else _multi_write_response()
    )
    report = DocumentPlanner(client=client).plan_and_write(_transcript())

    assert len(report.content_kind.split()) == 14


def test_all_llm_prompts_respect_character_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan, map, and reduce prompts stay within the 8,000-char constraint."""
    monkeypatch.setattr(planner_module, "MAX_CHARS_PER_CHUNK", 6000)

    def handler(prompt: str) -> str:
        if _is_plan(prompt):
            return _outline_response(
                outline=[
                    {"id": "s1", "heading": "Tóm tắt", "kind": "summary"},
                    {"id": "s2", "heading": "Chủ đề chính", "kind": "topic"},
                ]
            )
        if _is_reduce(prompt):
            return "Bản hoàn chỉnh."
        return _multi_write_response(
            {
                "s1": "x" * 7000,
                "s2": "y" * 7000,
            }
        )

    client = StubLLMClient(handler)
    DocumentPlanner(client=client).plan_and_write(_transcript(lines=500))

    assert all(len(prompt) <= MAX_LLM_INPUT_CHARS for prompt in client.calls)


@pytest.mark.parametrize(
    ("kind", "topic_heading"),
    [
        ("Cuộc họp", "Tiến độ giao diện"),
        ("Bài giảng", "Khái niệm động lượng"),
        ("Podcast", "Chiến thuật pressing"),
    ],
)
def test_open_outline_varies_by_content(kind: str, topic_heading: str) -> None:
    """The planner preserves content-specific topic headings."""
    response = _outline_response(
        kind,
        [
            {"id": "s1", "heading": "Tổng quan", "kind": "summary"},
            {"id": "s2", "heading": topic_heading, "kind": "custom_kind"},
        ],
    )
    client = StubLLMClient(
        lambda prompt: (
            response
            if _is_plan(prompt)
            else _multi_write_response(
                {
                    "s1": "Nội dung tổng quan.",
                    "s2": "Nội dung chủ đề.",
                }
            )
        )
    )
    report = DocumentPlanner(client=client).plan_and_write(_transcript())

    assert report.content_kind == kind
    assert report.sections[1].heading == topic_heading
    assert report.sections[1].kind == "topic"


def test_configured_language_is_used_to_load_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt loading follows the planner language configuration."""
    from meetasr.llm.llm_utils import prompts

    captured: dict[str, str] = {}

    def fake_loader(language: str) -> dict[str, str]:
        captured["language"] = language
        return {
            "plan": "{transcript_sample}",
            "multi_write": "{content_kind}{sections_list}{chunk}",
            "reduce_section": "{content_kind}{heading}{kind}{chunk}",
        }

    monkeypatch.setattr(prompts, "load_generic_prompts", fake_loader)
    planner = DocumentPlanner(
        client=StubLLMClient(lambda prompt: ""),
        language="en",
    )

    assert planner.language == "en"
    assert captured["language"] == "en"


def test_parse_json_object_variants() -> None:
    assert _parse_json_object('{"key": "value"}')["key"] == "value"
    assert _parse_json_object('```json\n{"key": "value"}\n```')["key"] == "value"
    assert _parse_json_object('Result: {"key": "value"} done')["key"] == "value"
    with pytest.raises(ValueError, match="empty response"):
        _parse_json_object(" \n ")
    with pytest.raises(ValueError, match="JSON object"):
        _parse_json_object("[]")


def test_document_report_serialization() -> None:
    """The generic schema emits JSON and Markdown without Phase 1 fields."""
    client = StubLLMClient(
        lambda prompt: (
            _outline_response() if _is_plan(prompt) else _multi_write_response()
        )
    )
    report = DocumentPlanner(client=client).plan_and_write(_transcript())

    payload = json.loads(report.to_json())
    markdown = report.to_markdown()
    assert payload["sections"][0]["kind"] == "summary"
    assert "found" not in payload["sections"][0]
    assert "# Cuộc họp công việc" in markdown
    assert "## Tổng quan" in markdown
