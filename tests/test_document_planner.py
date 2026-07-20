"""Tests for DocumentPlanner with mock LLM client."""

import json

import pytest

from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.llm.planner import (
    MAX_AGGREGATE_DATA_CHARS,
    MAX_REDUCE_DATA_CHARS,
    SHORT_TRANSCRIPT_CHARS,
    DocumentPlanner,
    _parse_json_object,
)
from meetasr.llm.planner_chunk import OVERLAP_LINES, chunk_with_overlap
from meetasr.llm.planner_validation import normalize_extraction, normalize_outline
from meetasr.schemas import TranscriptResult, SentenceInfo
from meetasr.schemas_doc import DocumentReport


# ------------------------------------------------------------------
# Mock LLM
# ------------------------------------------------------------------

class MockLLMClient(AbsLLMClient):
    """Mock LLM that returns preset responses based on prompt keywords."""

    def __init__(self, responses: dict[str, str]) -> None:
        """Args:
            responses: Map of keyword → response string.
                If prompt contains keyword, return that response.
        """
        self.responses = responses
        self.calls: list[str] = []

    def chat(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        for keyword, response in self.responses.items():
            if keyword in prompt:
                return response
        return ""


class CountingLLMClient(AbsLLMClient):
    """Mock LLM that counts calls per keyword category."""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.calls: list[str] = []

    def chat(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        self.calls.append(prompt)
        for keyword, response in self.responses.items():
            if keyword in prompt:
                return response
        return ""


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _short_transcript() -> TranscriptResult:
    """Transcript below the character threshold triggers the short path."""
    return TranscriptResult(
        key="short_meeting",
        text="Chúng ta sẽ làm tính năng A. Nam sẽ thiết kế UI.",
        duration=120.0,
        sentence_info=[
            SentenceInfo(text="Chúng ta sẽ làm tính năng A.", start=0.5, end=5.0, speaker=0),
            SentenceInfo(text="Nam sẽ thiết kế UI.", start=5.5, end=10.0, speaker=1),
        ],
    )


def _long_transcript(n_lines: int = 400) -> TranscriptResult:
    """Transcript above the character threshold triggers the long path."""
    lines = [f"[{i * 5.0:.1f}s] Speaker {i % 3}: Đây là câu số {i} với nội dung khá dài để đảm bảo chunk." for i in range(n_lines)]
    text = "\n".join(lines)
    sentences = [
        SentenceInfo(
            text=f"Đây là câu số {i} với nội dung khá dài để đảm bảo chunk.",
            start=i * 5.0,
            end=(i + 1) * 5.0,
            speaker=i % 3,
        )
        for i in range(n_lines)
    ]
    return TranscriptResult(
        key="long_meeting",
        text=text,
        duration=n_lines * 5.0,
        sentence_info=sentences,
    )


def _single_pass_response() -> str:
    """Response mimicking single_pass_vi.txt output."""
    return json.dumps({
        "content_kind": "Cuộc họp công việc",
        "sections": [
            {
                "id": "s1",
                "heading": "Tóm tắt",
                "kind": "summary",
                "markdown": "Cuộc họp bàn về tính năng A, Nam phụ trách UI.",
            },
            {
                "id": "s2",
                "heading": "Action Items",
                "kind": "action_items",
                "markdown": "- Nam: Thiết kế UI",
            },
        ],
    }, ensure_ascii=False)


def _structured_extract_response() -> str:
    """Response for structured_extract_vi.txt."""
    return json.dumps({
        "summary": "Thảo luận về tiến độ dự án.",
        "action_items": [{"task": "Review code", "who": "Khương", "deadline": "Thứ 6"}],
        "decisions": [{"content": "Dời deadline", "made_by": "Team"}],
        "key_points": ["Tiến độ đúng hạn"],
        "quotes": [],
    }, ensure_ascii=False)


def _aggregate_plan_response() -> str:
    """Response for aggregate_plan_vi.txt."""
    return json.dumps({
        "content_kind": "Cuộc họp công việc",
        "outline": [
            {"id": "s1", "heading": "Tóm tắt", "kind": "summary", "source_keys": ["summary"]},
            {"id": "s2", "heading": "Action Items", "kind": "action_items", "source_keys": ["action_items"]},
            {"id": "s3", "heading": "Quyết định", "kind": "decisions", "source_keys": ["decisions"]},
        ],
    }, ensure_ascii=False)


# ------------------------------------------------------------------
# Test: Short transcript → 1 LLM call
# ------------------------------------------------------------------

class TestShortTranscript:

    def test_short_transcript_single_call(self):
        """A short transcript uses one single-pass LLM call."""
        mock = MockLLMClient({
            # single_pass prompt contains the full transcript
            "Chúng ta sẽ làm": _single_pass_response(),
        })
        planner = DocumentPlanner(client=mock)
        report = planner.plan_and_write(_short_transcript())

        assert isinstance(report, DocumentReport)
        assert report.content_kind == "Cuộc họp công việc"
        assert len(report.sections) == 2
        assert report.sections[0].heading == "Tóm tắt"
        assert report.sections[0].found is True
        assert "tính năng A" in report.sections[0].markdown
        assert report.language == "vi"
        # Only 1 LLM call for short path
        assert len(mock.calls) == 1

    def test_short_path_fallback_on_error(self):
        """If LLM returns invalid JSON, short path falls back gracefully."""
        mock = MockLLMClient({
            "any": "this is not json at all {{{{",
        })
        planner = DocumentPlanner(client=mock)
        report = planner.plan_and_write(_short_transcript())

        assert isinstance(report, DocumentReport)
        assert report.content_kind == "Tài liệu"
        assert report.sections == []

    def test_empty_transcript_does_not_call_llm(self):
        """An empty transcript returns an empty report without inviting hallucination."""
        mock = MockLLMClient({})
        planner = DocumentPlanner(client=mock)
        report = planner.plan_and_write(
            TranscriptResult(key="empty", text="", duration=0.0)
        )

        assert report.content_kind == "Tài liệu"
        assert report.sections == []
        assert mock.calls == []

    def test_malformed_sections_are_ignored_and_duplicate_ids_are_fixed(self):
        """Single-pass output is normalized before building the report."""
        response = json.dumps({
            "content_kind": "Podcast",
            "sections": [
                "invalid",
                {"id": "s1", "heading": "A", "kind": "summary", "markdown": "One"},
                {"id": "s1", "heading": "B", "kind": "quotes", "markdown": "Two"},
                {"id": "s4", "heading": "Empty", "kind": "notes", "markdown": ""},
            ],
        })
        mock = MockLLMClient({"Chúng ta sẽ làm": response})
        report = DocumentPlanner(client=mock).plan_and_write(_short_transcript())

        assert [section.id for section in report.sections] == ["s1", "s3"]
        assert [section.heading for section in report.sections] == ["A", "B"]

    def test_short_path_prompt_stays_within_context_budget(self):
        """The largest single-pass input must stay below 8,000 characters."""
        mock = MockLLMClient({"a": _single_pass_response()})
        transcript = TranscriptResult(
            key="context-limit",
            text="a" * (SHORT_TRANSCRIPT_CHARS - 1),
            duration=1.0,
        )
        DocumentPlanner(client=mock).plan_and_write(transcript)
        assert len(mock.calls) == 1
        assert len(mock.calls[0]) <= 8000


# ------------------------------------------------------------------
# Test: Long transcript → N extraction + 1 aggregate + K reduce
# ------------------------------------------------------------------

class TestLongTranscript:

    def test_long_transcript_structured_extraction(self):
        """A long transcript uses extract, aggregate, and reduce calls."""
        mock = MockLLMClient({
            # Structured extract prompt contains "chunk"
            "Đây là câu số": _structured_extract_response(),
            # Aggregate plan prompt contains "structured_summaries"
            "tóm tắt cấu trúc": _aggregate_plan_response(),
            # Reduce prompt contains "raw_data"
            "dữ liệu thô": "Nội dung markdown cho section.",
        })
        planner = DocumentPlanner(client=mock)
        report = planner.plan_and_write(_long_transcript())

        assert isinstance(report, DocumentReport)
        assert report.content_kind == "Cuộc họp công việc"
        assert len(report.sections) >= 1
        for sec in report.sections:
            assert sec.found is True
            assert sec.markdown.strip()

    def test_call_count_long(self):
        """For long transcript: total calls = N(extract) + 1(aggregate) + K(reduce)."""
        counting = CountingLLMClient({
            "Đây là câu số": _structured_extract_response(),
            "tóm tắt cấu trúc": _aggregate_plan_response(),
            "dữ liệu thô": "Section markdown content.",
        })
        planner = DocumentPlanner(client=counting)
        report = planner.plan_and_write(_long_transcript())

        formatted = planner._format_transcript(_long_transcript())
        n_chunks = len(chunk_with_overlap(formatted, overlap_lines=OVERLAP_LINES))
        assert len(report.sections) == 3
        assert counting.call_count == n_chunks + 1 + len(report.sections)

    def test_all_long_path_prompts_stay_within_context_budget(self):
        """Every long-path prompt must remain below the 8,000-character limit."""
        counting = CountingLLMClient({
            "Đây là câu số": _structured_extract_response(),
            "tóm tắt cấu trúc": _aggregate_plan_response(),
            "dữ liệu thô": "Section markdown content.",
        })
        DocumentPlanner(client=counting).plan_and_write(_long_transcript(1000))
        assert all(len(prompt) <= 8000 for prompt in counting.calls)


# ------------------------------------------------------------------
# Test: No hallucination — empty sections filtered
# ------------------------------------------------------------------

class TestNoHallucination:

    def test_no_hallucination_empty_section(self):
        """Extraction returns empty → section should be excluded, not fabricated."""
        empty_extract = json.dumps({
            "summary": "",
            "action_items": [],
            "decisions": [],
            "key_points": [],
            "quotes": [],
        }, ensure_ascii=False)

        mock = MockLLMClient({
            "Đây là câu số": empty_extract,
            "tóm tắt cấu trúc": _aggregate_plan_response(),
            "dữ liệu thô": "",  # reduce returns empty
        })
        planner = DocumentPlanner(client=mock)
        report = planner.plan_and_write(_long_transcript())

        # This must not be a vacuous loop: no source data means no sections.
        assert report.sections == []

    def test_non_summary_section_does_not_fallback_to_summaries(self):
        """Missing action evidence must not be replaced with general summaries."""
        planner = DocumentPlanner(client=MockLLMClient({}))
        raw_data = planner._gather_for_section(
            {
                "id": "s2",
                "heading": "Công việc",
                "kind": "action_items",
                "source_keys": ["action_items"],
            },
            [{"chunk_index": 0, "summary": "Có thảo luận chung."}],
        )
        assert raw_data == ""

    def test_nested_extra_fields_are_available_to_sections(self):
        """Open-ended fields under extra are flattened for planning and reducing."""
        extraction = normalize_extraction(
            {"summary": "Bàn ngân sách.", "extra": {"ngan_sach": "500 triệu"}},
            chunk_index=0,
        )
        planner = DocumentPlanner(client=MockLLMClient({}))
        raw_data = planner._gather_for_section(
            {
                "id": "s2",
                "heading": "Ngân sách",
                "kind": "finance",
                "source_keys": ["ngan_sach"],
            },
            [extraction],
        )
        assert raw_data == "ngan_sach: 500 triệu"

    def test_non_summary_outline_cannot_use_summary_as_evidence(self):
        """Malformed source keys must not route summaries into factual sections."""
        outline = normalize_outline([
            {
                "id": "s2",
                "heading": "Công việc",
                "kind": "action_items",
                "source_keys": ["summary", "action_items"],
            }
        ])
        assert outline[0]["source_keys"] == ["action_items"]

    def test_aggregate_and_reduce_data_are_bounded(self):
        """Internal context passed to final LLM calls must stay bounded."""
        planner = DocumentPlanner(client=MockLLMClient({}))
        extractions = [
            {"chunk_index": index, "summary": "x" * 1000, "facts": ["y" * 1000]}
            for index in range(20)
        ]
        raw_data = planner._gather_for_section(
            {"kind": "facts", "source_keys": ["facts"]}, extractions
        )
        assert len(raw_data) <= MAX_REDUCE_DATA_CHARS

        # The aggregate prompt is captured even when its response is invalid.
        planner._aggregate_plan(extractions)
        aggregate_prompt = planner.client.calls[-1]
        assert len(aggregate_prompt) <= MAX_AGGREGATE_DATA_CHARS + 1800


# ------------------------------------------------------------------
# Test: Different content types → different outlines
# ------------------------------------------------------------------

class TestDifferentContentTypes:

    def test_different_content_types(self):
        """Three different content kinds produce reports with correct content_kind."""
        for kind_name in ["Cuộc họp công việc", "Bài giảng", "Podcast"]:
            aggregate = json.dumps({
                "content_kind": kind_name,
                "outline": [
                    {"id": "s1", "heading": "Tóm tắt", "kind": "summary",
                     "source_keys": ["summary"]},
                ],
            }, ensure_ascii=False)

            mock = MockLLMClient({
                "Đây là câu số": _structured_extract_response(),
                "tóm tắt cấu trúc": aggregate,
                "dữ liệu thô": "Content here.",
            })
            planner = DocumentPlanner(client=mock)
            report = planner.plan_and_write(_long_transcript())
            assert report.content_kind == kind_name


# ------------------------------------------------------------------
# Test: Helper — _parse_json_object
# ------------------------------------------------------------------

class TestParseJson:

    def test_parse_plain_json(self):
        raw = '{"content_kind": "Test", "sections": []}'
        result = _parse_json_object(raw)
        assert result["content_kind"] == "Test"

    def test_parse_json_in_code_block(self):
        raw = '```json\n{"content_kind": "Test"}\n```'
        result = _parse_json_object(raw)
        assert result["content_kind"] == "Test"

    def test_parse_json_with_surrounding_text(self):
        raw = 'Here is the result: {"key": "value"} end.'
        result = _parse_json_object(raw)
        assert result["key"] == "value"

    def test_parse_invalid_json_raises(self):
        with pytest.raises(Exception):
            _parse_json_object("not json at all")

    def test_parse_json_array_raises(self):
        with pytest.raises(ValueError, match="JSON object"):
            _parse_json_object("[]")


def test_configured_language_is_used_to_load_prompts(monkeypatch):
    """DocumentPlanner must load the configured prompt language."""
    from meetasr.llm.llm_utils import prompts

    captured: dict[str, str] = {}

    def fake_loader(language: str) -> dict[str, str]:
        captured["language"] = language
        return {
            "single_pass": "{transcript}",
            "structured_extract": "{chunk}",
            "aggregate_plan": "{structured_summaries}",
            "reduce_section": "{heading}{kind}{content_kind}{raw_data}",
        }

    monkeypatch.setattr(prompts, "load_generic_prompts", fake_loader)
    planner = DocumentPlanner(client=MockLLMClient({}), language="en")
    assert planner.language == "en"
    assert captured["language"] == "en"


# ------------------------------------------------------------------
# Test: DocumentReport serialization
# ------------------------------------------------------------------

class TestDocumentReport:

    def test_to_markdown(self):
        """DocumentReport.to_markdown() produces valid markdown."""
        mock = MockLLMClient({
            "Chúng ta sẽ làm": _single_pass_response(),
        })
        planner = DocumentPlanner(client=mock)
        report = planner.plan_and_write(_short_transcript())
        md = report.to_markdown()
        assert "# Cuộc họp công việc" in md
        assert "## Tóm tắt" in md

    def test_to_json(self):
        """DocumentReport.to_json() returns valid JSON."""
        mock = MockLLMClient({
            "Chúng ta sẽ làm": _single_pass_response(),
        })
        planner = DocumentPlanner(client=mock)
        report = planner.plan_and_write(_short_transcript())
        j = report.to_json()
        parsed = json.loads(j)
        assert "content_kind" in parsed
        assert "sections" in parsed
