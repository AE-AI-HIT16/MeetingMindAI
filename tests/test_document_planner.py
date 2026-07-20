"""Tests for DocumentPlanner with mock LLM client."""

import json
import pytest
from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.llm.planner import DocumentPlanner, _parse_json_object
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

        # Count calls: should NOT be 1 (that's short path)
        assert counting.call_count > 1

        # Parse what happened: aggregate returns 3 outline sections
        # Total = N_chunks + 1 (aggregate) + K (reduce per section with data)
        # We can't predict exact N_chunks, but we know it's > 1
        n_chunks = counting.call_count - 1 - len(report.sections)
        assert n_chunks >= 1  # at least 1 chunk extracted


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
