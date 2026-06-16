"""Tests for MeetingSummarizer with mock LLM client."""

import json
import pytest
from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.llm.summarizer import MeetingSummarizer
from meetasr.schemas import TranscriptResult, SentenceInfo


class MockLLMClient(AbsLLMClient):
    """Mock LLM that returns preset responses."""

    def __init__(self, responses: dict[str, str]):
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


def make_transcript() -> TranscriptResult:
    return TranscriptResult(
        key="meeting_001",
        text="Chúng ta sẽ làm tính năng A. Nam sẽ thiết kế UI vào tuần sau.",
        duration=120.0,
        sentence_info=[
            SentenceInfo(
                text="Chúng ta sẽ làm tính năng A.",
                start=0.5, end=5.0, speaker=0,
            ),
            SentenceInfo(
                text="Nam sẽ thiết kế UI vào tuần sau.",
                start=5.5, end=10.0, speaker=1,
            ),
        ],
    )


@pytest.fixture
def summarizer():
    mock = MockLLMClient({
        "tóm tắt": "Cuộc họp bàn về tính năng A.",
        "summarize": "Cuộc họp bàn về tính năng A.",
        "đề mục": json.dumps([
            {"title": "Tính năng A", "description": "Thảo luận phát triển", "start_time": 0.0, "end_time": 10.0}
        ], ensure_ascii=False),
        "topics": json.dumps([
            {"title": "Tính năng A", "description": "Thảo luận phát triển", "start_time": 0.0, "end_time": 10.0}
        ]),
        "action item": json.dumps([
            {"task": "Thiết kế UI", "assignee": "Nam", "deadline": None, "priority": "high", "mentioned_by": "Speaker 1", "timestamp": 5.5}
        ], ensure_ascii=False),
        "quyết định": json.dumps([]),
        "decisions": json.dumps([]),
    })
    return MeetingSummarizer(mock, language="vi")


class TestMeetingSummarizer:

    def test_summarize_returns_meeting_report(self, summarizer):
        transcript = make_transcript()
        report = summarizer.summarize(transcript)
        assert report.transcript is transcript
        assert isinstance(report.summary, str)
        assert isinstance(report.topics, list)
        assert isinstance(report.action_items, list)
        assert isinstance(report.decisions, list)

    def test_llm_called_for_each_component(self, summarizer):
        transcript = make_transcript()
        summarizer.summarize(transcript)
        # Should have made at least 4 LLM calls (summary, topics, actions, decisions)
        assert len(summarizer.client.calls) >= 4

    def test_invalid_json_returns_empty_list(self):
        """LLM returning bad JSON should not crash — returns empty list."""
        mock = MockLLMClient({
            "tóm tắt": "Tóm tắt tốt.",
            "summarize": "Tóm tắt tốt.",
            "đề mục": "không phải json",
            "topics": "not json",
            "action item": "{}",  # dict not list
            "quyết định": "invalid",
            "decisions": "invalid",
        })
        s = MeetingSummarizer(mock, language="vi")
        report = s.summarize(make_transcript())
        assert report.topics == []
        assert report.action_items == []
        assert report.decisions == []
        assert report.summary  # summary should still work

    def test_format_transcript_with_speakers(self, summarizer):
        transcript = make_transcript()
        text = summarizer._format_transcript(transcript)
        assert "Speaker 0:" in text
        assert "Speaker 1:" in text
        assert "[0.5s]" in text
