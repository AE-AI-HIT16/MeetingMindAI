import json
import os

import pytest
from dotenv import load_dotenv

from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.llm.summarizer import MeetingSummarizer
from meetasr.schemas import SentenceInfo, TranscriptResult

load_dotenv()


class _ChatClient(AbsLLMClient):
    """Simple wrapper test client using Groq/OpenAI-compatible interface."""

    def __init__(self, client):
        self.client = client

    def chat(self, prompt: str, temperature: float = 0.3, max_tokens: int = 4096) -> str:
        return self.client.chat(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def load_mock_transcript(path: str) -> TranscriptResult:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return TranscriptResult(
        key=data.get("key", ""),
        duration=data.get("duration", 0.0),
        text=data.get("text", ""),
        sentence_info=[
            SentenceInfo(**s) for s in data.get("sentence_info", [])
        ],
    )


@pytest.mark.skipif(
    os.environ.get("MEETASR_RUN_LIVE_TESTS") != "1",
    reason="calls the live Groq API; set MEETASR_RUN_LIVE_TESTS=1",
)
def test_meeting_summarizer_end_to_end():
    from meetasr.llm.groq_client import GroqClient

    # ----------------------------
    # 1. Init LLM client
    # ----------------------------
    raw_client = GroqClient(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

    client = _ChatClient(raw_client)

    # ----------------------------
    # 2. Init summarizer
    # ----------------------------
    summarizer = MeetingSummarizer(
        client=client,
        language="vi",
        temperature=0.3,
        max_tokens=2048
    )

    # ----------------------------
    # 3. Load test data
    # ----------------------------
    transcript = load_mock_transcript(
        "tests/data/consultation_mock_v2.json"
    )

    # ----------------------------
    # 4. Run pipeline
    # ----------------------------
    report = summarizer.summarize(
        transcript=transcript,
        asr_model="mock-asr",
        llm_model="groq-test"
    )

    # ----------------------------
    # 5. Assertions
    # ----------------------------

    # basic structure
    assert report is not None
    assert isinstance(report.summary, str)
    assert len(report.summary) > 0

    assert isinstance(report.topics, list)
    assert isinstance(report.action_items, list)
    assert isinstance(report.decisions, list)

    # topics validation
    if report.topics:
        t = report.topics[0]
        assert hasattr(t, "title")

    # action items validation
    if report.action_items:
        a = report.action_items[0]
        assert hasattr(a, "task")

    # decisions validation
    if report.decisions:
        d = report.decisions[0]
        assert hasattr(d, "content")

    # metadata
    assert report.language == "vi"
    assert report.processing_time > 0

    print("\n===== SUMMARY =====\n")
    print(report.summary)

    print("\n===== TOPICS =====\n")
    for t in report.topics:
        print(t)

    print("\n===== ACTION ITEMS =====\n")
    for a in report.action_items:
        print(a)

    print("\n===== DECISIONS =====\n")
    for d in report.decisions:
        print(d)
