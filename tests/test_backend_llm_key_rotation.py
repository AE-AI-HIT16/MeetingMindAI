"""Groq/OpenAI-compatible client rotates API keys when one is exhausted."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from meetasr.backend.llm import openai_client
from meetasr.backend.llm.openai_client import (
    LLMQuotaExhaustedError,
    OpenAIClient,
    split_api_keys,
)


class FakeAPIError(Exception):
    def __init__(self, status_code: int, message: str = "rate limited"):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers={})


def _ok(text: str):
    message = SimpleNamespace(content=text, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(openai_client, "_key_cooldowns", {})
    monkeypatch.setattr(openai_client.time, "sleep", lambda s: None)


def _client(api_key: str, behaviour: dict[str, list]) -> tuple[OpenAIClient, list[str]]:
    """behaviour: key -> list of results (Exception or str) consumed in order."""
    used: list[str] = []
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "m"
    client.timeout = 5
    client.retry_attempts = 3
    client.rate_limit_pacing = False
    client._rate_limit_remaining_tokens = None
    client._rate_limit_reset_at = None
    client.base_url = None

    def init_client(key, base_url):
        def create(**request):
            used.append(key)
            result = behaviour[key].pop(0)
            if isinstance(result, Exception):
                raise result
            return _ok(result)
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    client._init_client = init_client
    client._api_keys = split_api_keys(api_key)
    client._key_index = 0
    init_client(client._api_keys[0], None)
    return client, used


def test_split_api_keys_accepts_commas_and_newlines():
    assert split_api_keys(" k1, k2\nk3 ,") == ["k1", "k2", "k3"]
    assert split_api_keys("single") == ["single"]


def test_rate_limited_key_switches_to_next_key_immediately():
    client, used = _client("k1,k2", {"k1": [FakeAPIError(429)], "k2": ["xin chào"]})

    assert client.chat("hi") == "xin chào"
    assert used == ["k1", "k2"]


def test_revoked_key_is_skipped_on_later_calls():
    client, used = _client(
        "k1,k2",
        {"k1": [FakeAPIError(401, "invalid api key")], "k2": ["a", "b"]},
    )
    client.chat("1")
    client.chat("2")

    assert used == ["k1", "k2", "k2"]


def test_all_keys_exhausted_raises_quota_error():
    client, _ = _client("k1,k2", {"k1": [FakeAPIError(429)], "k2": [FakeAPIError(429)]})

    with pytest.raises(LLMQuotaExhaustedError):
        client.chat("hi")


def test_single_key_still_retries_429_then_reports_exhaustion():
    client, used = _client(
        "k1", {"k1": [FakeAPIError(429), FakeAPIError(429), FakeAPIError(429)]}
    )

    with pytest.raises(LLMQuotaExhaustedError):
        client.chat("hi")
    assert used == ["k1", "k1", "k1"]


def test_single_key_transient_429_recovers():
    client, used = _client("k1", {"k1": [FakeAPIError(429), "ok"]})

    assert client.chat("hi") == "ok"
    assert used == ["k1", "k1"]


def test_server_errors_do_not_rotate_keys():
    client, used = _client("k1,k2", {"k1": [FakeAPIError(500), "ok"], "k2": []})

    assert client.chat("hi") == "ok"
    assert used == ["k1", "k1"]


def test_planner_surfaces_quota_exhaustion_instead_of_empty_document():
    from meetasr.backend.llm.planner import DocumentPlanner
    from meetasr.backend.schemas import SentenceInfo, TranscriptResult

    class ExhaustedClient:
        def chat(self, *args, **kwargs):
            raise LLMQuotaExhaustedError("all keys exhausted")

    transcript = TranscriptResult(
        key="t", text="xin chào mọi người", duration=10.0,
        sentence_info=[SentenceInfo(text="xin chào mọi người", start=0.0, end=2.0, speaker=0)],
    )
    with pytest.raises(LLMQuotaExhaustedError):
        DocumentPlanner(client=ExhaustedClient()).plan_and_write(transcript)
