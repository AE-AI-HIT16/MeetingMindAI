"""Unit tests for OpenAI-compatible retry timing."""

import sys
import threading
from types import SimpleNamespace

import pytest

import meetasr.llm.openai_client as openai_client_module
from meetasr.llm.groq_client import GroqClient
from meetasr.llm.openai_client import (
    OpenAIClient,
    _parse_duration_seconds,
    _retry_wait_seconds,
)


class RateLimitError(Exception):
    """Minimal provider error carrying optional response headers."""

    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        headers = {"retry-after": retry_after} if retry_after is not None else {}
        self.response = type("Response", (), {"headers": headers})()


def test_retry_wait_uses_provider_header() -> None:
    error = RateLimitError("rate limited", retry_after="2.91")
    assert round(_retry_wait_seconds(error, attempt=0), 2) == 3.41


def test_retry_wait_parses_provider_message() -> None:
    error = RateLimitError("Please try again in 2.91s.")
    assert round(_retry_wait_seconds(error, attempt=0), 2) == 3.41


def test_retry_wait_keeps_larger_exponential_delay() -> None:
    error = RateLimitError("Please try again in 500ms.")
    assert _retry_wait_seconds(error, attempt=2) == 4.0


def test_chat_forwards_json_mode_and_reads_multipart_content() -> None:
    """Structured mode reaches the provider and multipart text is preserved."""
    captured: dict = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(
                content=[{"type": "text", "text": '{"ok": true}'}],
                refusal=None,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "stub-model"
    client.retry_attempts = 1
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    result = client.chat(
        "Trả JSON.",
        response_format={"type": "json_object"},
        reasoning_effort="low",
    )

    assert result == '{"ok": true}'
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["reasoning_effort"] == "low"


def test_paced_chat_reads_raw_rate_limit_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Groq-style raw headers update pacing state after a successful call."""
    captured: dict = {}

    class RawResponse:
        headers = {
            "x-ratelimit-remaining-tokens": "321",
            "x-ratelimit-reset-tokens": "2.5s",
        }

        @staticmethod
        def parse():
            message = SimpleNamespace(content="ok", refusal=None)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=message, finish_reason="stop")
                ]
            )

    class RawCompletions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return RawResponse()

    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "stub-model"
    client.retry_attempts = 1
    client.rate_limit_pacing = True
    client._rate_limit_lock = threading.Lock()
    client._rate_limit_remaining_tokens = None
    client._rate_limit_reset_at = None
    client._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=RawCompletions(),
            )
        )
    )
    monkeypatch.setattr(openai_client_module.time, "monotonic", lambda: 100.0)

    assert client.chat("hello", max_tokens=100) == "ok"
    assert captured["max_tokens"] == 100
    assert client._rate_limit_remaining_tokens == 321
    assert client._rate_limit_reset_at == 102.5


def test_pacing_waits_until_token_window_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request larger than remaining quota waits before it is sent."""
    sleeps: list[float] = []
    client = OpenAIClient.__new__(OpenAIClient)
    client.rate_limit_pacing = True
    client._rate_limit_remaining_tokens = 100
    client._rate_limit_reset_at = 105.0
    monkeypatch.setattr(openai_client_module.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(openai_client_module.time, "sleep", sleeps.append)

    client._pace_before_request(
        {
            "messages": [{"role": "user", "content": "x" * 1000}],
            "max_tokens": 1600,
        }
    )

    assert sleeps == [5.5]
    assert client._rate_limit_remaining_tokens is None
    assert client._rate_limit_reset_at is None


def test_duration_parser_supports_compound_reset_headers() -> None:
    assert _parse_duration_seconds("2m59.56s") == 179.56
    assert _parse_duration_seconds("7.66s") == 7.66
    assert _parse_duration_seconds(None) is None


def test_invalid_request_is_not_retried() -> None:
    """A provider 400 reaches planner fallback without repeated API calls."""
    calls = 0

    class BadRequestError(Exception):
        status_code = 400
        response = SimpleNamespace(status_code=400, headers={})

    class Completions:
        @staticmethod
        def create(**kwargs):
            nonlocal calls
            calls += 1
            raise BadRequestError("invalid JSON request")

    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "stub-model"
    client.retry_attempts = 3
    client.rate_limit_pacing = False
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    with pytest.raises(RuntimeError, match="invalid JSON request"):
        client.chat("hello")

    assert calls == 1


def test_length_truncation_is_exposed_without_repeating_same_call() -> None:
    """Planner fallback handles truncation instead of retrying the same cap."""
    calls = 0

    class Completions:
        @staticmethod
        def create(**kwargs):
            nonlocal calls
            calls += 1
            message = SimpleNamespace(content="partial", refusal=None)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=message, finish_reason="length")
                ]
            )

    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "stub-model"
    client.retry_attempts = 3
    client.rate_limit_pacing = False
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    with pytest.raises(RuntimeError, match="reached max_tokens"):
        client.chat("hello")

    assert calls == 1


def test_groq_enables_rate_limit_pacing_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        OpenAIClient,
        "_init_client",
        lambda self, api_key, base_url: None,
    )

    client = GroqClient(api_key="test-key")

    assert client.rate_limit_pacing is True


def test_openai_sdk_retries_are_disabled(monkeypatch) -> None:
    """Only OpenAIClient's bounded retry loop should retry provider calls."""
    captured: dict = {}

    class StubOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=StubOpenAI),
    )

    OpenAIClient(api_key="test-key", retry_attempts=3)

    assert captured["max_retries"] == 0
