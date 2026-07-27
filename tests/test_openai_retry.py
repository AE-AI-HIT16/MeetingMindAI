"""Unit tests for OpenAI-compatible retry timing."""

import sys
from types import SimpleNamespace

from meetasr.llm.openai_client import OpenAIClient, _retry_wait_seconds


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
    )

    assert result == '{"ok": true}'
    assert captured["response_format"] == {"type": "json_object"}


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
