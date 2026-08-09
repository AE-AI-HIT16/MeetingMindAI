"""OpenAI-compatible LLM client (OpenAI, Azure, Groq, Together, etc.)."""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from typing import Any, Optional

from meetasr.backend.llm.abs_llm import AbsLLMClient
from meetasr.backend.register import tables


class LLMCompletionTruncatedError(RuntimeError):
    """Provider stopped because the configured completion budget was exhausted."""


@tables.register("llm_classes", key="openai")
class OpenAIClient(AbsLLMClient):
    """LLM client for OpenAI API and any OpenAI-compatible endpoint.

    Works with: OpenAI, Azure OpenAI, Groq, Together, LM Studio, Ollama.
    """

    def __init__(
        self,
        api_key: str = "sk-placeholder",
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        timeout: int = 60,
        retry_attempts: int = 3,
        rate_limit_pacing: bool = False,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: API key. For Ollama use any non-empty string.
            model: Model name (e.g. "gpt-4o-mini", "llama3.2").
            base_url: Override base URL. None = OpenAI default.
            timeout: Request timeout in seconds.
            retry_attempts: Number of retries on failure.
            rate_limit_pacing: Pace calls using provider token-limit headers.
        """
        self.model = model
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.rate_limit_pacing = rate_limit_pacing
        self._rate_limit_lock = threading.Lock()
        self._rate_limit_remaining_tokens: int | None = None
        self._rate_limit_reset_at: float | None = None
        self._init_client(api_key, base_url)
        self.base_url = base_url

    def _init_client(self, api_key: str, base_url: Optional[str]):
        """Create the openai.OpenAI client."""
        try:
            from openai import OpenAI
            # Retry in exactly one place. The OpenAI SDK otherwise retries
            # internally and then this class retries again, multiplying Groq
            # requests and making a 429 rate limit worse.
            kwargs: dict = {
                "api_key": api_key,
                "timeout": self.timeout,
                "max_retries": 0,
            }
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        except ImportError:
            raise RuntimeError("openai is not installed. Run: pip install openai")

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Send prompt and return response text.

        Args:
            prompt: User message.
            system: System prompt / role instruction.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            response_format: Optional OpenAI-compatible structured output mode.
            reasoning_effort: Optional provider-supported reasoning level.

        Returns:
            LLM response as string.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            request["response_format"] = response_format
        if reasoning_effort:
            request["reasoning_effort"] = reasoning_effort

        if getattr(self, "rate_limit_pacing", False):
            with self._rate_limit_lock:
                return self._complete_with_retries(request)
        return self._complete_with_retries(request)

    def _complete_with_retries(self, request: dict[str, Any]) -> str:
        """Complete one request, honoring pacing and bounded retries."""
        last_error: Exception | None = None
        attempts_used = 0
        for attempt in range(self.retry_attempts):
            attempts_used = attempt + 1
            try:
                self._pace_before_request(request)
                resp, headers = self._create_completion(request)
                self._update_rate_limit_state(headers)
                choice = resp.choices[0]
                content = _message_text(choice.message.content)
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    raise LLMCompletionTruncatedError(
                        "LLM completion reached max_tokens before finishing"
                    )
                if not content.strip():
                    logging.warning(
                        "LLM returned empty content (model=%s, finish_reason=%s, "
                        "refusal=%s).",
                        self.model,
                        finish_reason,
                        bool(getattr(choice.message, "refusal", None)),
                    )
                return content
            except Exception as e:
                last_error = e
                response = getattr(e, "response", None)
                self._update_rate_limit_state(
                    getattr(response, "headers", None)
                )
                wait = _retry_wait_seconds(e, attempt)
                retryable = _is_retryable_error(e)
                if attempt + 1 < self.retry_attempts and retryable:
                    logging.warning(
                        "LLM call failed (attempt %d/%d): %s. "
                        "Retrying in %ss...",
                        attempt + 1,
                        self.retry_attempts,
                        e,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logging.warning(
                        "LLM call failed (attempt %d/%d): %s. Not retrying.",
                        attempt + 1,
                        self.retry_attempts,
                        e,
                    )
                    break

        raise RuntimeError(
            f"LLM call failed after {attempts_used} attempts: {last_error}"
        )

    def _create_completion(
        self,
        request: dict[str, Any],
    ) -> tuple[Any, Any | None]:
        """Create a completion and retain raw headers when pacing is enabled."""
        completions = self._client.chat.completions
        raw_api = getattr(completions, "with_raw_response", None)
        if not getattr(self, "rate_limit_pacing", False) or raw_api is None:
            return completions.create(**request), None

        raw_response = raw_api.create(**request)
        return raw_response.parse(), getattr(raw_response, "headers", None)

    def _pace_before_request(self, request: dict[str, Any]) -> None:
        """Wait for the provider token window when the next call cannot fit."""
        if not getattr(self, "rate_limit_pacing", False):
            return
        remaining = self._rate_limit_remaining_tokens
        reset_at = self._rate_limit_reset_at
        if remaining is None or reset_at is None:
            return

        now = time.monotonic()
        if now >= reset_at:
            self._rate_limit_remaining_tokens = None
            self._rate_limit_reset_at = None
            return

        requested = _estimate_requested_tokens(request)
        if requested <= remaining:
            return

        wait = max(0.0, reset_at - now) + 0.5
        logging.info(
            "Pacing LLM request for %.2fs "
            "(estimated_tokens=%d, remaining_tokens=%d).",
            wait,
            requested,
            remaining,
        )
        time.sleep(wait)
        self._rate_limit_remaining_tokens = None
        self._rate_limit_reset_at = None

    def _update_rate_limit_state(self, headers: Any | None) -> None:
        """Store Groq-compatible remaining-token and reset headers."""
        if headers is None or not getattr(self, "rate_limit_pacing", False):
            return
        remaining_raw = headers.get("x-ratelimit-remaining-tokens")
        reset_raw = headers.get("x-ratelimit-reset-tokens")
        try:
            remaining = int(float(remaining_raw))
        except (TypeError, ValueError):
            return
        reset_seconds = _parse_duration_seconds(reset_raw)
        if reset_seconds is None:
            return
        self._rate_limit_remaining_tokens = max(0, remaining)
        self._rate_limit_reset_at = time.monotonic() + reset_seconds


def _message_text(content: object) -> str:
    """Normalize string or multipart OpenAI-compatible message content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for item in content:
        text = (
            item.get("text")
            if isinstance(item, dict)
            else getattr(item, "text", None)
        )
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _estimate_requested_tokens(request: dict[str, Any]) -> int:
    """Conservatively estimate input plus reserved output tokens for pacing."""
    messages = request.get("messages")
    chars = 0
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    chars += len(content)
    estimated_input = max(1, math.ceil(chars / 2))
    max_tokens = request.get("max_tokens", 0)
    try:
        reserved_output = max(0, int(max_tokens))
    except (TypeError, ValueError):
        reserved_output = 0
    return estimated_input + reserved_output


def _parse_duration_seconds(value: object) -> float | None:
    """Parse provider reset durations such as '7.66s' or '2m59.56s'."""
    if not isinstance(value, str) or not value.strip():
        return None
    units = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}
    matches = re.findall(r"([\d.]+)\s*([dhms])", value.lower())
    if not matches:
        return None
    try:
        return sum(float(amount) * units[unit] for amount, unit in matches)
    except (KeyError, ValueError):
        return None


def _is_retryable_error(error: Exception) -> bool:
    """Retry transient provider failures but not invalid 4xx requests."""
    if isinstance(error, LLMCompletionTruncatedError):
        return False
    response = getattr(error, "response", None)
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(response, "status_code", None)
    if status is None:
        return True
    try:
        status_code = int(status)
    except (TypeError, ValueError):
        return True
    return status_code in (408, 409, 429) or status_code >= 500


def _retry_wait_seconds(error: Exception, attempt: int) -> float:
    """Honor provider retry hints while retaining exponential backoff."""
    fallback = float(2 ** attempt)
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    retry_after = headers.get("retry-after") if headers else None
    try:
        if retry_after is not None:
            return max(fallback, float(retry_after) + 0.5)
    except (TypeError, ValueError):
        pass

    match = re.search(
        r"try again in\s+([\d.]+)\s*(ms|s)",
        str(error),
        re.IGNORECASE,
    )
    if not match:
        return fallback
    value = float(match.group(1))
    seconds = value / 1000 if match.group(2).lower() == "ms" else value
    return max(fallback, seconds + 0.5)
