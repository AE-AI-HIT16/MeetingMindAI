"""OpenAI-compatible LLM client (OpenAI, Azure, Groq, Together, etc.)."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from meetasr.register import tables
from meetasr.llm.abs_llm import AbsLLMClient


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
    ):
        """Initialize OpenAI client.

        Args:
            api_key: API key. For Ollama use any non-empty string.
            model: Model name (e.g. "gpt-4o-mini", "llama3.2").
            base_url: Override base URL. None = OpenAI default.
            timeout: Request timeout in seconds.
            retry_attempts: Number of retries on failure.
        """
        self.model = model
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self._init_client(api_key, base_url)
        self.base_url = base_url

    def _init_client(self, api_key: str, base_url: Optional[str]):
        """Create the openai.OpenAI client."""
        try:
            from openai import OpenAI
            kwargs: dict = {"api_key": api_key, "timeout": self.timeout}
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
    ) -> str:
        """Send prompt and return response text.

        Args:
            prompt: User message.
            system: System prompt / role instruction.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            response_format: Optional OpenAI-compatible structured output mode.

        Returns:
            LLM response as string.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    request["response_format"] = response_format
                resp = self._client.chat.completions.create(**request)
                choice = resp.choices[0]
                content = _message_text(choice.message.content)
                if not content.strip():
                    logging.warning(
                        "LLM returned empty content (model=%s, finish_reason=%s, "
                        "refusal=%s).",
                        self.model,
                        getattr(choice, "finish_reason", None),
                        bool(getattr(choice.message, "refusal", None)),
                    )
                return content
            except Exception as e:
                last_error = e
                wait = _retry_wait_seconds(e, attempt)
                logging.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.retry_attempts}): "
                    f"{e}. Retrying in {wait}s..."
                )
                if attempt + 1 < self.retry_attempts:
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {self.retry_attempts} attempts: {last_error}"
        )


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
