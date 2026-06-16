"""OpenAI-compatible LLM client (OpenAI, Azure, Groq, Together, etc.)."""

from __future__ import annotations

import logging
import time
from typing import Optional

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
    ) -> str:
        """Send prompt and return response text.

        Args:
            prompt: User message.
            system: System prompt / role instruction.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.

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
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logging.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.retry_attempts}): "
                    f"{e}. Retrying in {wait}s..."
                )
                time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {self.retry_attempts} attempts: {last_error}"
        )
