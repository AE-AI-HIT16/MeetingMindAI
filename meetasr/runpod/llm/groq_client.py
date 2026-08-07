"""Groq client via OpenAI-compatible API."""

from meetasr.runpod.llm.openai_client import OpenAIClient
from meetasr.runpod.register import tables


@tables.register("llm_classes", key="groq")
class GroqClient(OpenAIClient):
    """Groq LLM client.

    Uses Groq's OpenAI-compatible endpoint.

    Example:
        >>> client = GroqClient(
        ...     api_key="gsk_xxxxx",
        ...     model="llama-3.3-70b-versatile"
        ... )
        >>> client.chat("Hello")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 60,
        retry_attempts: int = 3,
        rate_limit_pacing: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://api.groq.com/openai/v1",
            timeout=timeout,
            retry_attempts=retry_attempts,
            rate_limit_pacing=rate_limit_pacing,
        )
