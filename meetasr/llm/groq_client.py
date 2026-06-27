"""Groq client via OpenAI-compatible API."""

from meetasr.register import tables
from meetasr.llm.openai_client import OpenAIClient


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
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://api.groq.com/openai/v1",
            timeout=timeout,
            retry_attempts=retry_attempts,
        )