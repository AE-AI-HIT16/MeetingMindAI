"""Ollama client — local LLM via OpenAI-compatible API."""

from meetasr.register import tables
from meetasr.llm.openai_client import OpenAIClient


@tables.register("llm_classes", key="ollama")
class OllamaClient(OpenAIClient):
    """Ollama local LLM client.

    Wraps OpenAIClient pointing to Ollama's OpenAI-compatible endpoint.
    Ollama must be running locally: https://ollama.com

    Example:
        >>> client = OllamaClient(model="llama3.2")
        >>> client.chat("Tóm tắt đoạn văn này...")
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: int = 120,
        retry_attempts: int = 2,
    ):
        """Initialize Ollama client.

        Args:
            model: Ollama model name (e.g. "llama3.2", "qwen2.5", "gemma3").
            host: Ollama server URL.
            timeout: Request timeout in seconds (longer for local LLMs).
            retry_attempts: Retry attempts on failure.
        """
        super().__init__(
            api_key="ollama",              # Ollama doesn't check key
            model=model,
            base_url=f"{host}/v1",
            timeout=timeout,
            retry_attempts=retry_attempts,
        )
