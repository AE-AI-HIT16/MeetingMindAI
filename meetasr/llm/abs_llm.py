"""Abstract LLM client interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class AbsLLMClient(ABC):
    """Abstract base for any LLM provider."""

    @abstractmethod
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict[str, Any]] = None,
    ) -> str:
        """Send a prompt and return the text response.

        Args:
            prompt: User message content.
            system: Optional system prompt / instruction.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Max tokens in response.
            response_format: Optional OpenAI-compatible structured output mode.

        Returns:
            LLM response as plain text string.

        Raises:
            RuntimeError: If the LLM call fails after retries.
        """
        ...
