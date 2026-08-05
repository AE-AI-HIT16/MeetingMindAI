"""Groq client via OpenAI-compatible API với hỗ trợ xoay vòng API key tự động."""

from __future__ import annotations

import logging
import time
from typing import Any

from meetasr.llm.openai_client import OpenAIClient, _is_retryable_error, _retry_wait_seconds
from meetasr.register import tables

logger = logging.getLogger(__name__)


@tables.register("llm_classes", key="groq")
class GroqClient(OpenAIClient):
    """Groq LLM client với tính năng xoay vòng API key tự động.

    Khi một key bị rate limit (HTTP 429), client tự động chuyển sang
    key tiếp theo trong danh sách mà không cần restart server.

    Example — 1 key:
        >>> client = GroqClient(api_key="gsk_key1")

    Example — nhiều key (tự động xoay khi hết hạn mức):
        >>> client = GroqClient(api_key="gsk_key1,gsk_key2,gsk_key3")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 60,
        retry_attempts: int = 3,
        rate_limit_pacing: bool = True,
    ):
        # Hỗ trợ cả 1 key lẫn nhiều key (phân tách bằng dấu phẩy)
        keys = [k.strip() for k in api_key.split(",") if k.strip()]
        self._api_keys: list[str] = keys if keys else [api_key]
        self._current_key_index: int = 0

        super().__init__(
            api_key=self._api_keys[0],
            model=model,
            base_url="https://api.groq.com/openai/v1",
            timeout=timeout,
            retry_attempts=retry_attempts,
            rate_limit_pacing=rate_limit_pacing,
        )
        if len(self._api_keys) > 1:
            logger.info(
                "GroqClient khởi tạo với %d API key. Tự động xoay vòng khi bị rate limit.",
                len(self._api_keys),
            )

    def _rotate_key(self) -> bool:
        """Chuyển sang key tiếp theo. Trả về True nếu đã xoay sang key mới."""
        if len(self._api_keys) <= 1:
            return False
        next_index = (self._current_key_index + 1) % len(self._api_keys)
        if next_index == self._current_key_index:
            return False
        self._current_key_index = next_index
        new_key = self._api_keys[self._current_key_index]
        logger.warning(
            "Groq rate limit — chuyển sang key #%d/%d.",
            self._current_key_index + 1,
            len(self._api_keys),
        )
        # Tạo lại OpenAI client với key mới
        self._init_client(new_key, self.base_url)
        return True

    def _complete_with_retries(self, request: dict[str, Any]) -> str:
        """Retry với xoay vòng key khi gặp rate limit 429."""
        last_error: Exception | None = None
        # Tổng số lần thử = retry_attempts × số key (mỗi key thử retry_attempts lần)
        total_attempts = self.retry_attempts * len(self._api_keys)

        for attempt in range(total_attempts):
            try:
                self._pace_before_request(request)
                resp, headers = self._create_completion(request)
                self._update_rate_limit_state(headers)
                from meetasr.llm.openai_client import (
                    LLMCompletionTruncatedError,
                    _message_text,
                )
                choice = resp.choices[0]
                content = _message_text(choice.message.content)
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    raise LLMCompletionTruncatedError(
                        "LLM completion reached max_tokens before finishing"
                    )
                if not content.strip():
                    logger.warning(
                        "LLM trả về nội dung rỗng (model=%s, finish_reason=%s).",
                        self.model,
                        finish_reason,
                    )
                return content

            except Exception as e:
                last_error = e
                response = getattr(e, "response", None)
                self._update_rate_limit_state(getattr(response, "headers", None))

                status = getattr(e, "status_code", None) or getattr(
                    response, "status_code", None
                )
                is_rate_limit = str(status) == "429"

                # Nếu bị rate limit VÀ còn key khác → xoay key ngay, không chờ
                if is_rate_limit and self._rotate_key():
                    logger.info("Thử lại ngay với key mới (attempt %d).", attempt + 1)
                    continue

                # Lỗi khác hoặc hết key → retry có chờ như bình thường
                wait = _retry_wait_seconds(e, attempt)
                if attempt + 1 < total_attempts and _is_retryable_error(e):
                    logger.warning(
                        "LLM call thất bại (attempt %d/%d): %s. Thử lại sau %ss...",
                        attempt + 1,
                        total_attempts,
                        e,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    break

        raise RuntimeError(
            f"LLM call thất bại sau {total_attempts} lần thử: {last_error}"
        )
