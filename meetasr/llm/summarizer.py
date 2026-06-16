"""Meeting Summarizer — orchestrates LLM calls to produce MeetingReport."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.schemas import (
    TranscriptResult,
    MeetingReport,
    Topic,
    ActionItem,
    Decision,
)

# Max characters in a single LLM call. Transcripts longer than this
# are split and processed via map-reduce.
MAX_CHARS_DIRECT = 8000

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


class MeetingSummarizer:
    """Orchestrates LLM calls to analyse a meeting transcript.

    Produces: summary paragraph, topics, action items, decisions.

    Args:
        client: Any AbsLLMClient implementation (OpenAI, Ollama, etc.)
        language: Output language code. "vi" (default) or "en".
        temperature: LLM sampling temperature.
        max_tokens: Max tokens per LLM call.
    """

    def __init__(
        self,
        client: AbsLLMClient,
        language: str = "vi",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.client = client
        self.language = language
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._prompts = self._load_prompts()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(
        self,
        transcript: TranscriptResult,
        asr_model: str = "",
        llm_model: str = "",
    ) -> MeetingReport:
        """Produce a full MeetingReport from a TranscriptResult.

        Args:
            transcript: ASR output with sentence_info.
            asr_model: ASR model name (for metadata).
            llm_model: LLM model name (for metadata).

        Returns:
            MeetingReport with summary, topics, action_items, decisions.
        """
        t0 = time.perf_counter()
        text = self._format_transcript(transcript)

        summary = self._get_summary(text)
        topics = self._get_topics(text)
        actions = self._get_action_items(text)
        decisions = self._get_decisions(text)

        return MeetingReport(
            transcript=transcript,
            summary=summary,
            topics=topics,
            action_items=actions,
            decisions=decisions,
            language=self.language,
            asr_model=asr_model,
            llm_model=llm_model,
            processing_time=round(time.perf_counter() - t0, 2),
        )

    # ------------------------------------------------------------------
    # Transcript formatting
    # ------------------------------------------------------------------

    def _format_transcript(self, result: TranscriptResult) -> str:
        """Convert TranscriptResult to human-readable text for LLM."""
        if result.sentence_info:
            lines = []
            for s in result.sentence_info:
                ts = f"[{s.start:.1f}s]"
                spk = f"Speaker {s.speaker}: " if s.speaker is not None else ""
                lines.append(f"{ts} {spk}{s.text}")
            return "\n".join(lines)
        # Fallback: just the full text
        return result.text

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _get_summary(self, text: str) -> str:
        """Generate summary paragraph."""
        if len(text) <= MAX_CHARS_DIRECT:
            return self._call_summary_direct(text)
        return self._call_summary_mapreduce(text)

    def _call_summary_direct(self, text: str) -> str:
        prompt = self._prompts["summarize"].format(transcript=text)
        try:
            return self.client.chat(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ).strip()
        except Exception as e:
            logging.warning(f"Summary LLM call failed: {e}")
            return "[Không thể tạo tóm tắt]" if self.language == "vi" else "[Summary unavailable]"

    def _call_summary_mapreduce(self, text: str) -> str:
        """Summarize long transcripts via chunking → chunk summaries → final summary."""
        chunks = self._split_text(text, MAX_CHARS_DIRECT)
        logging.info(f"Transcript too long ({len(text)} chars), splitting into {len(chunks)} chunks")
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logging.info(f"Summarizing chunk {i + 1}/{len(chunks)}...")
            s = self._call_summary_direct(chunk)
            chunk_summaries.append(s)

        # Merge chunk summaries into final summary
        merged = "\n\n".join(chunk_summaries)
        reduce_prompt = self._prompts["summarize"].format(transcript=merged)
        try:
            return self.client.chat(
                reduce_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ).strip()
        except Exception as e:
            logging.warning(f"Reduce summary failed: {e}")
            return "\n\n".join(chunk_summaries)   # return parts as fallback

    def _get_topics(self, text: str) -> list[Topic]:
        """Extract main topics as list of Topic objects."""
        prompt = self._prompts["topics"].format(transcript=self._truncate(text))
        raw = self._safe_json_call(prompt, fallback=[])
        return [
            Topic(
                title=item.get("title", ""),
                description=item.get("description", ""),
                start_time=float(item.get("start_time", 0.0)),
                end_time=float(item.get("end_time", 0.0)),
            )
            for item in raw
            if item.get("title")
        ]

    def _get_action_items(self, text: str) -> list[ActionItem]:
        """Extract action items as list of ActionItem objects."""
        prompt = self._prompts["action_items"].format(transcript=self._truncate(text))
        raw = self._safe_json_call(prompt, fallback=[])
        return [
            ActionItem(
                task=item.get("task", ""),
                assignee=item.get("assignee"),
                deadline=item.get("deadline"),
                priority=item.get("priority", "medium"),
                mentioned_by=item.get("mentioned_by"),
                timestamp=_optional_float(item.get("timestamp")),
            )
            for item in raw
            if item.get("task")
        ]

    def _get_decisions(self, text: str) -> list[Decision]:
        """Extract decisions as list of Decision objects."""
        prompt = self._prompts["decisions"].format(transcript=self._truncate(text))
        raw = self._safe_json_call(prompt, fallback=[])
        return [
            Decision(
                content=item.get("content", ""),
                made_by=item.get("made_by", "Team"),
                timestamp=_optional_float(item.get("timestamp")),
            )
            for item in raw
            if item.get("content")
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_json_call(self, prompt: str, fallback: list) -> list:
        """Call LLM and parse JSON response. Returns fallback on any error."""
        for attempt in range(2):
            try:
                raw = self.client.chat(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ).strip()
                # Strip markdown code fences if model wraps output
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parse failed (attempt {attempt + 1}): {e}")
            except Exception as e:
                logging.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
        logging.warning("Returning empty fallback for this LLM call.")
        return fallback

    def _load_prompts(self) -> dict[str, str]:
        """Load prompt templates from files."""
        lang = self.language
        # Fallback to English if language-specific prompt not found
        def _read(name: str) -> str:
            for suffix in [f"_{lang}.txt", "_en.txt"]:
                path = os.path.join(_PROMPT_DIR, name + suffix)
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        return f.read()
            raise FileNotFoundError(f"Prompt file not found for '{name}' in {_PROMPT_DIR}")

        return {
            "summarize": _read("summarize"),
            "topics": _read("topics"),
            "action_items": _read("action_items"),
            "decisions": _read("decisions"),
        }

    def _truncate(self, text: str, max_chars: int = MAX_CHARS_DIRECT) -> str:
        """Truncate text to max_chars for topic/action/decision extraction."""
        if len(text) <= max_chars:
            return text
        # Take first half + last quarter to capture intro and closing
        half = max_chars // 2
        return text[:half] + "\n...[truncated]...\n" + text[-(max_chars // 4):]

    @staticmethod
    def _split_text(text: str, chunk_size: int) -> list[str]:
        """Split text into chunks at line boundaries."""
        lines = text.split("\n")
        chunks, current, current_len = [], [], 0
        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > chunk_size and current:
                chunks.append("\n".join(current))
                current, current_len = [], 0
            current.append(line)
            current_len += line_len
        if current:
            chunks.append("\n".join(current))
        return chunks


def _optional_float(val) -> Optional[float]:
    """Convert value to float or None."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
