"""Meeting Summarizer — orchestrates LLM calls to produce MeetingReport."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
import re
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
        from meetasr.llm.llm_utils.prompts import load_prompts
        self._prompts = load_prompts(language=self.language)

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

        return result.text

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _get_summary(self, text: str) -> str:
        """Generate summary paragraph."""
        return self._call_summary_direct(text)

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
                
                # Extract markdown code fence if present
                match = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
                if match:
                    raw = match.group(1).strip()
                else:
                    # Otherwise, try to extract array brackets just in case
                    start = raw.find('[')
                    end = raw.rfind(']')
                    if start != -1 and end != -1:
                        raw = raw[start:end+1]
                        
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parse failed (attempt {attempt + 1}): {e}\nRaw snippet: {raw[:100]}...")
            except Exception as e:
                logging.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
        logging.warning("Returning empty fallback for this LLM call.")
        return fallback

    def _truncate(self, text: str, max_chars: int = MAX_CHARS_DIRECT) -> str:
        """Truncate text to max_chars for topic/action/decision extraction."""
        return text


def _optional_float(val) -> Optional[float]:
    """Convert value to float or None."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
