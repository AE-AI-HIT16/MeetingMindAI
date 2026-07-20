"""DocumentPlanner — content-agnostic summarization via Structured Extraction.

Replaces MeetingSummarizer for Phase 2. MeetingSummarizer is kept untouched
for Phase 1 API compatibility.

Architecture (optimized from doc 14 original Plan→Write):
- < 30 min transcript: 1 LLM call (full text, plan + write combined)
- >= 30 min: N structured extraction + 1 aggregate + K reduce
  Total: N + 1 + K calls (vs N + K*N + K in original design)

Chunking strategy: overlap 5 lines between chunks to preserve pronoun
reference context (see docs/chunking_analysis.md §4). Monster lines
(> max_chars) are sub-split via planner_chunk module.
"""

from __future__ import annotations

import json
import logging
import re
import time

from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.llm.planner_chunk import OVERLAP_LINES, chunk_with_overlap
from meetasr.schemas import TranscriptResult
from meetasr.schemas_doc import DocSection, DocumentReport

logger = logging.getLogger(__name__)

MAX_CHARS_PER_CHUNK = 6000
SHORT_TRANSCRIPT_CHARS = 18000  # ~30 min of transcript
MAX_CONCURRENT_LLM = 5
DOCUMENT_LANGUAGE = "vi"


class DocumentPlanner:
    """Content-agnostic document planner using structured extraction.

    Args:
        client: Any AbsLLMClient implementation.
        temperature: LLM sampling temperature.
        max_tokens: Max tokens per LLM call.
    """

    def __init__(
        self,
        client: AbsLLMClient,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self.client = client
        self.language = DOCUMENT_LANGUAGE
        self.temperature = temperature
        self.max_tokens = max_tokens
        from meetasr.llm.llm_utils.prompts import load_generic_prompts
        self._prompts = load_generic_prompts(language=DOCUMENT_LANGUAGE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_and_write(self, transcript: TranscriptResult) -> DocumentReport:
        """Full pipeline: structured extract → aggregate plan → reduce."""
        t0 = time.perf_counter()
        full_text = self._format_transcript(transcript)

        if len(full_text) < SHORT_TRANSCRIPT_CHARS:
            report = self._short_path(full_text)
        else:
            report = self._long_path(full_text)

        report.language = self.language
        report.llm_model = getattr(self.client, "model", "")
        report.processing_time = round(time.perf_counter() - t0, 2)
        return report

    # ------------------------------------------------------------------
    # Short path (< 30 min) — 1 LLM call via single_pass_vi.txt
    # ------------------------------------------------------------------

    def _short_path(self, full_text: str) -> DocumentReport:
        """Single LLM call for short transcripts using single_pass prompt.

        Uses single_pass_vi.txt which returns complete JSON with
        content_kind + sections (each with markdown already written).
        This is a TRUE single call — no separate plan then reduce.
        """
        prompt = self._prompts["single_pass"].format(transcript=full_text)
        try:
            raw = self.client.chat(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )
            data = _parse_json_object(raw)
            content_kind = data.get("content_kind", "Tài liệu")
            raw_sections = data.get("sections", [])
        except Exception as e:
            logger.warning("Short-path single_pass failed (%s), fallback.", e)
            content_kind = "Tài liệu"
            raw_sections = []

        sections = []
        for sec in raw_sections:
            md = sec.get("markdown", "")
            found = bool(md.strip())
            sections.append(DocSection(
                id=sec.get("id", "s"),
                heading=sec.get("heading", ""),
                kind=sec.get("kind", "summary"),
                markdown=md,
                found=found,
            ))

        sections = [s for s in sections if s.found]
        return DocumentReport(content_kind=content_kind, sections=sections)

    # ------------------------------------------------------------------
    # Long path (>= 30 min) — structured extraction + aggregate + reduce
    # ------------------------------------------------------------------

    def _long_path(self, full_text: str) -> DocumentReport:
        """Structured extraction pipeline for long transcripts."""
        chunks = chunk_with_overlap(
            full_text,
            max_chars=MAX_CHARS_PER_CHUNK,
            overlap_lines=OVERLAP_LINES,
        )
        logger.info(
            "Long path: %d chunks (overlap=%d lines), running extraction.",
            len(chunks), OVERLAP_LINES,
        )

        # Step 1: Structured extraction (sequential — parallel is future optimization)
        extractions = self._run_extractions(chunks)

        # Step 2: Aggregate plan
        content_kind, outline = self._aggregate_plan(extractions)

        # Step 3: Reduce each section from extraction results
        sections = []
        for sec in outline:
            raw_data = self._gather_for_section(sec, extractions)
            if not raw_data.strip():
                continue
            md = self._call_reduce(sec, content_kind, raw_data)
            found = bool(md.strip())
            sections.append(DocSection(
                id=sec.get("id", "s"),
                heading=sec.get("heading", ""),
                kind=sec.get("kind", "summary"),
                markdown=md,
                found=found,
            ))

        sections = [s for s in sections if s.found]
        return DocumentReport(content_kind=content_kind, sections=sections)

    def _run_extractions(self, chunks: list[str]) -> list[dict]:
        """Run structured extraction on all chunks (sequential with logging)."""
        results = []
        for i, chunk in enumerate(chunks):
            logger.info("Extracting chunk %d/%d", i + 1, len(chunks))
            prompt = self._prompts["structured_extract"].format(chunk=chunk)
            try:
                raw = self.client.chat(
                    prompt, temperature=self.temperature, max_tokens=2048
                )
                data = _parse_json_object(raw)
                data["chunk_index"] = i
                results.append(data)
            except Exception as e:
                logger.warning("Extraction failed for chunk %d: %s", i, e)
                results.append({"chunk_index": i, "summary": ""})
        return results

    def _aggregate_plan(self, extractions: list[dict]) -> tuple[str, list[dict]]:
        """Aggregate structured results into content_kind + outline.

        Dynamically scans ALL keys from extraction dicts so the
        aggregate LLM sees every label (including new/unexpected ones).
        The LLM then declares source_keys per outline section to tell
        _gather_for_section exactly which keys to collect.
        """
        summaries = []
        skip_keys = {"chunk_index", "summary"}
        for ext in extractions:
            idx = ext.get("chunk_index", "?")
            s = ext.get("summary", "")
            # Dynamic: report ALL non-empty fields so LLM sees every label
            extras = []
            for key, val in ext.items():
                if key in skip_keys:
                    continue
                if isinstance(val, list) and val:
                    extras.append(f"  {key}: {len(val)} items")
                elif isinstance(val, dict) and val:
                    extras.append(f"  {key}: present")
                elif isinstance(val, str) and val.strip():
                    extras.append(f"  {key}: present")
            line = f"[{idx}] {s}"
            if extras:
                line += "\n" + "\n".join(extras)
            summaries.append(line)

        combined = "\n".join(summaries)
        prompt = self._prompts["aggregate_plan"].format(
            structured_summaries=combined
        )
        try:
            raw = self.client.chat(
                prompt, temperature=self.temperature, max_tokens=1024
            )
            data = _parse_json_object(raw)
            content_kind = data.get("content_kind", "Tài liệu")
            outline = data.get("outline", [])
            if not outline:
                raise ValueError("empty outline")
            return content_kind, outline
        except Exception as e:
            logger.warning("Aggregate plan failed (%s), fallback.", e)
            return "Tài liệu", [
                {"id": "s1", "heading": "Tóm tắt", "kind": "summary",
                 "source_keys": ["summary"]},
            ]

    def _gather_for_section(self, section: dict, extractions: list[dict]) -> str:
        """Gather data using source_keys declared by aggregate LLM.

        The aggregate step returns source_keys per outline section,
        resolving label aliases (e.g. 'doanh_thu', 'revenue' → same
        section). This method mechanically collects matching keys.
        """
        source_keys = section.get(
            "source_keys", [section.get("kind", "summary")]
        )
        parts: list[str] = []

        for ext in extractions:
            for key in source_keys:
                val = ext.get(key)
                if not val:
                    continue
                if isinstance(val, list):
                    parts.append(json.dumps(val, ensure_ascii=False))
                else:
                    parts.append(str(val))

        # Fallback: if nothing gathered, collect summaries as context
        if not parts:
            for ext in extractions:
                s = ext.get("summary", "")
                if s:
                    parts.append(s)

        return "\n\n".join(parts)

    def _call_reduce(self, section: dict, content_kind: str, raw_data: str) -> str:
        """Reduce gathered data into final markdown for one section."""
        prompt = self._prompts["reduce_section"].format(
            heading=section.get("heading", ""),
            kind=section.get("kind", "summary"),
            content_kind=content_kind,
            raw_data=raw_data,
        )
        try:
            return self.client.chat(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            ).strip()
        except Exception as e:
            logger.warning("Reduce failed for '%s': %s", section.get("heading"), e)
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_transcript(self, result: TranscriptResult) -> str:
        """Convert TranscriptResult to text for LLM."""
        if result.sentence_info:
            lines = []
            for s in result.sentence_info:
                spk = f"Speaker {s.speaker}: " if s.speaker is not None else ""
                lines.append(f"[{s.start:.1f}s] {spk}{s.text}")
            return "\n".join(lines)
        return result.text

    # NOTE: _chunk() removed — chunking now handled by
    # meetasr.llm.planner_chunk.chunk_with_overlap()
    # See docs/chunking_analysis.md for rationale.


def _parse_json_object(raw: str) -> dict:
    """Extract JSON object from LLM response.

    Handles both raw JSON and markdown-wrapped code blocks.

    Args:
        raw: Raw LLM response text.

    Returns:
        Parsed JSON as dict.

    Raises:
        json.JSONDecodeError: If no valid JSON found.
    """
    match = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    text = match.group(1).strip() if match else raw.strip()
    if not match:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    return json.loads(text, strict=False)
