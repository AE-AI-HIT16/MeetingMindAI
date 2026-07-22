"""DocumentPlanner — content-agnostic summarization via Structured Extraction.

Replaces MeetingSummarizer for Phase 2. MeetingSummarizer is kept untouched
for Phase 1 API compatibility.

Architecture (optimized from doc 14 original Plan→Write):
- Transcript that fits the configured context budget: 1 LLM call
- Longer transcript: N structured extraction + 1 aggregate + K reduce
  Total: N + 1 + K calls (vs N + K*N + K in original design)

Chunking strategy: overlap 5 lines between chunks to preserve pronoun
reference context (see docs/chunking_analysis.md §4). Monster lines
(> max_chars) are sub-split via planner_chunk module.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.llm.planner_chunk import OVERLAP_LINES, chunk_with_overlap
from meetasr.llm.planner_validation import (
    join_text_parts,
    normalize_extraction,
    normalize_outline,
    normalize_written_sections,
    parse_json_object as _parse_json_object,
)
from meetasr.schemas import TranscriptResult
from meetasr.schemas_doc import DocSection, DocumentReport

logger = logging.getLogger(__name__)

MAX_CHARS_PER_CHUNK = 6000
SHORT_TRANSCRIPT_CHARS = 6400
MAX_AGGREGATE_DATA_CHARS = 6200
MAX_REDUCE_DATA_CHARS = 7000


@dataclass
class PlannerRunMetrics:
    """Metrics collected by one :meth:`DocumentPlanner.plan_and_write` run.

    Counts are owned by the planner rather than inferred from optional client
    attributes. A call is counted when attempted, including failed LLM calls.
    """

    path: str = "empty"
    input_chars: int = 0
    chunk_count: int = 0
    section_count: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    calls_by_stage: dict[str, int] = field(default_factory=dict)
    failures_by_stage: dict[str, int] = field(default_factory=dict)
    llm_seconds_by_stage: dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a logging/serialization-safe snapshot."""
        return {
            "path": self.path,
            "input_chars": self.input_chars,
            "chunk_count": self.chunk_count,
            "section_count": self.section_count,
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
            "calls_by_stage": dict(self.calls_by_stage),
            "failures_by_stage": dict(self.failures_by_stage),
            "llm_seconds_by_stage": {
                stage: round(seconds, 4)
                for stage, seconds in self.llm_seconds_by_stage.items()
            },
            "total_seconds": round(self.total_seconds, 4),
        }


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
        language: str = "vi",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self.client = client
        self.language = language.strip() or "vi"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_run_metrics: PlannerRunMetrics | None = None
        from meetasr.llm.llm_utils.prompts import load_generic_prompts
        self._prompts = load_generic_prompts(language=self.language)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_and_write(self, transcript: TranscriptResult) -> DocumentReport:
        """Full pipeline: structured extract → aggregate plan → reduce."""
        t0 = time.perf_counter()
        full_text = self._format_transcript(transcript)
        metrics = PlannerRunMetrics(input_chars=len(full_text))

        if not full_text.strip():
            logger.warning("DocumentPlanner received an empty transcript.")
            report = DocumentReport(content_kind="Tài liệu")
        elif len(full_text) < SHORT_TRANSCRIPT_CHARS:
            metrics.path = "short"
            report = self._short_path(full_text, metrics)
        else:
            metrics.path = "long"
            report = self._long_path(full_text, metrics)

        report.language = self.language
        report.llm_model = getattr(self.client, "model", "")
        metrics.section_count = len(report.sections)
        metrics.total_seconds = time.perf_counter() - t0
        report.processing_time = round(metrics.total_seconds, 2)
        self.last_run_metrics = metrics
        logger.info("DocumentPlanner metrics: %s", metrics.to_dict())
        return report

    # ------------------------------------------------------------------
    # Short path — one call within the 8,000-character input budget
    # ------------------------------------------------------------------

    def _short_path(
        self,
        full_text: str,
        metrics: PlannerRunMetrics | None = None,
    ) -> DocumentReport:
        """Single LLM call for short transcripts using single_pass prompt.

        Uses single_pass_vi.txt which returns complete JSON with
        content_kind + sections (each with markdown already written).
        This is a TRUE single call — no separate plan then reduce.
        """
        prompt = self._prompts["single_pass"].format(transcript=full_text)
        try:
            raw = self._call_llm(
                "single_pass",
                prompt,
                max_tokens=self.max_tokens,
                metrics=metrics,
            )
            data = _parse_json_object(raw)
            content_kind = _safe_text(data.get("content_kind"), "Tài liệu")
            sections = normalize_written_sections(data.get("sections"))
        except Exception as e:
            logger.warning("Short-path single_pass failed (%s), fallback.", e)
            content_kind = "Tài liệu"
            sections = []
        return DocumentReport(content_kind=content_kind, sections=sections)

    # ------------------------------------------------------------------
    # Long path — structured extraction + aggregate + reduce
    # ------------------------------------------------------------------

    def _long_path(
        self,
        full_text: str,
        metrics: PlannerRunMetrics | None = None,
    ) -> DocumentReport:
        """Structured extraction pipeline for long transcripts."""
        chunks = chunk_with_overlap(
            full_text,
            max_chars=MAX_CHARS_PER_CHUNK,
            overlap_lines=OVERLAP_LINES,
        )
        if metrics is not None:
            metrics.chunk_count = len(chunks)
        logger.info(
            "Long path: %d chunks (overlap=%d lines), running extraction.",
            len(chunks), OVERLAP_LINES,
        )

        # Step 1: Structured extraction (sequential — parallel is future optimization)
        extractions = self._run_extractions(chunks, metrics)

        # Step 2: Aggregate plan
        content_kind, outline = self._aggregate_plan(extractions, metrics)

        # Step 3: Reduce each section from extraction results
        sections = []
        for sec in outline:
            raw_data = self._gather_for_section(sec, extractions)
            if not raw_data.strip():
                continue
            md = self._call_reduce(sec, content_kind, raw_data, metrics)
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

    def _run_extractions(
        self,
        chunks: list[str],
        metrics: PlannerRunMetrics | None = None,
    ) -> list[dict]:
        """Run structured extraction on all chunks (sequential with logging)."""
        results = []
        for i, chunk in enumerate(chunks):
            logger.info("Extracting chunk %d/%d", i + 1, len(chunks))
            prompt = self._prompts["structured_extract"].format(chunk=chunk)
            try:
                raw = self._call_llm(
                    "extraction",
                    prompt,
                    max_tokens=2048,
                    metrics=metrics,
                )
                data = normalize_extraction(_parse_json_object(raw), i)
                results.append(data)
            except Exception as e:
                logger.warning("Extraction failed for chunk %d: %s", i, e)
                results.append({"chunk_index": i, "summary": ""})
        return results

    def _aggregate_plan(
        self,
        extractions: list[dict],
        metrics: PlannerRunMetrics | None = None,
    ) -> tuple[str, list[dict]]:
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
                elif isinstance(val, (int, float, bool)):
                    extras.append(f"  {key}: present")
            line = f"[{idx}] {s}"
            if extras:
                line += "\n" + "\n".join(extras)
            summaries.append(line)

        combined = join_text_parts(summaries, MAX_AGGREGATE_DATA_CHARS)
        prompt = self._prompts["aggregate_plan"].format(
            structured_summaries=combined
        )
        try:
            raw = self._call_llm(
                "aggregate",
                prompt,
                max_tokens=1024,
                metrics=metrics,
            )
            data = _parse_json_object(raw)
            content_kind = _safe_text(data.get("content_kind"), "Tài liệu")
            outline = normalize_outline(data.get("outline"))
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
        kind = _safe_text(section.get("kind"), "summary")
        source_keys = section.get("source_keys", [kind])
        if isinstance(source_keys, str):
            source_keys = [source_keys]
        elif not isinstance(source_keys, list):
            source_keys = [kind]
        parts: list[str] = []
        seen: set[tuple[str, str]] = set()

        for ext in extractions:
            for key in source_keys:
                if not isinstance(key, str):
                    continue
                val = ext.get(key)
                if not val:
                    continue
                if isinstance(val, (list, dict)):
                    rendered = json.dumps(val, ensure_ascii=False)
                else:
                    rendered = str(val)
                marker = (key, rendered)
                if marker not in seen:
                    parts.append(f"{key}: {rendered}")
                    seen.add(marker)

        # Only the summary section may fall back to chunk summaries. Other
        # sections without evidence must be omitted instead of fabricated.
        if not parts and kind == "summary" and "summary" in source_keys:
            for ext in extractions:
                s = ext.get("summary", "")
                if s:
                    parts.append(s)

        return join_text_parts(parts, MAX_REDUCE_DATA_CHARS)

    def _call_reduce(
        self,
        section: dict,
        content_kind: str,
        raw_data: str,
        metrics: PlannerRunMetrics | None = None,
    ) -> str:
        """Reduce gathered data into final markdown for one section."""
        prompt = self._prompts["reduce_section"].format(
            heading=section.get("heading", ""),
            kind=section.get("kind", "summary"),
            content_kind=content_kind,
            raw_data=raw_data,
        )
        try:
            return self._call_llm(
                "reduce",
                prompt,
                max_tokens=self.max_tokens,
                metrics=metrics,
            ).strip()
        except Exception as e:
            logger.warning("Reduce failed for '%s': %s", section.get("heading"), e)
            return ""

    def _call_llm(
        self,
        stage: str,
        prompt: str,
        *,
        max_tokens: int,
        metrics: PlannerRunMetrics | None,
    ) -> str:
        """Call the LLM and record planner-owned attempt/failure/timing metrics."""
        if metrics is not None:
            metrics.llm_calls += 1
            metrics.calls_by_stage[stage] = metrics.calls_by_stage.get(stage, 0) + 1

        started = time.perf_counter()
        try:
            return self.client.chat(
                prompt,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            if metrics is not None:
                metrics.llm_failures += 1
                metrics.failures_by_stage[stage] = (
                    metrics.failures_by_stage.get(stage, 0) + 1
                )
            raise
        finally:
            if metrics is not None:
                elapsed = time.perf_counter() - started
                metrics.llm_seconds_by_stage[stage] = (
                    metrics.llm_seconds_by_stage.get(stage, 0.0) + elapsed
                )

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


def _safe_text(value: object, default: str) -> str:
    """Return a non-empty stripped string or a default value."""
    return value.strip() if isinstance(value, str) and value.strip() else default
