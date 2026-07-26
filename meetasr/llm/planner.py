"""Phase 2 content-agnostic DocumentPlanner using Plan → Write."""

from __future__ import annotations

import logging
import time
from typing import Any

from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.llm.planner_chunk import (
    chunk_on_lines,
    format_time_range,
    representative_sample,
)
from meetasr.llm.planner_validation import (
    DRAFT_SEPARATOR,
    bounded_join,
    build_plan_retry_prompt,
    fallback_outline,
    normalize_outline,
    normalize_report_outline,
    normalize_title,
    parse_json_object as _parse_json_object,
    PLAN_RESPONSE_FORMAT,
    safe_text,
    validate_plan_payload,
)
from meetasr.schemas import TranscriptResult
from meetasr.schemas_doc import DocSection, DocumentReport

logger = logging.getLogger(__name__)

MAX_LLM_INPUT_CHARS = 8000
MAX_CHARS_PER_CHUNK = 6000
SAMPLE_CHARS_FOR_PLAN = 4000
MAX_LABEL_CHARS = 200
MAX_MAP_TOKENS = 1200
MAX_PLAN_TOKENS = 2048
VI_SYSTEM_PROMPT = (
    "Mọi nội dung bạn tạo phải bằng tiếng Việt, có căn cứ từ nguồn và "
    "không được biến lỗi transcript thành dữ kiện."
)


class DocumentPlanner:
    """Build a generic structured document from a transcript."""

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
        self._system_prompt = VI_SYSTEM_PROMPT if self.language == "vi" else None

        from meetasr.llm.llm_utils.prompts import load_generic_prompts

        self._prompts = load_generic_prompts(language=self.language)

    def plan_and_write(self, transcript: TranscriptResult) -> DocumentReport:
        """Plan an outline, then map-reduce every proposed section.

        Empty transcripts return an empty document without calling the LLM.
        Invalid LLM responses degrade to the documented summary-only fallback.
        """
        started = time.perf_counter()
        full_text = self._format_transcript(transcript)

        if not full_text.strip():
            logger.warning("DocumentPlanner received an empty transcript.")
            report = DocumentReport(content_kind="Tài liệu")
        else:
            content_kind, outline = self._plan(full_text)
            budget = self._write_data_budget(outline, content_kind)
            chunks = self._chunk(full_text, budget)

            section_drafts: dict[str, list[str]] = {section["id"]: [] for section in outline}
            for chunk in chunks:
                if not chunk.strip():
                    continue
                extracted = self._call_multi_write(outline, content_kind, chunk)
                for section in outline:
                    sec_id = section["id"]
                    draft = extracted.get(sec_id)
                    if draft and isinstance(draft, str) and draft.strip():
                        section_drafts[sec_id].append(draft.strip())

            sections = []
            for section in outline:
                drafts = section_drafts[section["id"]]
                if not drafts:
                    continue
                if len(drafts) == 1:
                    markdown = drafts[0]
                else:
                    markdown = self._reduce(section, content_kind, drafts)
                
                if markdown.strip():
                    sections.append(
                        DocSection(
                            id=section["id"],
                            heading=section["heading"],
                            kind=section["kind"],
                            markdown=markdown,
                        )
                    )
            
            report = DocumentReport(
                content_kind=content_kind,
                sections=sections,
            )

        report.language = self.language
        report.llm_model = safe_text(getattr(self.client, "model", ""), "")
        report.processing_time = round(time.perf_counter() - started, 2)
        return report

    def plan_only(self, transcript_sample: str) -> tuple[str, list[dict[str, str]]]:
        """Run only the plan step for the realtime document stream."""
        safe_sample = (
            transcript_sample if isinstance(transcript_sample, str) else ""
        )
        return self._plan(safe_sample)

    def write_one_section(
        self,
        section: dict[str, Any],
        content_kind: str,
        text: str,
    ) -> DocSection:
        """Write one planned section from new realtime transcript text."""
        normalized = normalize_outline([section])
        safe_section = normalized[0] if normalized else fallback_outline()[0]
        safe_content_kind = normalize_title(content_kind)
        safe_text_chunk = text if isinstance(text, str) else ""
        chunks = self._chunk(
            safe_text_chunk,
            self._write_data_budget([safe_section], safe_content_kind),
        )

        section_drafts = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            extracted = self._call_multi_write([safe_section], safe_content_kind, chunk)
            draft = extracted.get(safe_section["id"])
            if draft and isinstance(draft, str) and draft.strip():
                section_drafts.append(draft.strip())
        
        if not section_drafts:
            markdown = ""
        elif len(section_drafts) == 1:
            markdown = section_drafts[0]
        else:
            markdown = self._reduce(safe_section, safe_content_kind, section_drafts)

        return DocSection(
            id=safe_section["id"],
            heading=safe_section["heading"],
            kind=safe_section["kind"],
            markdown=markdown,
        )

    def _plan(self, full_text: str) -> tuple[str, list[dict[str, str]]]:
        """Ask the LLM to identify content kind and propose an outline."""
        if not full_text.strip():
            return "Tài liệu", fallback_outline()
        sample = self._representative_sample(
            full_text, min(SAMPLE_CHARS_FOR_PLAN, self._plan_data_budget())
        )
        prompt = self._prompts["plan"].format(transcript_sample=sample)
        try:
            raw = self.client.chat(
                prompt,
                system=self._system_prompt,
                temperature=0.0,
                max_tokens=MAX_PLAN_TOKENS,
                response_format=PLAN_RESPONSE_FORMAT,
            )
            try:
                data = validate_plan_payload(_parse_json_object(raw))
            except (TypeError, ValueError) as parse_error:
                logger.warning(
                    "Plan response is empty or invalid (%s); retrying in JSON mode.",
                    parse_error,
                )
                retried = self.client.chat(
                    build_plan_retry_prompt(raw, parse_error, sample),
                    system=self._system_prompt,
                    temperature=0.0,
                    max_tokens=MAX_PLAN_TOKENS,
                    response_format=PLAN_RESPONSE_FORMAT,
                )
                data = validate_plan_payload(_parse_json_object(retried))
            content_kind = normalize_title(data.get("content_kind"))
            outline = normalize_report_outline(data.get("outline"))
            return content_kind, outline
        except Exception as exc:
            logger.warning(
                "Plan step failed (%s); using the required fallback outline.",
                exc,
            )
            return "Tài liệu", fallback_outline()

    def _call_multi_write(
        self,
        outline: list[dict[str, str]],
        content_kind: str,
        chunk: str,
    ) -> dict[str, str]:
        """Extract data for multiple sections from a single chunk."""
        sections_list = self._format_sections_list(outline)
        prompt = self._prompts["multi_write"].format(
            content_kind=content_kind,
            sections_list=sections_list,
            chunk=chunk,
        )
        try:
            # We explicitly ask for JSON in the prompt, so we can parse it even 
            # if response_format=json_object is not strictly supported by the LLM.
            response = self.client.chat(
                prompt,
                system=self._system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return _parse_json_object(response)
        except Exception as exc:
            logger.warning(
                "Multi-write step failed (chunk length %d): %s",
                len(chunk),
                exc,
            )
            return {}

    def _reduce(
        self,
        section: dict[str, str],
        content_kind: str,
        drafts: list[str],
    ) -> str:
        """Merge chunk drafts into one coherent section."""
        joined = DRAFT_SEPARATOR.join(drafts)
        merge_note = (
            "Dưới đây là các bản nháp rời rạc cho cùng một mục, hãy gộp "
            "và viết lại thành một bản hoàn chỉnh, mạch lạc, không lặp ý:"
        )
        data_budget = self._reduce_data_budget(section, content_kind)
        reduce_data = (
            f"{merge_note}\n\n"
            f"{bounded_join(drafts, data_budget - len(merge_note) - 2)}"
        )
        prompt = self._prompts["reduce_section"].format(
            content_kind=content_kind,
            heading=section["heading"],
            kind=section["kind"],
            chunk=reduce_data,
        )
        try:
            response = self.client.chat(
                prompt,
                system=self._system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.strip() if isinstance(response, str) else joined
        except Exception as exc:
            logger.warning(
                "Reduce step failed for section '%s': %s",
                section["heading"],
                exc,
            )
            return joined

    def _representative_sample(self, text: str, max_chars: int) -> str:
        return representative_sample(text, max_chars)

    def _format_transcript(self, result: TranscriptResult) -> str:
        """Format sentence timestamps and speakers for grounding."""
        if result.sentence_info:
            lines = []
            for sentence in result.sentence_info:
                speaker = "" if sentence.speaker is None else (
                    f"Speaker {sentence.speaker}: "
                )
                timestamp = format_time_range(sentence.start, sentence.end)
                lines.append(f"{timestamp} {speaker}{sentence.text}")
            return "\n".join(lines)
        return result.text if isinstance(result.text, str) else ""

    def _chunk(self, text: str, max_chars: int) -> list[str]:
        return chunk_on_lines(text, max_chars)

    def _plan_data_budget(self) -> int:
        empty_prompt = self._prompts["plan"].format(transcript_sample="")
        return max(1, MAX_LLM_INPUT_CHARS - len(empty_prompt))

    def _write_data_budget(
        self,
        outline: list[dict[str, Any]] | dict[str, Any],
        content_kind: str,
    ) -> int:
        safe_outline = outline if isinstance(outline, list) else [outline]
        empty_prompt = self._prompts["multi_write"].format(
            content_kind=safe_text(content_kind, "Tài liệu")[:MAX_LABEL_CHARS],
            sections_list=self._format_sections_list(safe_outline),
            chunk="",
        )
        return max(
            1,
            min(MAX_CHARS_PER_CHUNK, MAX_LLM_INPUT_CHARS - len(empty_prompt)),
        )

    def _reduce_data_budget(
        self,
        section: dict[str, Any],
        content_kind: str,
    ) -> int:
        empty_prompt = self._prompts["reduce_section"].format(
            content_kind=safe_text(content_kind, "Tài liệu")[:MAX_LABEL_CHARS],
            heading=safe_text(section.get("heading"), "Tóm tắt")[:MAX_LABEL_CHARS],
            kind=safe_text(section.get("kind"), "summary")[:MAX_LABEL_CHARS],
            chunk="",
        )
        return max(1, MAX_LLM_INPUT_CHARS - len(empty_prompt))

    @staticmethod
    def _format_sections_list(outline: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"- {safe_text(section.get('id'), '')}: "
            f"{safe_text(section.get('heading'), 'Tóm tắt')} "
            f"(kind: {safe_text(section.get('kind'), 'summary')})"
            for section in outline
        )
