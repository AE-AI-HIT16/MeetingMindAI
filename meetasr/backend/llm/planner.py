"""Phase 2 content-agnostic DocumentPlanner using Plan → Write."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from meetasr.backend.llm.abs_llm import AbsLLMClient
from meetasr.backend.llm.planner_chunk import (
    chunk_on_lines,
    format_time_range,
    representative_sample,
)
from meetasr.backend.llm.planner_validation import (
    DRAFT_SEPARATOR,
    PLAN_RESPONSE_FORMAT,
    bounded_join,
    build_plan_retry_prompt,
    fallback_outline,
    normalize_outline,
    normalize_report_outline,
    normalize_title,
    safe_text,
    validate_plan_payload,
)
from meetasr.backend.llm.planner_validation import (
    parse_json_object as _parse_json_object,
)
from meetasr.backend.schemas import TranscriptResult
from meetasr.backend.schemas_doc import DocSection, DocumentReport

logger = logging.getLogger(__name__)

MAX_LLM_INPUT_CHARS = 8000
MAX_CHARS_PER_CHUNK = 6000
SAMPLE_CHARS_FOR_PLAN = 4000
MAX_LABEL_CHARS = 200
MAX_MAP_TOKENS = 1600
MAX_REDUCE_TOKENS = 1600
MAX_PLAN_TOKENS = 2048
MAX_MEDIUM_FALLBACK_TOKENS = 4096
GPT_OSS_MODEL_PREFIX = "openai/gpt-oss-"
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

        from meetasr.backend.llm.llm_utils.prompts import load_generic_prompts

        self._prompts = load_generic_prompts(language=self.language)

    def plan_and_write(
        self,
        transcript: TranscriptResult,
        progress_callback: Callable[[float], None] | None = None,
    ) -> DocumentReport:
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
            self._notify_progress(progress_callback, 0.05)
            content_kind, outline = self._plan(full_text)
            self._notify_progress(progress_callback, 0.15)
            budget = self._write_data_budget(outline, content_kind)
            chunks = self._chunk(full_text, budget)

            section_drafts: dict[str, list[str]] = {section["id"]: [] for section in outline}
            for chunk_index, chunk in enumerate(chunks, start=1):
                if not chunk.strip():
                    continue
                extracted = self._call_multi_write(outline, content_kind, chunk)
                for section in outline:
                    sec_id = section["id"]
                    draft = extracted.get(sec_id)
                    if draft and isinstance(draft, str) and draft.strip():
                        section_drafts[sec_id].append(draft.strip())
                self._notify_progress(
                    progress_callback,
                    0.15 + 0.55 * chunk_index / max(len(chunks), 1),
                )

            sections = []
            for section_index, section in enumerate(outline, start=1):
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
                self._notify_progress(
                    progress_callback,
                    0.7 + 0.25 * section_index / max(len(outline), 1),
                )

            report = DocumentReport(
                content_kind=content_kind,
                sections=sections,
            )

        report.language = self.language
        report.llm_model = safe_text(getattr(self.client, "model", ""), "")
        report.processing_time = round(time.perf_counter() - started, 2)
        return report

    @staticmethod
    def _notify_progress(
        callback: Callable[[float], None] | None,
        progress: float,
    ) -> None:
        if callback is None:
            return
        try:
            callback(max(0.0, min(progress, 0.95)))
        except Exception:
            logger.exception("Document progress callback failed.")

    def plan_only(self, transcript_sample: str) -> tuple[str, list[dict[str, str]]]:
        """Run only the plan step for the realtime document stream."""
        safe_sample = transcript_sample if isinstance(transcript_sample, str) else ""
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
        response: object = ""
        try:
            call_kwargs: dict[str, Any] = {
                "system": self._system_prompt,
                "temperature": self.temperature,
                "max_tokens": min(self.max_tokens, MAX_MAP_TOKENS),
                "response_format": PLAN_RESPONSE_FORMAT,
            }
            reasoning_effort = self._reasoning_effort()
            if reasoning_effort:
                call_kwargs["reasoning_effort"] = reasoning_effort
            response = self.client.chat(prompt, **call_kwargs)
            return self._validate_multi_write_response(response, outline)
        except Exception as primary_exc:
            partial = self._usable_multi_write_partial(response, outline)
            if self._reasoning_effort() == "low":
                logger.warning(
                    "Low-reasoning multi-write failed (%s); " "retrying once with medium.",
                    primary_exc,
                )
                try:
                    return self._retry_multi_write_with_medium(prompt, outline)
                except Exception as fallback_exc:
                    logger.warning(
                        "Medium-reasoning multi-write fallback failed: %s",
                        fallback_exc,
                    )
            if partial:
                return partial
            logger.warning(
                "Multi-write step failed (chunk length %d): %s",
                len(chunk),
                primary_exc,
            )
            return {}

    def _retry_multi_write_with_medium(
        self,
        prompt: str,
        outline: list[dict[str, str]],
    ) -> dict[str, str]:
        """Retry one failed GPT-OSS map with a larger medium budget."""
        response = self.client.chat(
            prompt,
            system=self._system_prompt,
            temperature=self.temperature,
            max_tokens=MAX_MEDIUM_FALLBACK_TOKENS,
            response_format=PLAN_RESPONSE_FORMAT,
            reasoning_effort="medium",
        )
        return self._validate_multi_write_response(response, outline)

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
            f"{merge_note}\n\n" f"{bounded_join(drafts, data_budget - len(merge_note) - 2)}"
        )
        prompt = self._prompts["reduce_section"].format(
            content_kind=content_kind,
            heading=section["heading"],
            kind=section["kind"],
            chunk=reduce_data,
        )
        try:
            call_kwargs: dict[str, Any] = {
                "system": self._system_prompt,
                "temperature": self.temperature,
                "max_tokens": min(self.max_tokens, MAX_REDUCE_TOKENS),
            }
            reasoning_effort = self._reasoning_effort()
            if reasoning_effort:
                call_kwargs["reasoning_effort"] = reasoning_effort
            response = self.client.chat(prompt, **call_kwargs)
            if not isinstance(response, str) or not response.strip():
                raise ValueError("reduce returned empty content")
            return response.strip()
        except Exception as primary_exc:
            if self._reasoning_effort() == "low":
                logger.warning(
                    "Low-reasoning reduce failed for section '%s' (%s); "
                    "retrying once with medium.",
                    section["heading"],
                    primary_exc,
                )
                try:
                    return self._retry_reduce_with_medium(prompt)
                except Exception as fallback_exc:
                    logger.warning(
                        "Medium-reasoning reduce fallback failed for section '%s': %s",
                        section["heading"],
                        fallback_exc,
                    )
            logger.warning(
                "Reduce step failed for section '%s': %s",
                section["heading"],
                primary_exc,
            )
            return joined

    def _retry_reduce_with_medium(self, prompt: str) -> str:
        """Retry one failed GPT-OSS reduce with a larger medium budget."""
        response = self.client.chat(
            prompt,
            system=self._system_prompt,
            temperature=self.temperature,
            max_tokens=MAX_MEDIUM_FALLBACK_TOKENS,
            reasoning_effort="medium",
        )
        if not isinstance(response, str) or not response.strip():
            raise ValueError("medium reduce returned empty content")
        return response.strip()

    def _validate_multi_write_response(
        self,
        response: object,
        outline: list[dict[str, str]],
    ) -> dict[str, str]:
        """Require valid JSON with one string value for every planned section."""
        data = _parse_json_object(response)
        expected_ids = [section["id"] for section in outline]
        missing = [section_id for section_id in expected_ids if section_id not in data]
        if missing:
            raise ValueError(f"multi-write response is missing section IDs: {missing}")
        invalid = [
            section_id for section_id in expected_ids if not isinstance(data[section_id], str)
        ]
        if invalid:
            raise ValueError(f"multi-write response has non-string sections: {invalid}")
        return {section_id: data[section_id] for section_id in expected_ids}

    def _usable_multi_write_partial(
        self,
        response: object,
        outline: list[dict[str, str]],
    ) -> dict[str, str]:
        """Preserve valid string sections if a completeness fallback fails."""
        try:
            data = _parse_json_object(response)
        except Exception:
            return {}
        return {
            section["id"]: data[section["id"]]
            for section in outline
            if isinstance(data.get(section["id"]), str)
        }

    def _reasoning_effort(self) -> str | None:
        """Use the tested low-reasoning budget only for Groq GPT-OSS models."""
        model = safe_text(getattr(self.client, "model", ""), "").lower()
        return "low" if model.startswith(GPT_OSS_MODEL_PREFIX) else None

    def _representative_sample(self, text: str, max_chars: int) -> str:
        return representative_sample(text, max_chars)

    def _format_transcript(self, result: TranscriptResult) -> str:
        """Format sentence timestamps and speakers for grounding."""
        if result.sentence_info:
            lines = []
            for sentence in result.sentence_info:
                speaker = "" if sentence.speaker is None else (f"Speaker {sentence.speaker}: ")
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
