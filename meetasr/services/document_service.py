"""Shared Phase 2 document orchestration independent of HTTP."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from sqlmodel import Session, select

from meetasr.db.models_phase2 import (
    Document,
    DocumentMode,
    Job,
    Source,
    TranscriptSegment,
)
from meetasr.db.user_model import User
from meetasr.export import (
    ExportArtifact,
    ExportService,
    create_export_service,
)
from meetasr.export.markdown_parser import strip_leading_document_heading
from meetasr.schemas import SentenceInfo, TranscriptResult
from meetasr.utils.document_naming import (
    export_title,
    source_document_label,
)

FinalDocumentMode = Literal["summary", "full_text"]


class DocumentServiceError(Exception):
    """Base error raised by the shared Phase 2 document service."""


class DocumentNotFoundError(DocumentServiceError):
    """Requested Document does not exist."""


class InvalidDocumentStateError(DocumentServiceError):
    """Document/Source state cannot satisfy the requested operation."""


class TranscriptUnavailableError(DocumentServiceError):
    """A Source has no persisted transcript to finalize."""


class PlannerUnavailableError(DocumentServiceError):
    """Summary generation was requested without a configured planner."""


class DocumentGenerationError(DocumentServiceError):
    """The configured planner could not produce a usable final document."""


class DocumentService:
    """Build, persist and export Documents from canonical transcript data."""

    def __init__(
        self,
        db: Session,
        *,
        planner: Any | None = None,
        export_service: ExportService | None = None,
    ) -> None:
        self.db = db
        self.planner = planner
        self.export_service = export_service

    def get_document(self, document_id: str) -> Document:
        document = self.db.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(
                f"Document '{document_id}' không tồn tại."
            )
        return document

    def build_transcript(self, source_id: str) -> TranscriptResult:
        """Rehydrate the pipeline transcript contract from persisted segments."""
        source = self.db.get(Source, source_id)
        if source is None:
            raise InvalidDocumentStateError(
                f"Source '{source_id}' không còn tồn tại."
            )

        job = self.db.exec(
            select(Job).where(Job.source_id == source.id)
        ).first()
        if job is None:
            raise TranscriptUnavailableError("Source chưa có Job xử lý.")

        segments = self.db.exec(
            select(TranscriptSegment)
            .where(TranscriptSegment.job_id == job.id)
            .order_by(
                TranscriptSegment.start_ms,
                TranscriptSegment.id,
            )
        ).all()
        if not segments:
            raise TranscriptUnavailableError(
                "Chưa có transcript để tạo tài liệu."
            )

        sentences = [
            SentenceInfo(
                text=segment.text,
                start=segment.start_ms / 1000,
                end=segment.end_ms / 1000,
                speaker=segment.speaker,
            )
            for segment in segments
        ]
        duration = max(
            source.duration or 0.0,
            max((sentence.end for sentence in sentences), default=0.0),
        )
        return TranscriptResult(
            key=source.filename,
            text=" ".join(sentence.text for sentence in sentences),
            duration=duration,
            language="vi",
            sentence_info=sentences,
        )

    async def finalize_document(
        self,
        live_document_id: str,
        mode: FinalDocumentMode,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> Document:
        """Create one final Document from a live Document ID.

        Finalization is idempotent: once a non-empty document for the requested
        mode exists, return it instead of charging the LLM provider again.
        """
        live_document = self.get_document(live_document_id)
        if live_document.mode != DocumentMode.LIVE:
            raise InvalidDocumentStateError(
                "Finalize phải bắt đầu từ Document có mode='live'."
            )

        if mode not in (DocumentMode.SUMMARY, DocumentMode.FULL_TEXT):
            raise InvalidDocumentStateError(
                f"Document mode '{mode}' không hỗ trợ finalize."
            )

        document = self.db.exec(
            select(Document)
            .where(Document.source_id == live_document.source_id)
            .where(Document.mode == mode)
        ).first()
        if document is not None and document.markdown.strip():
            return document

        transcript = self.build_transcript(live_document.source_id)
        if mode == DocumentMode.FULL_TEXT:
            source = self.db.get(Source, live_document.source_id)
            source_label = (
                source_document_label(source.filename, source.created_at)
                if source is not None
                else None
            )
            markdown = self.full_text_markdown(
                transcript,
                source_label=source_label,
            )
        else:
            if self.planner is None:
                raise PlannerUnavailableError(
                    "DocumentPlanner chưa được cấu hình."
                )
            try:
                if progress_callback is None:
                    report = await asyncio.to_thread(
                        self.planner.plan_and_write,
                        transcript,
                    )
                else:
                    report = await asyncio.to_thread(
                        self.planner.plan_and_write,
                        transcript,
                        progress_callback=progress_callback,
                    )
            except Exception as exc:
                raise DocumentGenerationError(
                    "Dịch vụ AI chưa thể tạo bản tóm tắt. "
                    "Có thể nhà cung cấp đang giới hạn lượt gọi; "
                    "vui lòng đợi khoảng một phút rồi thử lại."
                ) from exc

            usable_sections = [
                section
                for section in report.sections
                if section.markdown.strip()
            ]
            if not usable_sections:
                raise DocumentGenerationError(
                    "Dịch vụ AI chưa trả về nội dung tóm tắt hợp lệ. "
                    "Vui lòng đợi khoảng một phút rồi thử lại."
                )
            markdown = report.to_markdown()

        if document is None:
            document = Document(
                source_id=live_document.source_id,
                mode=mode,
            )
        document.markdown = markdown
        document.updated_at = datetime.utcnow()
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def save_live_document(self, source_id: str, markdown: str) -> Document:
        """Create or replace the deterministic live draft for a Source."""
        source = self.db.get(Source, source_id)
        if source is None:
            raise InvalidDocumentStateError(
                f"Source '{source_id}' không còn tồn tại."
            )

        document = self.db.exec(
            select(Document)
            .where(Document.source_id == source_id)
            .where(Document.mode == DocumentMode.LIVE)
        ).first()
        if document is None:
            document = Document(
                source_id=source_id,
                mode=DocumentMode.LIVE,
            )
        document.markdown = markdown
        document.updated_at = datetime.utcnow()
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def export_document(
        self,
        document_id: str,
        format: Literal["md", "docx", "pdf"],
        preset: str | None = None,
    ) -> ExportArtifact:
        """Convert a persisted canonical Markdown document into a file."""
        document = self.get_document(document_id)
        source = self.db.get(Source, document.source_id)
        title = export_title(
            source.filename if source is not None else None,
            source.created_at if source is not None else None,
            document.mode,
        )
        author = "MeetingMind AI"
        if source is not None and source.user_id is not None:
            owner = self.db.get(User, source.user_id)
            if owner is not None and owner.name.strip():
                author = owner.name.strip()
        markdown = document.markdown
        if document.mode == DocumentMode.FULL_TEXT:
            body = strip_leading_document_heading(markdown)
            if source is not None:
                source_note = (
                    f"_Nguồn: {source_document_label(source.filename, source.created_at)}_"
                )
                if not body.startswith("_Nguồn:"):
                    body = f"{source_note}\n\n{body}" if body else source_note
            markdown = "# Toàn văn cuộc họp"
            if body:
                markdown = f"{markdown}\n\n{body}"
        export_service = self.export_service or create_export_service()
        return export_service.export(
            markdown,
            format,
            title=title,
            preset=preset,
            context={"author": author},
        )

    @staticmethod
    def full_text_markdown(
        transcript: TranscriptResult,
        *,
        source_label: str | None = None,
    ) -> str:
        """Render a readable transcript without invoking an LLM."""
        label = source_label or Path(transcript.key).stem or "Tài liệu"
        lines = ["# Toàn văn cuộc họp", "", f"_Nguồn: {label}_", ""]
        if not transcript.sentence_info:
            lines.append("_Không phát hiện lời nói trong bản ghi âm._")
        for sentence in transcript.sentence_info:
            minutes, seconds = divmod(int(sentence.start), 60)
            speaker = (
                f"Người nói {sentence.speaker + 1}"
                if sentence.speaker is not None
                else "Chưa xác định"
            )
            lines.extend(
                [
                    (
                        f"**[{minutes:02d}:{seconds:02d}] "
                        f"{speaker}:** {sentence.text}"
                    ),
                    "",
                ]
            )
        return "\n".join(lines).strip()
