"""Public API contracts for the Phase 2 source-processing flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field, model_validator


if TYPE_CHECKING:
    from meetasr.backend.db.models_phase2 import (
        Document,
        DocumentGenerationJob,
        TranscriptSegment,
    )


# ============================================================
# ASR INTERNAL SCHEMAS
# ============================================================


@dataclass
class Segment:
    """A speech segment detected by VAD."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def start_s(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_s(self) -> float:
        return self.end_ms / 1000.0


@dataclass
class SpeakerTurn:
    """A time range attributed to one diarized speaker."""

    start_ms: int
    end_ms: int
    speaker: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_segment(self) -> Segment:
        """Return time range without speaker label."""
        return Segment(
            self.start_ms,
            self.end_ms,
        )


@dataclass
class SentenceInfo:
    """A single sentence with timing and speaker information."""

    text: str
    start: float
    end: float
    speaker: Optional[int] = None
    char_timestamps: list[list[int]] = field(default_factory=list)


# ============================================================
# PUBLIC API CONTRACTS
# ============================================================


class CreateSourceResponse(BaseModel):
    """Identifiers needed to display a Source and follow its processing Job."""

    sourceId: str
    jobId: str
    status: Literal["queued"]


class TranscriptSegmentPayload(BaseModel):
    """Canonical transcript segment sent through REST and WebSocket."""

    id: int | None = None
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker: int | None = Field(default=None, ge=0)
    text: str

    @model_validator(mode="after")
    def validate_time_range(self) -> "TranscriptSegmentPayload":
        """Reject invalid timestamps."""
        if self.end_ms < self.start_ms:
            raise ValueError(
                "end_ms must be greater than or equal to start_ms"
            )
        return self

    @classmethod
    def from_db(
        cls,
        segment: "TranscriptSegment",
    ) -> "TranscriptSegmentPayload":
        return cls(
            id=segment.id,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            speaker=segment.speaker,
            text=segment.text,
        )

if TYPE_CHECKING:
    from meetasr.backend.db.models_phase2 import (
        Document,
        DocumentGenerationJob,
        TranscriptSegment,
    )


class CreateSourceResponse(BaseModel):
    """Identifiers needed to display a Source and follow its processing Job."""

    sourceId: str
    jobId: str
    status: Literal["queued"]


class TranscriptSegmentPayload(BaseModel):
    """Canonical transcript segment sent through REST and WebSocket."""

    id: int | None = None
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker: int | None = Field(default=None, ge=0)
    text: str

    @model_validator(mode="after")
    def validate_time_range(self) -> "TranscriptSegmentPayload":
        """Reject segments whose end timestamp precedes their start."""
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self

    @classmethod
    def from_db(cls, segment: "TranscriptSegment") -> "TranscriptSegmentPayload":
        """Convert a persisted Phase 2 segment into its public payload."""
        return cls(
            id=segment.id,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            speaker=segment.speaker,
            text=segment.text,
        )


class StatusEvent(BaseModel):
    """Job stage/progress update."""

    type: Literal["status"] = "status"
    stage: Literal["extracting_audio", "transcribing", "generating_doc"]
    progress: float = Field(ge=0.0, le=1.0)


class TranscriptDeltaEvent(BaseModel):
    """A newly persisted transcript segment."""

    type: Literal["transcript_delta"] = "transcript_delta"
    segment: TranscriptSegmentPayload


class DocDeltaEvent(BaseModel):
    """A complete replacement for one live-document section."""

    type: Literal["doc_delta"] = "doc_delta"
    section_id: str
    markdown: str


class SpeakerAssignment(BaseModel):
    """Final speaker assigned to one stable persisted segment."""

    segment_id: int
    speaker: int | None = Field(default=None, ge=0)


class SpeakerUpdateEvent(BaseModel):
    """Apply full-file diarization results by persisted segment ID."""

    type: Literal["speaker_update"] = "speaker_update"
    updates: list[SpeakerAssignment]


class DoneEvent(BaseModel):
    """Terminal success event for an upload Job."""

    type: Literal["done"] = "done"
    duration_ms: int = Field(ge=0)
    num_segments: int = Field(ge=0)
    live_document_id: str | None = None


class ErrorEvent(BaseModel):
    """Terminal failure event for an upload Job."""

    type: Literal["error"] = "error"
    code: str
    message: str


class FinalizeDocumentRequest(BaseModel):
    """Requested final representation of a live document."""

    mode: Literal["summary", "full_text"]


class DocumentGenerationResponse(BaseModel):
    """Accepted/background state returned by the finalize endpoint."""

    generationJobId: str
    documentId: str
    sourceId: str
    mode: Literal["summary", "full_text"]
    status: Literal["queued", "processing", "done", "failed"]
    stage: Literal["queued", "generating", "saving", "done", "failed"]
    progress: float = Field(ge=0.0, le=1.0)
    error: str | None = None

    @classmethod
    def from_db(
        cls,
        generation: "DocumentGenerationJob",
    ) -> "DocumentGenerationResponse":
        return cls(
            generationJobId=generation.id,
            documentId=generation.document_id,
            sourceId=generation.source_id,
            mode=generation.mode,
            status=generation.status,
            stage=generation.stage,
            progress=generation.progress,
            error=generation.error,
        )


class DocumentGenerationStatusEvent(BaseModel):
    """Progress update for one background document-generation job."""

    type: Literal["document_status"] = "document_status"
    generation_job_id: str
    document_id: str
    stage: Literal["queued", "generating", "saving"]
    progress: float = Field(ge=0.0, le=1.0)


class DocumentGenerationDoneEvent(BaseModel):
    """Terminal success for background document generation."""

    type: Literal["document_done"] = "document_done"
    generation_job_id: str
    document_id: str
    mode: Literal["summary", "full_text"]


class DocumentGenerationErrorEvent(BaseModel):
    """Terminal failure for background document generation."""

    type: Literal["document_error"] = "document_error"
    generation_job_id: str
    document_id: str
    code: str
    message: str
    retry_after: int | None = Field(default=None, ge=0)


class DocumentResponse(BaseModel):
    """Public Phase 2 Document representation."""

    id: str
    sourceId: str
    mode: Literal["live", "summary", "full_text"]
    markdown: str
    createdAt: str
    updatedAt: str

    @classmethod
    def from_db(cls, document: "Document") -> "DocumentResponse":
        return cls(
            id=document.id,
            sourceId=document.source_id,
            mode=document.mode,
            markdown=document.markdown,
            createdAt=document.created_at.isoformat(),
            updatedAt=document.updated_at.isoformat(),
        )
