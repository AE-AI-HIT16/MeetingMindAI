"""DB models cho Phase 2 — NotebookLM-style audio/video pipeline.
    Mối quan hệ giữa các bảng (Table relationships)::

    Source (1) ──► Job (1)
    Source (1) ──► Document (nhiều — summary, full_text, live…)
    Job    (1) ──► TranscriptSegment (nhiều)
"""
import uuid
from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


'''
    Enum class
'''

class JobStatus:
    """ Các giá trị hợp lệ cho ``Job.status``."""

    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class JobStage:
    """Các giá trị hợp lệ cho ``Job.stage``."""

    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    GENERATING_DOC = "generating_doc"


class DocumentMode:
    """Các giá trị hợp lệ cho ``Document.mode``."""

    LIVE = "live"           # intermediate — updated incrementally
    SUMMARY = "summary"     # structured summary (topics, decisions, action items)
    FULL_TEXT = "full_text" # lightly edited verbatim transcript


class MediaType:
    """Các giá trị hợp lệ cho ``Source.media_type``."""

    AUDIO = "audio"
    VIDEO = "video"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a UUID v4"""
    return str(uuid.uuid4())


class Source(SQLModel, table=True):
    """Quản lí tệp media gốc mà người dùng tải lên
        Lưu trữ metadata của file bao gồm tên gốc (filename),
     định dạng (media_type là audio/video), thời lượng (duration) và đường dẫn
    """

    __tablename__ = "sources"

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    filename: str = Field(index=True)           # original file name, e.g. "meeting.mp4"
    media_type: str                              # "audio" | "video"
    duration: Optional[float] = None            # seconds; None until extracted
    storage_path: str                           # local path or MinIO object key
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    job: Optional["Job"] = Relationship(
        back_populates="source",
        sa_relationship_kwargs={"uselist": False, "cascade": "all, delete-orphan"},
    )
    documents: List["Document"] = Relationship(
        back_populates="source",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Job(SQLModel, table=True):
    """
    Theo dõi tiến trình xử lý AI ngầm của một `Source`.
    """

    __tablename__ = "jobs"

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    source_id: str = Field(foreign_key="sources.id")

    status: str = Field(default=JobStatus.QUEUED)       # see JobStatus
    stage: str = Field(default="")                      # see JobStage
    progress: float = Field(default=0.0)                # 0.0 → 1.0
    error: Optional[str] = None                         # set only when status=failed

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    source: Optional[Source] = Relationship(back_populates="job")
    segments: List["TranscriptSegment"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )



class TranscriptSegment(SQLModel, table=True):
    """
    Phục vụ tính năng hiển thị chữ theo thời gian thực
    Lưu trữ từng câu/đoạn chữ nhỏ giọt được AI nhận diện xong. Chứa mốc
     thời gian bắt đầu/kết thúc (start_ms, end_ms), nhãn người nói (speaker) và
     nội dung chữ (text) và được ghi liên tục
    """

    __tablename__ = "transcript_segments"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="jobs.id", index=True)

    start_ms: int           # segment start, milliseconds from file beginning
    end_ms: int             # segment end, milliseconds from file beginning
    speaker: Optional[int] = None   # speaker label; None until diarization runs
    text: str               # recognised text for this segment

    # Relationships
    job: Optional[Job] = Relationship(back_populates="segments")


class Document(SQLModel, table=True):
    """
    Lưu trữ các tài liệu (Docs) định dạng Markdown do LLM sinh ra sau khi đọc bản bóc băng
    Thiết kế này cho phép một tệp `Source` có thể sinh ra nhiều tài liệu khác nhau mà không bị ghi đè:
     + 'live': Bản nháp đang được cập nhật cuốn chiếu.
     + 'summary': Bản tóm tắt có cấu trúc (chủ đề, quyết định, hành động).
     + 'full_text': Bản toàn văn đã được hiệu đính.
    """

    __tablename__ = "documents"

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    source_id: str = Field(foreign_key="sources.id", index=True)

    mode: str = Field(default=DocumentMode.LIVE)    # see DocumentMode
    markdown: str = Field(default="")               # full markdown content
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    source: Optional[Source] = Relationship(back_populates="documents")
