
from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class Transcript(SQLModel, table=True):
    __tablename__ = "transcripts"

    meeting_id: str = Field(primary_key=True, foreign_key="meetings.id")
    text: str
    duration: float
    language: str
    meeting: Optional["Meeting"] = Relationship(back_populates="transcript")


class Sentence(SQLModel, table=True):
    __tablename__ = "sentences"

    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: str = Field(foreign_key="meetings.id")
    text: str
    start: float
    end: float
    speaker: Optional[int] = None
    char_timestamps: Optional[str] = None

    meeting: Optional["Meeting"] = Relationship(back_populates="sentences")


class Report(SQLModel, table=True):
    __tablename__ = "reports"

    meeting_id: str = Field(primary_key=True, foreign_key="meetings.id")
    summary: str
    processing_time: float

    #lưu trữ cấu hình model llm
    llm_model: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    meeting: Optional["Meeting"] = Relationship(back_populates="report")


class Topic(SQLModel, table=True):
    __tablename__ = "topics"

    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: str = Field(foreign_key="meetings.id")
    title: str
    description: str
    start_time: float
    end_time: float

    meeting: Optional["Meeting"] = Relationship(back_populates="topics")


class ActionItem(SQLModel, table=True):
    __tablename__ = "action_items"

    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: str = Field(foreign_key="meetings.id")
    task: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    priority: str = Field(default="medium")
    mentioned_by: Optional[str] = None
    timestamp: Optional[float] = None

    meeting: Optional["Meeting"] = Relationship(back_populates="action_items")


class Decision(SQLModel, table=True):
    __tablename__ = "decisions"

    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: str = Field(foreign_key="meetings.id")
    content: str
    made_by: str = Field(default="Team")
    timestamp: Optional[float] = None

    meeting: Optional["Meeting"] = Relationship(back_populates="decisions")


class Meeting(SQLModel, table=True):
    __tablename__ = "meetings"

    id: str = Field(primary_key=True)
    title: str
    duration: float = 0.0
    status: str = Field(default="pending")
    language: str = "vi"
    audio_path: str
    asr_model: Optional[str] = None
    llm_model: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    #relationships
    transcript: Optional[Transcript] = Relationship(
        back_populates="meeting",
        sa_relationship_kwargs={"uselist": False, "cascade": "all, delete-orphan"}
    )
    report: Optional[Report] = Relationship(
        back_populates="meeting",
        sa_relationship_kwargs={"uselist": False, "cascade": "all, delete-orphan"}
    )
    sentences: List[Sentence] = Relationship(
        back_populates="meeting",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    topics: List[Topic] = Relationship(
        back_populates="meeting",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    action_items: List[ActionItem] = Relationship(
        back_populates="meeting",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    decisions: List[Decision] = Relationship(
        back_populates="meeting",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
