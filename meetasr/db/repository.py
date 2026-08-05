"""Database repository for CRUD operations on Meeting data."""

import json
import logging
from typing import List, Optional

from sqlmodel import Session, select

from meetasr.db.models import ActionItem, Decision, Meeting, Report, Sentence, Topic, Transcript
from meetasr.schemas import MeetingReport

logger = logging.getLogger(__name__)

def create_meeting(
    db: Session,
    id: str,
    title: str,
    audio_path: str,
    asr_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    duration: float = 0.0
) -> Meeting:
    """Create a new pending meeting record."""
    meeting = Meeting(
        id=id,
        title=title,
        audio_path=audio_path,
        asr_model=asr_model,
        llm_model=llm_model,
        duration=duration,
        status="pending"
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    return meeting


def update_meeting_status(db: Session, meeting_id: str, status: str) -> Optional[Meeting]:
    """Update the status of a meeting."""
    meeting = db.get(Meeting, meeting_id)
    if not meeting:
        logger.error(f"Meeting {meeting_id} not found.")
        return None
    meeting.status = status
    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    return meeting


def get_meeting(db: Session, meeting_id: str) -> Optional[Meeting]:
    """Retrieve a meeting by ID."""
    return db.get(Meeting, meeting_id)


def save_meeting_result(db: Session, meeting_id: str, report_data: MeetingReport) -> None:
    """
    Save the complete results of a meeting (transcript, sentences, report, etc.).
    Uses a single transaction to ensure data integrity.
    """
    meeting = db.get(Meeting, meeting_id)
    if not meeting:
        raise ValueError(f"Meeting {meeting_id} not found")

    try:
        # Update meeting metadata
        meeting.duration = report_data.transcript.duration
        if report_data.asr_model:
            meeting.asr_model = report_data.asr_model
        if report_data.llm_model:
            meeting.llm_model = report_data.llm_model
        db.add(meeting)

        # save Transcript
        db.add(Transcript(
            meeting_id=meeting_id,
            text=report_data.transcript.text,
            duration=report_data.transcript.duration,
            language=report_data.transcript.language
        ))

        # save Sentences
        for sent in report_data.transcript.sentence_info:
            char_ts = json.dumps(sent.char_timestamps) if sent.char_timestamps else None
            db.add(Sentence(
                meeting_id=meeting_id,
                text=sent.text,
                start=sent.start,
                end=sent.end,
                speaker=sent.speaker,
                char_timestamps=char_ts
            ))

        # save Report Summary
        if report_data.summary:
            db.add(Report(
                meeting_id=meeting_id,
                summary=report_data.summary,
                processing_time=report_data.processing_time,
                llm_model=report_data.llm_model
            ))

        #save Topics
        for topic in report_data.topics:
            db.add(Topic(
                meeting_id=meeting_id,
                title=topic.title,
                description=topic.description,
                start_time=topic.start_time,
                end_time=topic.end_time
            ))

        #save Action Items
        for item in report_data.action_items:
            db.add(ActionItem(
                meeting_id=meeting_id,
                task=item.task,
                assignee=item.assignee,
                deadline=item.deadline,
                priority=item.priority,
                mentioned_by=item.mentioned_by,
                timestamp=item.timestamp
            ))

        for decision in report_data.decisions:
            db.add(Decision(
                meeting_id=meeting_id,
                content=decision.content,
                made_by=decision.made_by,
                timestamp=decision.timestamp
            ))

        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save meeting results for {meeting_id}: {e}")
        raise


def get_meetings(db: Session, skip: int = 0, limit: int = 20) -> List[Meeting]:
    """Retrieve a list of meetings with pagination."""
    return db.exec(select(Meeting).order_by(Meeting.created_at.desc()).offset(skip).limit(limit)).all()


def delete_meeting(db: Session, meeting_id: str) -> bool:
    """Delete a meeting and all its associated records (cascade)."""
    meeting = db.get(Meeting, meeting_id)
    if meeting:
        db.delete(meeting)
        db.commit()
        return True
    return False
