"""Database repository for CRUD operations on Meeting data."""

import logging
import json
from typing import Optional, List
from sqlmodel import Session, select
from meetasr.db.models import Meeting, Transcript, Sentence, Report, Topic, ActionItem, Decision
from meetasr.schemas import MeetingReport

logger = logging.getLogger(__name__)

def create_meeting(
    db: Session, 
    id: str, 
    title: str, 
    audio_path: str, 
    asr_model: Optional[str] = None, 
    llm_model: Optional[str] = None
) -> Meeting:
    """Create a new pending meeting record."""
    meeting = Meeting(
        id=id,
        title=title,
        audio_path=audio_path,
        asr_model=asr_model,
        llm_model=llm_model,
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

        # 1. Save Transcript
        db.add(Transcript(
            meeting_id=meeting_id,
            text=report_data.transcript.text,
            duration=report_data.transcript.duration,
            language=report_data.transcript.language
        ))

        # 2. Save Sentences
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

        # 3. Save Report Summary
        if report_data.summary:
            db.add(Report(
                meeting_id=meeting_id,
                summary=report_data.summary,
                processing_time=report_data.processing_time
            ))

        # 4. Save Topics
        for topic in report_data.topics:
            db.add(Topic(
                meeting_id=meeting_id,
                title=topic.title,
                description=topic.description,
                start_time=topic.start_time,
                end_time=topic.end_time
            ))

        # 5. Save Action Items
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

        # 6. Save Decisions
        for decision in report_data.decisions:
            db.add(Decision(
                meeting_id=meeting_id,
                content=decision.content,
                made_by=decision.made_by,
                timestamp=decision.timestamp
            ))

        # Commit everything as a single transaction
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save meeting results for {meeting_id}: {e}")
        raise
