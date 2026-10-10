"""CRUD endpoints for meetings."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from meetasr.backend.db.connection import get_db
from meetasr.backend.db import repository
from meetasr.backend.db.models import (
    Meeting, Transcript, Report, Sentence, Topic, ActionItem, Decision
)

router = APIRouter(tags=["Database"])

# --- Schemas ---

class MeetingReportResponse(BaseModel):
    """Schema for returning full meeting details."""
    id: str
    title: str
    status: str
    duration: float
    audio_path: str
    created_at: datetime
    updated_at: datetime
    
    transcript: Optional[Transcript] = None
    sentences: List[Sentence] = []
    report: Optional[Report] = None
    topics: List[Topic] = []
    action_items: List[ActionItem] = []
    decisions: List[Decision] = []

    class Config:
        from_attributes = True

# --- Routes ---

@router.get("/v1/meetings", response_model=List[Meeting])
def list_meetings(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)) -> List[Meeting]:
    """Retrieve a list of meetings with pagination."""
    return repository.get_meetings(db, skip=skip, limit=limit)


@router.get("/v1/meetings/{meeting_id}/report", response_model=MeetingReportResponse)
def get_meeting_report(meeting_id: str, db: Session = Depends(get_db)) -> Meeting:
    """Get complete details of a meeting, including its transcript and reports."""
    meeting = repository.get_meeting(db, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    return meeting


@router.delete("/v1/meetings/{meeting_id}", status_code=204)
def delete_meeting_endpoint(meeting_id: str, db: Session = Depends(get_db)) -> None:
    """Delete a meeting and all associated data."""
    success = repository.delete_meeting(db, meeting_id)
    if not success:
        raise HTTPException(status_code=404, detail="Meeting not found")
