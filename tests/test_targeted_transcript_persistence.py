"""Transaction tests for targeted realtime transcript persistence."""

from __future__ import annotations

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from meetasr.api.schemas_phase2 import TranscriptSegmentPayload
from meetasr.db.models_phase2 import Job, MediaType, Source, TranscriptSegment
from meetasr.schemas import (
    SentenceInfo,
    TargetedRetranscriptionResult,
    TargetedRetranscriptionStats,
)
from meetasr.services.targeted_transcript_persistence import (
    persist_targeted_transcript,
)


def test_invalid_replacement_rolls_back_every_segment_change() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        source = Source(
            filename="realtime.wav",
            media_type=MediaType.AUDIO,
            storage_path="realtime.wav",
        )
        db.add(source)
        db.flush()
        job = Job(source_id=source.id)
        db.add(job)
        db.flush()
        records = [
            TranscriptSegment(
                job_id=job.id,
                start_ms=0,
                end_ms=1000,
                text="giữ nguyên",
            ),
            TranscriptSegment(
                job_id=job.id,
                start_ms=1000,
                end_ms=2000,
                text="câu hỗn hợp",
            ),
        ]
        db.add_all(records)
        db.commit()
        for record in records:
            db.refresh(record)
        job_id = job.id
        source_id = source.id
        persisted = [TranscriptSegmentPayload.from_db(record) for record in records]

    finalized = TargetedRetranscriptionResult(
        sentence_info=[
            SentenceInfo(text="giữ nguyên", start=0.0, end=1.0, speaker=0),
            SentenceInfo(text="không hợp lệ", start=1.5, end=1.5, speaker=1),
        ],
        source_indices=[0, 1],
        replaced_indices=(1,),
        stats=TargetedRetranscriptionStats(1, 1, 1, 0, 500),
    )

    with pytest.raises(
        RuntimeError,
        match="invalid replacement segment",
    ):
        persist_targeted_transcript(
            engine,
            job_id,
            persisted,
            finalized,
            2000,
        )

    with Session(engine) as db:
        records = db.exec(
            select(TranscriptSegment)
            .where(TranscriptSegment.job_id == job_id)
            .order_by(TranscriptSegment.start_ms)
        ).all()
        assert [record.text for record in records] == [
            "giữ nguyên",
            "câu hỗn hợp",
        ]
        assert [record.speaker for record in records] == [None, None]
        assert db.get(Source, source_id).duration is None
    engine.dispose()
