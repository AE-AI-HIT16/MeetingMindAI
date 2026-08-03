"""Transactional persistence for targeted realtime transcript replacements."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from meetasr.api.schemas_phase2 import (
    SpeakerAssignment,
    TranscriptSegmentPayload,
)
from meetasr.db.models_phase2 import Job, Source, TranscriptSegment
from meetasr.schemas import SentenceInfo, TargetedRetranscriptionResult


@dataclass(frozen=True, slots=True)
class TargetedPersistenceResult:
    """Committed transcript state after targeted replacements."""

    source_id: str
    segments: list[TranscriptSegmentPayload]
    speaker_updates: list[SpeakerAssignment]


def persist_targeted_transcript(
    engine: Engine,
    job_id: str,
    persisted: list[TranscriptSegmentPayload],
    finalized: TargetedRetranscriptionResult,
    duration_ms: int,
) -> TargetedPersistenceResult:
    """Replace targeted rows atomically while retaining stable single-row IDs."""
    if len(finalized.sentence_info) != len(finalized.source_indices):
        raise ValueError("finalized source mapping must align with sentences")

    grouped = _group_by_source_index(finalized)
    replaced_indices = set(finalized.replaced_indices)
    with Session(engine) as db:
        try:
            job = db.get(Job, job_id)
            if job is None:
                raise RuntimeError(f"Job '{job_id}' không tồn tại.")
            source = db.get(Source, job.source_id)
            records = db.exec(
                select(TranscriptSegment)
                .where(TranscriptSegment.job_id == job_id)
                .order_by(TranscriptSegment.start_ms, TranscriptSegment.id)
            ).all()
            _validate_persisted_records(records, persisted, grouped)
            if source is None:
                raise RuntimeError(f"Source của Job '{job_id}' không tồn tại.")

            speaker_updates: list[SpeakerAssignment] = []
            for index, record in enumerate(records):
                outputs = grouped[index]
                if index in replaced_indices:
                    db.delete(record)
                    for sentence in outputs:
                        db.add(_replacement_record(job_id, record, sentence))
                    continue

                if len(outputs) != 1 or record.id is None:
                    raise RuntimeError("Stable realtime segment mapping is invalid.")
                record.speaker = outputs[0].speaker
                db.add(record)
                speaker_updates.append(
                    SpeakerAssignment(
                        segment_id=record.id,
                        speaker=record.speaker,
                    )
                )

            source.duration = max(source.duration or 0.0, duration_ms / 1000)
            db.add(source)
            db.commit()
            segments = db.exec(
                select(TranscriptSegment)
                .where(TranscriptSegment.job_id == job_id)
                .order_by(TranscriptSegment.start_ms, TranscriptSegment.id)
            ).all()
            return TargetedPersistenceResult(
                source_id=source.id,
                segments=[TranscriptSegmentPayload.from_db(record) for record in segments],
                speaker_updates=speaker_updates,
            )
        except Exception:
            db.rollback()
            raise


def _group_by_source_index(
    finalized: TargetedRetranscriptionResult,
) -> dict[int, list[SentenceInfo]]:
    grouped: dict[int, list[SentenceInfo]] = {}
    for source_index, sentence in zip(
        finalized.source_indices,
        finalized.sentence_info,
    ):
        grouped.setdefault(source_index, []).append(sentence)
    return grouped


def _validate_persisted_records(
    records: list[TranscriptSegment],
    persisted: list[TranscriptSegmentPayload],
    grouped: dict[int, list[SentenceInfo]],
) -> None:
    record_ids = [record.id for record in records]
    expected_ids = [item.id for item in persisted]
    if record_ids != expected_ids:
        raise RuntimeError("Persisted realtime transcript changed during finalization.")
    if set(grouped) != set(range(len(records))):
        raise RuntimeError("Finalized transcript does not cover every persisted segment.")


def _replacement_record(
    job_id: str,
    original: TranscriptSegment,
    sentence: SentenceInfo,
) -> TranscriptSegment:
    text = sentence.text.strip()
    start_ms = max(
        original.start_ms,
        min(original.end_ms, int(round(sentence.start * 1000))),
    )
    end_ms = max(
        start_ms,
        min(original.end_ms, int(round(sentence.end * 1000))),
    )
    if not text or end_ms <= start_ms:
        raise RuntimeError("Targeted ASR produced an invalid replacement segment.")
    return TranscriptSegment(
        job_id=job_id,
        start_ms=start_ms,
        end_ms=end_ms,
        speaker=sentence.speaker,
        text=text,
    )
