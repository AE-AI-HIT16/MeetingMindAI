"""Realtime recording WebSocket endpoint."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlmodel import Session

from meetasr.api.auth_deps import resolve_token_user
from meetasr.db.connection import get_db
from meetasr.db.models_phase2 import Job, JobStatus, MediaType, Source
from meetasr.storage.backend import StorageBackend
from meetasr.streaming.asr_worker import ASRWorker
from meetasr.streaming.audio_archive import ArchivedAudio
from meetasr.streaming.audio_receiver import AudioReceiver
from meetasr.streaming.final_transcript_queue import FinalTranscriptJob
from meetasr.streaming.graceful_stop import drain_realtime_session
from meetasr.streaming.partial_buffer_cleaner import PartialBufferCleaner
from meetasr.streaming.protocol import RealtimeAuthFrame, receive_stream_frame
from meetasr.streaming.session import StreamSession
from meetasr.streaming.temp_asr_woker import TempASRWorker
from meetasr.streaming.window_builder import SegmentWindowBuilder
from meetasr.streaming.worker import AudioWorker

router = APIRouter()
logger = logging.getLogger("realtime")


@router.websocket("/v1/realtime/stream")
async def realtime_stream(
    websocket: WebSocket,
    db: Session = Depends(get_db),
) -> None:
    """Receive PCM audio until stop, then drain all accepted audio safely."""
    await websocket.accept()
    logger.info("WS accepted client=%s", websocket.client)

    try:
        first_frame = await receive_stream_frame(websocket)
    except WebSocketDisconnect:
        return

    user_id = None
    pending_frame = first_frame
    if isinstance(first_frame, RealtimeAuthFrame):
        pending_frame = None
        if first_frame.token:
            try:
                user_id = resolve_token_user(first_frame.token, db).id
            except HTTPException as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "realtime_auth_failed",
                        "message": exc.detail,
                    }
                )
                await websocket.close(code=1008)
                return

    session = StreamSession(websocket)
    source, job = _create_realtime_job(db, user_id)
    session.source_id = source.id
    session.job_id = job.id

    await websocket.send_json(
        {
            "type": "session_init",
            "source_id": source.id,
            "job_id": job.id,
        }
    )

    pipeline = getattr(websocket.app.state, "realtime_pipeline", None)
    asr_service = getattr(websocket.app.state, "realtime_asr_service", None)
    if pipeline is None or asr_service is None:
        logger.error("Realtime pipeline not loaded")
        await websocket.close(code=1013)
        return

    logger.info("Realtime VAD: %s", type(pipeline.vad).__name__)
    receiver = AudioReceiver(session)
    window_builder = SegmentWindowBuilder(session)
    asr_worker = ASRWorker(session, asr_service)
    temp_asr_worker = TempASRWorker(
        session,
        pipeline,
        coordinator=getattr(
            websocket.app.state,
            "inference_coordinator",
            None,
        ),
    )
    partial_buffer_cleaner = PartialBufferCleaner(session)
    worker = AudioWorker(session, pipeline, window_builder)

    session.asr_task = asyncio.create_task(asr_worker.run())
    session.temp_asr_task = asyncio.create_task(temp_asr_worker.run())
    session.partial_cleaner_task = asyncio.create_task(
        partial_buffer_cleaner.run()
    )
    session.worker_task = asyncio.create_task(worker.run())

    stop_requested = False
    drain_succeeded = False

    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                frame = await receive_stream_frame(websocket)
            if frame == "stop":
                stop_requested = True
                await websocket.send_json(
                    {
                        "type": "stream_stopping",
                        "job_id": session.job_id,
                    }
                )
                break
            if isinstance(frame, RealtimeAuthFrame):
                raise ValueError("Realtime auth frame is only valid before audio.")

            session.audio_archive.append(frame)
            await receiver.receive(frame)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception:
        logger.exception("Realtime stream crashed")
    finally:
        try:
            await drain_realtime_session(session, receiver, worker)
            drain_succeeded = True
        except Exception:
            logger.exception("Realtime session drain failed")

        finalizer_enqueued = await _enqueue_final_transcript(
            websocket,
            session,
            db=db,
        )
        if not finalizer_enqueued:
            stored_job = db.get(Job, session.job_id)
            if stored_job is not None:
                stored_job.status = JobStatus.FAILED
                stored_job.error = "Final transcript queue unavailable."
                db.add(stored_job)
                db.commit()

        if stop_requested:
            if drain_succeeded and finalizer_enqueued:
                event = {
                    "type": "stream_stopped",
                    "job_id": session.job_id,
                }
            elif not drain_succeeded:
                event = {
                    "type": "error",
                    "code": "stream_drain_failed",
                    "message": "Không thể xử lý hết phần audio cuối.",
                }
            else:
                event = {
                    "type": "error",
                    "code": "stream_finalization_failed",
                    "message": "Không thể xếp lịch hoàn tất audio realtime.",
                }
            try:
                await websocket.send_json(event)
            except Exception:
                logger.exception("Failed to acknowledge realtime stop")

        try:
            await session.close()
        except Exception:
            logger.exception("Session close failed")

        if stop_requested:
            try:
                await websocket.close(code=1000)
            except Exception:
                logger.debug("Realtime socket already closed", exc_info=True)

        logger.info("WS cleanup done")


def _create_realtime_job(
    db: Session,
    user_id: str | None,
) -> tuple[Source, Job]:
    """Create an owned realtime Source and its processing Job."""
    source = Source(
        filename=f"realtime_{datetime.utcnow().isoformat()}.wav",
        media_type=MediaType.AUDIO,
        storage_path="",
        user_id=user_id,
    )
    db.add(source)
    db.flush()
    job = Job(source_id=source.id, status=JobStatus.PROCESSING)
    db.add(job)
    db.commit()
    db.refresh(source)
    db.refresh(job)
    return source, job


async def _enqueue_final_transcript(
    websocket: WebSocket,
    session: StreamSession,
    *,
    db: Session,
    storage: StorageBackend | None = None,
) -> bool:
    """Queue the full accepted recording for post-session finalization."""
    audio = None
    try:
        final_transcript_queue = getattr(
            websocket.app.state,
            "final_transcript_queue",
            None,
        )
        if final_transcript_queue is None:
            logger.error("Final transcript queue not initialized")
            return False

        audio = session.audio_archive.detach()
        await _persist_realtime_media(
            session,
            audio,
            db,
            storage,
        )
        enqueued = await final_transcript_queue.put(
            FinalTranscriptJob(
                job_id=session.job_id,
                audio=audio,
                coverage=session.coverage.snapshot(),
            )
        )
        if not enqueued:
            audio.cleanup()
            return False
        logger.info(
            "Final transcription queued samples=%d spooled=%s",
            audio.sample_count,
            audio.path is not None,
        )
        return True
    except Exception:
        if audio is not None:
            audio.cleanup()
        logger.exception("Failed to enqueue offline job")
        return False


async def _persist_realtime_media(
    session: StreamSession,
    audio: ArchivedAudio,
    db: Session,
    storage: StorageBackend | None,
) -> None:
    """Store a playable WAV and commit its key before finalization starts."""
    from meetasr.api.routes.sources import get_storage_backend

    source = db.get(Source, session.source_id)
    if source is None:
        raise RuntimeError(f"Source '{session.source_id}' không tồn tại.")
    backend = storage or get_storage_backend()
    wav_stream = audio.open_wav()
    storage_key = None
    try:
        storage_key = await backend.save(wav_stream, source.filename)
        source.storage_path = storage_key
        db.add(source)
        db.commit()
    except Exception:
        db.rollback()
        if storage_key is not None:
            await backend.delete(storage_key)
        raise
    finally:
        wav_stream.close()
