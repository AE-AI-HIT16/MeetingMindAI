from meetasr.backend.streaming.session import StreamSession
from meetasr.backend.streaming.worker import AudioWorker
from meetasr.backend.streaming.audio_receiver import AudioReceiver
from meetasr.backend.streaming.asr_worker import ASRWorker
from meetasr.backend.streaming.window_builder import SegmentWindowBuilder
import io
import wave
import asyncio
import logging
import time
from typing import Annotated, Optional

import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Request
from sqlmodel import Session
from meetasr.backend.db.connection import get_db, engine
from meetasr.backend.db.models_phase2 import Source, MediaType, Job, JobStatus
from meetasr.backend.db.user_model import User
from meetasr.backend.api.auth_deps import get_current_user, user_id_from_token

def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)      # mono
        wav_file.setsampwidth(2)      # 16-bit (2 bytes)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_io.getvalue()

router = APIRouter()

logger = logging.getLogger("realtime")


# Offline jobs started at stop outlive the WS handler; keep refs so they aren't GC'd.
_background_tasks: set[asyncio.Task] = set()


def _assign_owner(source_id: str, token: str | None) -> None:
    """Attach the realtime Source to the user behind ``token`` (if valid)."""
    user_id = user_id_from_token(token)
    if user_id is None:
        return
    with Session(engine) as db:
        source = db.get(Source, source_id)
        if source is not None and source.user_id is None and db.get(User, user_id):
            source.user_id = user_id
            db.add(source)
            db.commit()


async def _warmup(asr_service) -> None:
    try:
        await asr_service.warmup()
        logger.info("RunPod warmup job queued")
    except Exception:
        logger.warning("RunPod warmup failed", exc_info=True)


# Upload page warmup: at most one RunPod job per interval for all users.
WARMUP_MIN_INTERVAL_S = 30.0
_last_warmup_at = 0.0


@router.post("/v1/runpod/warmup", status_code=202)
async def runpod_warmup(
    request: Request,
    current_user: Annotated[Optional[User], Depends(get_current_user)],
):
    """Start a RunPod worker cold start while the user picks/uploads a file.

    Logged-in users only (each call can bill GPU time) and throttled.
    """
    global _last_warmup_at
    if current_user is None:
        raise HTTPException(status_code=401, detail="Login required")
    asr_service = getattr(request.app.state, "realtime_asr_service", None)
    now = time.monotonic()
    if asr_service is not None and now - _last_warmup_at >= WARMUP_MIN_INTERVAL_S:
        _last_warmup_at = now
        task = asyncio.create_task(_warmup(asr_service))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    return {"status": "accepted"}


@router.websocket("/v1/realtime/stream")
async def realtime_stream(websocket: WebSocket, db: Session = Depends(get_db)):
    """
    Endpoint WebSocket nhận audio realtime từ frontend.
    Mỗi client kết nối sẽ tạo ra một session độc lập.
    """

    await websocket.accept()

    logger.info(
        "WS accepted client=%s",
        websocket.client,
    )

    session = StreamSession(websocket)
    # -------------------------------------------------------------------
    # Create a Source and Job in the database for this realtime session
    # -------------------------------------------------------------------
    from datetime import datetime
    temp_filename = f"realtime_{datetime.utcnow().isoformat()}.wav"
    source = Source(filename=temp_filename, media_type=MediaType.AUDIO, storage_path="")
    db.add(source)
    db.flush()  # obtain source.id
    job = Job(source_id=source.id, status=JobStatus.PROCESSING)
    db.add(job)
    db.commit()
    # Store IDs on the session for later use
    session.source_id = source.id
    session.job_id = job.id

    # Gửi source_id và job_id về frontend để điều hướng sau khi ghi xong
    await websocket.send_json({
        "type": "session_init",
        "source_id": source.id,
        "job_id": job.id,
    })

    receiver = AudioReceiver(session)

    pipeline = getattr(
        websocket.app.state,
        "realtime_pipeline",
        None,
    )

    asr_service = getattr(
        websocket.app.state,
        "realtime_asr_service",
        None,
    )

    if pipeline is None or asr_service is None:
        logger.error(
            "Realtime pipeline not loaded",
        )
        await websocket.close(code=1013)
        return

    logger.info(
        "Realtime VAD: %s",
        type(pipeline.vad).__name__,
    )

    # ?window=N → ASR window length (5–30 s). Shorter windows confirm the
    # browser's live captions sooner.
    try:
        session.max_window_seconds = min(30.0, max(5.0, float(websocket.query_params.get("window", 30))))
    except ValueError:
        pass

    window_builder = SegmentWindowBuilder(session)

    # Boot a serverless worker now so the first window doesn't pay cold start.
    warmup_task = asyncio.create_task(_warmup(asr_service))  # keep a ref (avoid GC)

    asr_worker = ASRWorker(
        session,
        asr_service,
    )

    session.asr_task = asyncio.create_task(
        asr_worker.run()
    )

    # Sentence-level realtime: partial (live word-by-word) ASR is disabled on
    # production. It would call RunPod every ~1s (costly + ~1-2s network latency
    # each, so not truly live). Live words come from browser Web Speech instead.

    worker = AudioWorker(
        session,
        pipeline,
        window_builder,
    )

    worker_task = asyncio.create_task(
        worker.run()
    )

    session.worker_task = worker_task

    try:

        while True:

            # Dùng receive() generic thay vì receive_bytes(): client có thể gửi
            # text (control) hoặc close frame — receive_bytes() sẽ KeyError 'bytes'.
            message = await websocket.receive()

            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                break

            audio = message.get("bytes")
            if audio is None:
                # Text/control message: {"type":"auth"} links the session to the
                # logged-in user (so it shows in their library); {"type":"stop"}
                # finalizes.
                text = message.get("text")
                if text:
                    try:
                        import json
                        control = json.loads(text)
                        if control.get("type") == "auth":
                            _assign_owner(session.source_id, control.get("token"))
                        elif control.get("type") == "stop":
                            await websocket.send_json({"type": "stream_stopping"})
                            break
                    except Exception:
                        logger.warning("Bad realtime control message", exc_info=True)
                continue

            # Lưu toàn bộ audio của phiên realtime
            session.audio_archive.append(audio)

            # Xử lý realtime như hiện tại
            await receiver.receive(audio)

    except WebSocketDisconnect:

        logger.info(
            "Client disconnected",
        )

    except Exception:

        logger.exception(
            "Realtime stream crashed",
        )

    finally:

        # ===================================================
        # Flush phần audio còn lại
        # ===================================================

        try:

            await receiver.flush()

            # Chờ AudioWorker xử lý hết chunk, rồi đẩy nốt phần audio còn kẹt
            # (pending_audio < 2 segment + buffer window cuối) vào ASR —
            # trước đây tối đa 30 s audio cuối không bao giờ được transcribe.
            await asyncio.wait_for(session.audio_queue.join(), timeout=5.0)
            if session.pending_audio.size:
                session.ready_segments.append(session.pending_audio.copy())
                session.pending_audio = np.empty(0, dtype=np.float32)
            await window_builder.process()
            await window_builder.flush()

            # Chờ transcribe nốt các window còn trong hàng đợi (tối đa ~20s)
            # rồi báo frontend đã hoàn tất để nó đóng kết nối sạch sẽ.
            try:
                await asyncio.wait_for(session.asr_queue.join(), timeout=20.0)
            except (asyncio.TimeoutError, Exception):
                pass

        except Exception:

            logger.exception(
                "Audio flush failed",
            )

        # Báo frontend realtime đã hoàn tất (frontend chờ event này để kết thúc)
        # rồi đóng WS đúng chuẩn: trước đây handler kết thúc mà không gửi close
        # frame → trình duyệt thấy mã 1006 và báo lỗi dù mọi thứ đã xong.
        try:
            await websocket.send_json({"type": "stream_stopped"})
            await websocket.close(code=1000)
        except Exception:
            pass

        # ===================================================
        # Save PCM audio to Cloudflare R2 and queue offline ASR
        # ===================================================
        try:
            pcm_bytes = session.audio_archive.get_pcm_bytes()
            if len(pcm_bytes) > 0:
                wav_bytes = pcm_to_wav_bytes(pcm_bytes)
                
                # Fetch storage backend
                from meetasr.backend.api.routes import sources
                storage = sources.get_storage_backend()
                filename = f"realtime_{session.job_id}.wav"
                
                # Upload WAV to Cloudflare R2
                storage_key = await storage.save(wav_bytes, filename)
                
                # Update DB Source path and Job status. If the offline ASR service
                # is available, start the job right away (RunPod worker is still
                # warm from the realtime windows); otherwise leave it QUEUED so the
                # job-events WebSocket triggers it on demand.
                offline_asr = getattr(websocket.app.state, "asr_service", None)
                with Session(engine) as session_db:
                    db_source = session_db.get(Source, session.source_id)
                    db_job = session_db.get(Job, session.job_id)
                    if db_source:
                        db_source.storage_path = storage_key
                        # Counted in the user's storage usage like uploads.
                        db_source.file_size_bytes = len(wav_bytes)
                        session_db.add(db_source)
                    if db_job:
                        # (JobStage has no QUEUED — that AttributeError used to
                        # abort this commit, so realtime jobs never got processed.)
                        db_job.status = JobStatus.PROCESSING if offline_asr else JobStatus.QUEUED
                        db_job.progress = 0.0
                        session_db.add(db_job)
                    session_db.commit()
                    
                if offline_asr is not None:
                    from meetasr.backend.realtime.job_worker import run_job_processing
                    task = asyncio.create_task(
                        run_job_processing(session.job_id, offline_asr, storage)
                    )
                    _background_tasks.add(task)
                    task.add_done_callback(_background_tasks.discard)

                logger.info(
                    "Offline transcription audio saved to storage key=%s, job %s",
                    storage_key,
                    "started" if offline_asr else "queued",
                )
        except Exception:
            logger.exception("Failed to save audio archive and queue offline job")

        # ===================================================
        # Cleanup session
        # ===================================================

        try:

            await session.close()

        except Exception:

            logger.exception(
                "Session close failed",
            )

        # ===================================================
        # Shutdown realtime worker
        # ===================================================

        if worker_task:

            worker_task.cancel()

            try:

                await worker_task

            except asyncio.CancelledError:

                pass

            except Exception:

                logger.exception(
                    "Worker shutdown failed",
                )

        logger.info(
            "WS cleanup done",
        )