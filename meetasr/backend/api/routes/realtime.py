from meetasr.backend.streaming.session import StreamSession
from meetasr.backend.streaming.worker import AudioWorker
from meetasr.backend.streaming.audio_receiver import AudioReceiver
from meetasr.backend.streaming.asr_worker import ASRWorker
from meetasr.backend.streaming.temp_asr_woker import TempASRWorker
from meetasr.backend.streaming.partial_buffer_cleaner import PartialBufferCleaner
from meetasr.backend.streaming.window_builder import SegmentWindowBuilder
import io
import wave
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlmodel import Session
from meetasr.backend.db.connection import get_db, engine
from meetasr.backend.db.models_phase2 import Source, MediaType, Job, JobStatus, JobStage

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

    window_builder = SegmentWindowBuilder(session)

    asr_worker = ASRWorker(
        session,
        asr_service,
    )

    session.asr_task = asyncio.create_task(
        asr_worker.run()
    )

    temp_asr_worker = TempASRWorker(
        session,
        pipeline,
    )

    session.temp_asr_task = asyncio.create_task(
        temp_asr_worker.run()
    )

    partial_buffer_cleaner = PartialBufferCleaner(
        session,
    )

    session.partial_cleaner_task = asyncio.create_task(
        partial_buffer_cleaner.run()
    )

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

            audio = await websocket.receive_bytes()

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

        except Exception:

            logger.exception(
                "Audio flush failed",
            )

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
                
                # Update DB Source path and Job status to QUEUED
                with Session(engine) as session_db:
                    db_source = session_db.get(Source, session.source_id)
                    db_job = session_db.get(Job, session.job_id)
                    if db_source:
                        db_source.storage_path = storage_key
                        session_db.add(db_source)
                    if db_job:
                        db_job.status = JobStatus.QUEUED
                        db_job.stage = JobStage.QUEUED
                        db_job.progress = 0.0
                        session_db.add(db_job)
                    session_db.commit()
                    
                logger.info(
                    "Offline transcription audio saved to storage key=%s, job queued",
                    storage_key,
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