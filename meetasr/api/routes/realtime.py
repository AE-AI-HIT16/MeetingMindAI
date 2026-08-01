from meetasr.streaming.session import StreamSession
from meetasr.streaming.worker import AudioWorker
from meetasr.streaming.audio_receiver import AudioReceiver
from meetasr.streaming.asr_worker import ASRWorker
from meetasr.streaming.temp_asr_woker import TempASRWorker
from meetasr.streaming.partial_buffer_cleaner import PartialBufferCleaner
from meetasr.streaming.window_builder import SegmentWindowBuilder
from meetasr.streaming.final_transcript_queue import FinalTranscriptJob

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

logger = logging.getLogger("realtime")


@router.websocket("/v1/realtime/stream")
async def realtime_stream(websocket: WebSocket):
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
        # Giao audio cho worker xử lý offline
        # ===================================================

        try:

            audio = session.audio_archive.get_numpy()

            if len(audio) > 0:

                final_transcript_queue = getattr(
                    websocket.app.state,
                    "final_transcript_queue",
                    None,
                )

                if final_transcript_queue is None:

                    logger.error(
                        "Final transcript queue not initialized ",
                    )

                else:

                    await final_transcript_queue.put(
                        FinalTranscriptJob(
                            audio=audio
                        )
                    )

                    logger.info(
                        "Offline transcription queued "
                        "samples=%d",
                        len(audio),
                    )

        except Exception:

            logger.exception(
                "Failed to enqueue offline job",
            )

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