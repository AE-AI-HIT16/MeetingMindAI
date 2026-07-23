from meetasr.streaming.session import StreamSession
from meetasr.streaming.worker import AudioWorker
from meetasr.streaming.audio_receiver import AudioReceiver
from meetasr.streaming.asr_worker import ASRWorker
from meetasr.streaming.window_builder import SegmentWindowBuilder

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


router = APIRouter()
logger = logging.getLogger("realtime")


@router.websocket("/v1/realtime/stream")
async def realtime_stream(websocket: WebSocket):
    """
    WebSocket nhận audio realtime từ frontend.
    """

    await websocket.accept()

    session = StreamSession(websocket)
    receiver = AudioReceiver(session)

    pipeline = websocket.app.state.pipeline

    window_builder = SegmentWindowBuilder(session)

    asr_worker = ASRWorker(session,pipeline,)

    session.asr_task = asyncio.create_task(
        asr_worker.run()
    )

    worker = AudioWorker(
        session,
        pipeline,
        window_builder
    )

    worker_task = asyncio.create_task(
        worker.run()
    )

    session.worker_task = worker_task

    try:
        while True:

            audio = await websocket.receive_bytes()

            await receiver.receive(audio)

    except WebSocketDisconnect:
        logger.info(
            "Client disconnected"
        )

    except Exception:
        logger.exception(
            "Realtime stream crashed"
        )

    finally:

        # Đẩy phần audio còn dư
        try:
            await receiver.flush()
        except Exception:
            logger.exception(
                "Audio flush failed"
            )


        # Đóng session
        try:
            await session.close()
        except Exception:
            logger.exception(
                "Session close failed"
            )


        # Dừng worker
        if worker_task:

            worker_task.cancel()

            try:
                await worker_task

            except asyncio.CancelledError:
                pass

            except Exception:
                logger.exception(
                    "Worker shutdown failed"
                )