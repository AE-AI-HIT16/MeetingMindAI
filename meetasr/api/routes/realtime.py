from meetasr.streaming.session import StreamSession
from meetasr.streaming.worker import AudioWorker
from meetasr.streaming.audio_receiver import AudioReceiver

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
    worker = AudioWorker(session, pipeline)

    session.worker_task = asyncio.create_task(worker.run())

    try:
        while True:
            audio = await websocket.receive_bytes()
            await receiver.receive(audio)

    except WebSocketDisconnect:
        logger.info("Client disconnected")

    except Exception:
        logger.exception("Realtime stream crashed")

    finally:
        await session.close()