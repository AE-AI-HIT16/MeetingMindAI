from meetasr.streaming.session import StreamSession
from meetasr.streaming.worker import AudioWorker
from meetasr.streaming.audio_receiver import AudioReceiver
from meetasr.auto.auto_pipeline import AutoPipeline

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect


router = APIRouter()

@router.websocket("/v1/realtime/stream")
async def realtime_stream(websocket: WebSocket):
    await websocket.accept()

    # Khởi tạo phiên
    session = StreamSession(websocket)

    # Khởi tạo nơi chunk dữ liệu
    receiver = AudioReceiver(session)

    # Khởi tạo luồng pipeline
    pipeline = websocket.app.state.pipeline

    worker = AudioWorker(session, pipeline)

    session.worker_task = asyncio.create_task(
        worker.run()
    )

    try:
        while True:

            audio = await websocket.receive_bytes()

            await receiver.receive(audio)

    except WebSocketDisconnect:
        pass

    finally:
        await session.close()