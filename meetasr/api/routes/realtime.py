from meetasr.streaming.session import StreamSession
from meetasr.streaming.worker import AudioWorker
from meetasr.streaming.audio_receiver import AudioReceiver

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect


router = APIRouter()

@router.websocket("/v1/realtime/stream")
async def realtime_stream(websocket: WebSocket):
    await websocket.accept()

    manager = websocket.app.state.models

    session = StreamSession(websocket)

    receiver = AudioReceiver(session)

    processor = DummyProcessor(manager)

    worker = AudioWorker(session, processor)

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