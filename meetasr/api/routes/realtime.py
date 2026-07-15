import asyncio

from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from meetasr.streaming.session import StreamSession
from meetasr.streaming.worker import AudioWorker
from meetasr.streaming.processor.dummy_processor import DummyProcessor


router = APIRouter()


@router.websocket("/v1/realtime/stream")
async def realtime_stream(websocket: WebSocket):

    await websocket.accept()

    # Lấy nơi chứ các model đã được khởi tạo
    manager = websocket.app.state.model_manager

    # Khởi tạo session
    session = StreamSession(websocket)

    # Khởi tạo hành động đếm byte và truyền vào model đã được khởi tạo trước đó
    processor = DummyProcessor(manager)

    # Thêm vào worker
    worker = AudioWorker(session, processor)

    # Lưu worker vào session đồng thời khởi chạy các process trong worker
    session.worker_task = asyncio.create_task(
        worker.run()
    )

    try:

        while True:

            audio = await websocket.receive_bytes()

            # lưu ring buffer
            session.audio_buffer.append(audio)

            # đưa vào queue
            await session.audio_queue.put(audio)

    except WebSocketDisconnect:

        session.worker_task.cancel()
        # Xóa dữ liệu của session khi ngắt kết nối đến websocket
        await session.close()