from meetasr.streaming.session import StreamSession
from meetasr.streaming.worker import AudioWorker
from meetasr.streaming.audio_receiver import AudioReceiver
from meetasr.streaming.asr_worker import ASRWorker
from meetasr.streaming.window_builder import SegmentWindowBuilder

import asyncio
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Tạo router để đăng ký endpoint WebSocket
router = APIRouter()

# Logger dùng để ghi log của module này
logger = logging.getLogger("realtime")


@router.websocket("/v1/realtime/stream")
async def realtime_stream(websocket: WebSocket):
    """
    Endpoint WebSocket nhận audio realtime từ frontend.
    Mỗi client kết nối sẽ tạo ra một session độc lập.
    """

    # Chấp nhận kết nối WebSocket
    await websocket.accept()

    # Sinh một session id ngắn để dễ theo dõi log
    session_id = uuid.uuid4().hex[:8]

    print("WS accepted", session_id, websocket.client)

    # Tạo đối tượng lưu trạng thái của phiên làm việc
    session = StreamSession(websocket)

    # Đối tượng nhận audio từ client
    receiver = AudioReceiver(session)

    # Lấy pipeline ASR đã khởi tạo khi server start
    pipeline = websocket.app.state.pipeline
    print (type(pipeline.vad))

    # Nếu pipeline chưa load thì đóng kết nối
    if pipeline is None:
        print("Pipeline not loaded", session_id)
        await websocket.close(code=1013)
        return

    # Ghép các chunk audio thành từng cửa sổ (window)
    # để đưa sang ASR
    window_builder = SegmentWindowBuilder(session)

    # Worker xử lý nhận kết quả ASR
    asr_worker = ASRWorker(session, pipeline)

    # Chạy ASR worker ở background
    session.asr_task = asyncio.create_task(
        asr_worker.run()
    )

    # Worker xử lý audio:
    # Queue audio -> tạo segment -> đưa sang pipeline
    worker = AudioWorker(
        session,
        pipeline,
        window_builder
    )

    # Chạy worker ở background
    worker_task = asyncio.create_task(
        worker.run()
    )

    # Lưu task để session có thể quản lý
    session.worker_task = worker_task

    try:

        # Vòng lặp chính
        # Liên tục nhận audio từ frontend
        while True:

            # Chờ client gửi dữ liệu audio dạng bytes
            audio = await websocket.receive_bytes()

            # Đưa audio cho AudioReceiver xử lý
            await receiver.receive(audio)

    # Client đóng WebSocket
    except WebSocketDisconnect:

        print("Client disconnected", session_id)

    # Các lỗi ngoài dự kiến
    except Exception:

        logger.exception(
            "Realtime stream crashed session=%s",
            session_id,
        )

    finally:

        # ===================================================
        # Cleanup
        # ===================================================

        # Đẩy phần audio còn sót trong buffer
        # để không bị mất đoạn cuối
        try:

            await receiver.flush()

        except Exception:

            logger.exception(
                "Audio flush failed session=%s",
                session_id,
            )

        # Đóng session
        # Giải phóng queue, websocket,...
        try:

            await session.close()

        except Exception:

            logger.exception(
                "Session close failed session=%s",
                session_id,
            )

        # Dừng worker background
        if worker_task:

            # Gửi tín hiệu cancel
            worker_task.cancel()

            try:

                # Đợi worker kết thúc
                await worker_task

            # Đây là trường hợp bình thường
            except asyncio.CancelledError:
                pass

            # Nếu worker lỗi khi shutdown
            except Exception:

                logger.exception(
                    "Worker shutdown failed session=%s",
                    session_id,
                )

        # Kết thúc quá trình cleanup
        logger.info("WS cleanup done session=%s", session_id)