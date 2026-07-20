from meetasr.streaming.session import StreamSession
from meetasr.streaming.worker import AudioWorker
from meetasr.streaming.audio_receiver import AudioReceiver

import asyncio
import logging
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect


router = APIRouter()

logger = logging.getLogger("realtime")
print("logger level:", logger.level)
print("effective level:", logger.getEffectiveLevel())


@router.websocket("/v1/realtime/stream")
async def realtime_stream(websocket: WebSocket):

    logger.info("=" * 80)
    logger.info("New websocket connection")

    await websocket.accept()

    logger.info("Websocket accepted")

    # Khởi tạo phiên
    session = StreamSession(websocket)

    logger.info("StreamSession created")

    # Khởi tạo nơi chunk dữ liệu
    receiver = AudioReceiver(session)

    logger.info("AudioReceiver created")

    # Khởi tạo luồng pipeline
    pipeline = websocket.app.state.pipeline

    logger.info(
        "Pipeline loaded: %s",
        type(pipeline).__name__
    )

    worker = AudioWorker(session, pipeline)

    logger.info(
        "AudioWorker created: %s",
        type(worker).__name__
    )

    session.worker_task = asyncio.create_task(
        worker.run()
    )

    logger.info(
        "Worker task started"
    )

    frame_count = 0
    total_bytes = 0

    try:

        while True:

            audio = await websocket.receive_bytes()

            pcm = np.frombuffer(audio, dtype=np.int16)

            logger.info(
                "PCM samples         : %d",
                pcm.size,
            )

            if pcm.size:
                logger.info(
                    "PCM min/max        : %d / %d",
                    int(pcm.min()),
                    int(pcm.max()),
                )

                logger.info(
                    "PCM mean           : %.2f",
                    float(pcm.mean()),
                )

                logger.info(
                    "PCM first 20       : %s",
                    pcm[:20].tolist(),
                )

            logger.info(
                "Frame raw bytes     : %d",
                len(audio),
            )

            logger.info(
                "First 32 bytes(hex) : %s",
                audio[:32].hex(" "),
            )

            logger.info(
                "Last 32 bytes(hex)  : %s",
                audio[-32:].hex(" "),
            )

            frame_count += 1
            total_bytes += len(audio)

            logger.info("-" * 80)
            logger.info("Frame #%d", frame_count)

            logger.info(
                "Python type      : %s",
                type(audio).__name__
            )

            logger.info(
                "Frame size       : %d bytes",
                len(audio)
            )

            logger.info(
                "Total received   : %.2f KB",
                total_bytes / 1024
            )

            logger.info(
                "First 16 bytes   : %s",
                audio[:16]
            )

            await receiver.receive(audio)

            logger.info(
                "Receiver finished processing frame #%d",
                frame_count
            )

    except WebSocketDisconnect:

        logger.info("Client disconnected")

    except Exception:

        logger.exception("Realtime stream crashed")

    finally:

        logger.info("Closing session")

        await session.close()

        logger.info("Session closed")