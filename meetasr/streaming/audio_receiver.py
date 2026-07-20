from __future__ import annotations
from meetasr.streaming.normalize import normalize_audio

import logging
import numpy as np

logger = logging.getLogger("meetasr.streaming.receiver")


class AudioReceiver:
    """
    Nhận byte audio từ WebSocket.

    Chức năng:
    - Gom các frame nhỏ thành một chunk lớn.
    - Khi đủ kích thước thì đẩy vào session.audio_queue.
    """

    def __init__(
        self,
        session,
        *,
        chunk_size: int = 32000,  # ví dụ ~1 giây PCM16kHz mono
    ):
        self.session = session

        self.chunk_size = chunk_size

        # buffer chứa byte đang ghép
        self._buffer = bytearray()

        logger.info("=" * 80)
        logger.info("AudioReceiver initialized")
        logger.info("Chunk size           : %d bytes", self.chunk_size)
        logger.info("Buffer initialized   : %d bytes", len(self._buffer))

    async def receive(self, audio: bytes) -> None:
        """
        Được gọi mỗi lần websocket nhận một frame.
        """

        logger.info("-" * 80)
        logger.info("Receive audio frame")
        logger.info("Input python type    : %s", type(audio).__name__)
        logger.info("Input size           : %d bytes", len(audio))
        logger.info("Buffer before        : %d bytes", len(self._buffer))
        logger.info("Queue size before    : %d", self.session.audio_queue.qsize())

        # Đây là giả định về dạng audio đầu vào, cần thay đổi sau khi đo thực tế
        logger.info(
            "Normalize config     : sample_rate=%d, channels=%d, sample_width=%d",
            48000,
            1,
            2,
        )

        audio = normalize_audio(
            audio,
            sample_rate=48000,
            channels=1,
            sample_width=2,
        )

        pcm = np.frombuffer(audio, dtype=np.int16)

        logger.info(
            "Normalized samples  : %d",
            pcm.size,
        )

        logger.info(
            "Normalized RMS      : %.2f",
            float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))),
        )

        logger.info(
            "Normalized first20  : %s",
            pcm[:20].tolist(),
        )

        logger.info("Output python type   : %s", type(audio).__name__)
        logger.info("Output size          : %d bytes", len(audio))

        # lưu vào ring buffer nếu cần replay/debug
        self.session.audio_buffer.append(audio)

        logger.info(
            "Ring buffer size     : %d",
            len(self.session.audio_buffer),
        )

        # ghép vào buffer
        self._buffer.extend(audio)

        logger.info(
            "Buffer after extend  : %d bytes",
            len(self._buffer),
        )

        # nếu đủ dữ liệu thì cắt thành chunk
        while len(self._buffer) >= self.chunk_size:

            logger.info(
                "Chunk condition met  : %d >= %d",
                len(self._buffer),
                self.chunk_size,
            )

            chunk = bytes(self._buffer[: self.chunk_size])

            logger.info(
                "Chunk created        : %d bytes",
                len(chunk),
            )

            del self._buffer[: self.chunk_size]

            logger.info(
                "Buffer after slice   : %d bytes",
                len(self._buffer),
            )

            await self.session.audio_queue.put(chunk)

            logger.info(
                "Chunk pushed to queue"
            )

            logger.info(
                "Queue size after     : %d",
                self.session.audio_queue.qsize(),
            )

        logger.info("Receive finished")

    async def flush(self) -> None:
        """
        Đẩy phần dữ liệu còn dư vào queue.
        Gọi khi websocket đóng.
        """

        logger.info("=" * 80)
        logger.info("Flush called")
        logger.info("Remaining buffer     : %d bytes", len(self._buffer))

        if self._buffer:

            await self.session.audio_queue.put(bytes(self._buffer))

            logger.info(
                "Remaining bytes pushed: %d bytes",
                len(self._buffer),
            )

            logger.info(
                "Queue size after     : %d",
                self.session.audio_queue.qsize(),
            )

            self._buffer.clear()

            logger.info("Buffer cleared")

        else:

            logger.info("No remaining data to flush")

    def reset(self) -> None:
        """
        Xóa toàn bộ dữ liệu đang giữ.
        """

        logger.info("=" * 80)
        logger.info("Receiver reset")
        logger.info("Buffer before reset  : %d bytes", len(self._buffer))

        self._buffer.clear()

        logger.info("Buffer after reset   : %d bytes", len(self._buffer))