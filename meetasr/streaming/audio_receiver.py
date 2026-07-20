from __future__ import annotations

from collections import deque


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

    async def receive(self, audio: bytes) -> None:
        """
        Được gọi mỗi lần websocket nhận một frame.
        """

        # lưu vào ring buffer nếu cần replay/debug
        self.session.audio_buffer.append(audio)

        # ghép vào buffer
        self._buffer.extend(audio)

        # nếu đủ dữ liệu thì cắt thành chunk
        while len(self._buffer) >= self.chunk_size:

            chunk = bytes(self._buffer[: self.chunk_size])

            del self._buffer[: self.chunk_size]

            await self.session.audio_queue.put(chunk)

    async def flush(self) -> None:
        """
        Đẩy phần dữ liệu còn dư vào queue.
        Gọi khi websocket đóng.
        """

        if self._buffer:

            await self.session.audio_queue.put(bytes(self._buffer))

            self._buffer.clear()

    def reset(self) -> None:
        """
        Xóa toàn bộ dữ liệu đang giữ.
        """

        self._buffer.clear()