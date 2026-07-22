from __future__ import annotations

from meetasr.streaming.audio_validator import validate_audio


class AudioReceiver:
    """
    Nhận audio từ WebSocket.

    Trách nhiệm:
    - Validate dữ liệu đầu vào.
    - Gom nhiều frame thành chunk.
    - Đưa chunk vào session.audio_queue.
    """

    def __init__(
        self,
        session,
        *,
        chunk_size: int = 32000,  # ~1 giây PCM16 mono 16kHz
    ):
        self.session = session
        self.chunk_size = chunk_size
        self._buffer = bytearray()

    async def receive(self, audio: bytes) -> None:
        """
        Được gọi mỗi lần nhận một frame audio từ WebSocket.
        """

        audio = validate_audio(audio)

        # Lưu lại nếu cần replay/debug
        self.session.audio_buffer.append(audio)

        # Ghép frame vào buffer
        self._buffer.extend(audio)

        # Cắt thành các chunk cố định
        while len(self._buffer) >= self.chunk_size:
            chunk = bytes(self._buffer[: self.chunk_size])
            del self._buffer[: self.chunk_size]

            await self.session.audio_queue.put(chunk)

    async def flush(self) -> None:
        """
        Đẩy phần dữ liệu còn lại vào queue khi kết thúc phiên.
        """

        if self._buffer:
            await self.session.audio_queue.put(bytes(self._buffer))
            self._buffer.clear()

    def reset(self) -> None:
        """
        Xóa toàn bộ dữ liệu đang giữ.
        """

        self._buffer.clear()