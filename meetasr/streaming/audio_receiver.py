from __future__ import annotations

from meetasr.streaming.validate import validate_audio


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
        chunk_size: int = 3200,  # ~0.1 giây PCM16 mono 16kHz
    ):
        self.session = session
        self.chunk_size = chunk_size
        self._buffer = bytearray()

    async def receive(self, audio: bytes) -> None:
        """
        Được gọi mỗi lần nhận một frame audio từ WebSocket.
        """

        audio = validate_audio(audio)

        # Ghép frame vào buffer (ring buffer float32 được worker ghi)
        self._buffer.extend(audio)

        # Cắt thành các chunk cố định
        while len(self._buffer) >= self.chunk_size:
            chunk = bytes(self._buffer[: self.chunk_size])
            del self._buffer[: self.chunk_size]

            accepted = await self.session.audio_queue.put(chunk)
            if accepted is False:
                coverage = getattr(self.session, "coverage", None)
                if coverage is not None:
                    coverage.record_audio_drop()

    async def flush(self) -> None:
        """
        Đẩy phần dữ liệu còn lại vào queue khi kết thúc phiên.
        """

        if self._buffer:
            accepted = await self.session.audio_queue.put(bytes(self._buffer))
            if accepted is False:
                coverage = getattr(self.session, "coverage", None)
                if coverage is not None:
                    coverage.record_audio_drop()
            self._buffer.clear()

    def reset(self) -> None:
        """
        Xóa toàn bộ dữ liệu đang giữ.
        """

        self._buffer.clear()
