import numpy as np

class AudioArchive:

    def __init__(self):
        self._buffer = bytearray()

    def append(self, audio: bytes):
        self._buffer.extend(audio)

    def reset(self):
        self._buffer.clear()

    def get_pcm_bytes(self) -> bytes:
        return bytes(self._buffer)

    def get_numpy(self) -> np.ndarray:
        pcm = np.frombuffer(
            self._buffer,
            dtype=np.int16,
        ).astype(np.float32)

        pcm /= 32768.0

        return pcm.copy()