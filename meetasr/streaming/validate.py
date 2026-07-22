from __future__ import annotations


def validate_audio(audio: bytes) -> bytes:
    """
    Validate audio contract.

    Frontend phải gửi:
    - PCM16
    - mono
    - 16kHz
    - little-endian

    Backend không chuẩn hóa lại audio.
    """

    if not isinstance(audio, (bytes, bytearray)):
        raise TypeError("Audio must be bytes.")

    if not audio:
        raise ValueError("Audio frame is empty.")

    # PCM16 = 2 bytes / sample
    if len(audio) % 2 != 0:
        raise ValueError(
            "Invalid PCM16 frame: byte length must be divisible by 2."
        )

    return bytes(audio)