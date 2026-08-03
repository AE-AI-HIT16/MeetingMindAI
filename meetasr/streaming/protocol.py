"""WebSocket input contract for realtime binary audio and control frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from fastapi import WebSocket, WebSocketDisconnect


@dataclass(frozen=True, slots=True)
class RealtimeAuthFrame:
    """Optional internal JWT sent before realtime audio starts."""

    token: str | None


async def receive_stream_frame(
    websocket: WebSocket,
) -> bytes | Literal["stop"] | RealtimeAuthFrame:
    """Receive one binary audio frame or the supported JSON stop command."""
    message = await websocket.receive()
    message_type = message.get("type")

    if message_type == "websocket.disconnect":
        raise WebSocketDisconnect(
            code=message.get("code", 1000),
            reason=message.get("reason", ""),
        )

    audio = message.get("bytes")
    if audio is not None:
        return audio

    text = message.get("text")
    if text is not None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Realtime control frame must be valid JSON.") from exc
        if isinstance(payload, dict) and payload.get("type") == "stop":
            return "stop"
        if isinstance(payload, dict) and payload.get("type") == "auth":
            token = payload.get("token")
            if token is not None and not isinstance(token, str):
                raise ValueError("Realtime auth token must be a string or null.")
            return RealtimeAuthFrame(token=token)

    raise ValueError("Unsupported realtime WebSocket frame.")
