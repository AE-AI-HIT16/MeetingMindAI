from __future__ import annotations

import asyncio
from typing import Any
import httpx
import os
import numpy as np

from meetasr.backend.api.schemas_phase2 import TranscriptSegmentPayload
from meetasr.backend.services.asr_service import ASRService, ASRServiceResult, numpy_to_wav_bytes

class RealtimeASRService:
    """
    Adapter for realtime ASR running on RunPod.

    On RunPod Serverless (api.runpod.ai) it speaks /runsync with the
    "realtime_transcribe" action (base64 audio). On a local HTTP ASR server it
    keeps the old multipart /v1/realtime/* endpoints.
    """

    def __init__(self, runpod_url: str | None = None, api_key: str | None = None):
        if not runpod_url:
            endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
            base_url = os.environ.get("RUNPOD_API_BASE_URL", "https://api.runpod.ai/v2").rstrip("/")
            if endpoint_id:
                runpod_url = f"{base_url}/{endpoint_id}"
            else:
                runpod_url = os.environ.get("RUNPOD_URL", "http://localhost:8001")
        self.runpod_url = runpod_url.rstrip("/")
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._lock = asyncio.Lock()
        self._is_serverless = "api.runpod.ai" in self.runpod_url
        # Reuse ASRService's /runsync caller (handles IN_QUEUE/IN_PROGRESS polling)
        self._serverless = ASRService(runpod_url=self.runpod_url, api_key=self.api_key)

    async def warmup(self) -> None:
        """Queue a no-op job (async /run, not awaited) so a serverless worker
        boots and loads models while the user is already seeing browser captions.
        Old images without a "warmup" action still load models in init_models()."""
        if not self._is_serverless:
            return
        async with httpx.AsyncClient(timeout=10.0, headers=self.headers) as client:
            response = await client.post(
                f"{self.runpod_url}/run",
                json={"input": {"action": "warmup"}},
            )
            response.raise_for_status()

    async def transcribe(
        self,
        audio: Any,
        *,
        offset_ms: int = 0,
        key: str | None = None,
    ) -> ASRServiceResult:

        if self._is_serverless:
            import base64
            payload: dict = {
                "audio_base64": base64.b64encode(numpy_to_wav_bytes(audio)).decode(),
                "language": os.environ.get("ASR_LANGUAGE", "vi"),
            }
            if key:
                payload["key"] = key
            res_json = await self._serverless._call_serverless(
                "realtime_transcribe", payload, timeout=120.0,
            )
        else:
            wav_bytes = numpy_to_wav_bytes(audio)
            files = {"file": ("audio.wav", wav_bytes)}
            data = {}
            if key:
                data["key"] = key
            async with self._lock:
                async with httpx.AsyncClient(timeout=60.0, headers=self.headers) as client:
                    response = await client.post(
                        f"{self.runpod_url}/v1/realtime/transcribe",
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    res_json = response.json()

        # Parse sentences
        sentence_info = res_json.get("sentence_info", [])
        segments = []
        for s in sentence_info:
            if not s["text"].strip():
                continue

            segments.append(
                TranscriptSegmentPayload(
                    start_ms=offset_ms + int(s["start"] * 1000),
                    end_ms=offset_ms + int(s["end"] * 1000),
                    speaker=s.get("speaker"),
                    text=s["text"].strip(),
                )
            )

        duration_ms = max(
            0,
            int(res_json.get("duration", 0) * 1000),
        )

        return ASRServiceResult(
            segments=segments,
            text=res_json.get("text", ""),
            duration_ms=duration_ms,
        )

    async def recognize_partial(self, audio: np.ndarray) -> list[dict]:
        """Recognize partial audio chunk for realtime streaming."""
        wav_bytes = numpy_to_wav_bytes(audio)
        files = {"file": ("chunk.wav", wav_bytes)}
        async with self._lock:
            async with httpx.AsyncClient(timeout=10.0, headers=self.headers) as client:
                response = await client.post(
                    f"{self.runpod_url}/v1/realtime/recognize",
                    files=files,
                )
                response.raise_for_status()
                return response.json()