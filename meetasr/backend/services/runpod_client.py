
from __future__ import annotations
import os
import asyncio
from typing import Any, Dict
import httpx

class RunPodClient:
    """Simple HTTP client to call RunPod model APIs with exponential backoff.
    Uses environment variables:
        RUNPOD_URL — base URL of the RunPod server.
        RUNPOD_ENDPOINT_ID — Endpoint ID (e.g. ngbl879qdgxbkx)
        RUNPOD_API_BASE_URL — Base API URL (e.g. https://api.runpod.ai/v2)
        RUNPOD_API_KEY — Bearer token for authentication.
    """

    def __init__(self, runpod_url: str | None = None, api_key: str | None = None) -> None:
        if not runpod_url:
            endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "")
            base_url = os.getenv("RUNPOD_API_BASE_URL", "https://api.runpod.ai/v2").rstrip("/")
            if endpoint_id:
                runpod_url = f"{base_url}/{endpoint_id}"
            else:
                runpod_url = os.getenv("RUNPOD_URL", "http://localhost:8001")
        self.base_url = runpod_url.rstrip("/")
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY", "")
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=self.headers, timeout=30.0)

    async def _request(self, method: str, endpoint: str, json_body: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Perform an HTTP request with up to 3 retries and exponential backoff.
        Raises an httpx.HTTPError if all attempts fail.
        """
        delays = [0.5, 1.0, 2.0]
        for attempt in range(3):
            try:
                response = await self.client.request(method, endpoint, json=json_body)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                if attempt == 2:
                    # All retries exhausted — raise a clear error for the caller.
                    raise RuntimeError(
                        f"RunPod request to {endpoint} failed after 3 attempts: {exc}"
                    ) from exc
                await asyncio.sleep(delays[attempt])
        # Unreachable
        raise RuntimeError("Unexpected error in RunpodClient._request")

    # Public API wrappers ---------------------------------------------------
    async def transcribe(self, audio_bytes: bytes) -> Dict[str, Any]:
        return await self._request("POST", "/v1/transcribe", {"audio": audio_bytes})

    async def prepare_incremental(self, audio_bytes: bytes) -> Dict[str, Any]:
        return await self._request("POST", "/v1/prepare_incremental", {"audio": audio_bytes})

    async def transcribe_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/v1/transcribe_segment", segment)

    async def finalize_incremental(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/v1/finalize_incremental", payload)

    async def close(self) -> None:
        await self.client.aclose()
