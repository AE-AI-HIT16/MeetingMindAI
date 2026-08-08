import asyncio
import logging
import os
import json
import numpy as np
from typing import Any
import httpx

from meetasr.backend.api.schemas_phase2 import TranscriptSegmentPayload, SentenceInfo, Segment, SpeakerTurn
from meetasr.backend.utils.audio import load_audio

logger = logging.getLogger(__name__)

class ASRServiceResult:
    """Normalized result consumed by both live and upload workers."""
    def __init__(self, segments: list[TranscriptSegmentPayload], text: str, duration_ms: int):
        self.segments = segments
        self.text = text
        self.duration_ms = duration_ms

class PreparedTranscription:
    """Audio decoded once and its full-file VAD timeline."""
    def __init__(self, audio: np.ndarray, vad_segments: list[Segment], duration_ms: int, speaker_turns: list[SpeakerTurn] | None = None):
        self.audio = audio
        self.vad_segments = vad_segments
        self.duration_ms = duration_ms
        self.speaker_turns = speaker_turns

def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Helper to convert audio numpy array to WAV bytes."""
    import io
    import scipy.io.wavfile as wavfile
    # Convert float32 to int16 if needed
    if audio.dtype != np.int16:
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    else:
        audio_int16 = audio
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)
    return buf.getvalue()

class ASRService:
    """Run one shared pipeline via RunPod HTTP API and normalize its result."""

    def __init__(self, runpod_url: str | None = None, api_key: str | None = None) -> None:
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
        self._transcribe_lock = asyncio.Lock()

    def _ensure_local_port_8001(self) -> None:
        """Auto-spawn GPU ML Engine on port 8001 if calling localhost and port 8001 is closed."""
        if os.path.exists("/.dockerenv"):
            return
        if "localhost:8001" in self.runpod_url or "127.0.0.1:8001" in self.runpod_url:
            try:
                import socket
                import subprocess
                import sys
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    if s.connect_ex(("127.0.0.1", 8001)) != 0:
                        subprocess.Popen(
                            [sys.executable, "-m", "uvicorn", "meetasr.runpod.app:app", "--host", "0.0.0.0", "--port", "8001"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
            except Exception:
                pass

    async def _post_with_retry(
        self,
        url: str,
        files: Any = None,
        data: Any = None,
        json_payload: Any = None,
        timeout: float = 600.0,
        max_retries: int = 8,
        initial_delay: float = 1.0,
    ) -> httpx.Response:
        self._ensure_local_port_8001()
        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout, headers=self.headers) as client:
                    if json_payload is not None:
                        response = await client.post(url, json=json_payload)
                    else:
                        response = await client.post(url, files=files, data=data)

                    if response.status_code in (502, 503, 504) and attempt < max_retries:
                        logger.warning(
                            "ASR service at %s returned HTTP %s (attempt %d/%d). Retrying in %.1fs...",
                            url, response.status_code, attempt, max_retries, delay
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 15.0)
                        continue

                    response.raise_for_status()
                    return response
            except (httpx.ConnectError, httpx.NetworkError, httpx.TimeoutException) as exc:
                if attempt == max_retries:
                    logger.error("ASR service connection failed after %d attempts: %s", max_retries, exc)
                    raise
                logger.warning(
                    "ASR service connection error to %s (attempt %d/%d): %s. Retrying in %.1fs...",
                    url, attempt, max_retries, exc, delay
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 15.0)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (502, 503, 504) and attempt < max_retries:
                    logger.warning(
                        "ASR service at %s returned HTTP %s (attempt %d/%d). Retrying in %.1fs...",
                        url, exc.response.status_code, attempt, max_retries, delay
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 15.0)
                else:
                    raise
        raise RuntimeError(f"Failed after {max_retries} attempts to call {url}")

    async def transcribe(
        self,
        audio_source: Any,
        *,
        offset_ms: int = 0,
        key: str | None = None,
    ) -> ASRServiceResult:
        # Prepare files dict with raw bytes for safe retries
        if isinstance(audio_source, str) and os.path.exists(audio_source):
            with open(audio_source, "rb") as f:
                file_bytes = f.read()
            files = {"file": (os.path.basename(audio_source), file_bytes)}
        else:
            audio_arr = load_audio(audio_source)
            wav_bytes = numpy_to_wav_bytes(audio_arr)
            files = {"file": ("audio.wav", wav_bytes)}

        async with self._transcribe_lock:
            data = {"key": key} if key else {}
            response = await self._post_with_retry(
                f"{self.runpod_url}/v1/transcribe",
                files=files,
                data=data,
                timeout=600.0,
            )
            res_json = response.json()

        # Parse output
        sentence_info = res_json.get("sentence_info", [])
        segments = [
            TranscriptSegmentPayload(
                start_ms=offset_ms + int(s["start"] * 1000),
                end_ms=offset_ms + int(s["end"] * 1000),
                speaker=s.get("speaker"),
                text=s["text"],
            )
            for s in sentence_info
            if s["text"].strip()
        ]

        duration_ms = max(0, int(res_json.get("duration", 0) * 1000))
        if not segments and res_json.get("text", "").strip():
            segments.append(
                TranscriptSegmentPayload(
                    start_ms=offset_ms,
                    end_ms=offset_ms + duration_ms,
                    speaker=None,
                    text=res_json["text"].strip(),
                )
            )

        return ASRServiceResult(
            segments=segments,
            text=res_json.get("text", ""),
            duration_ms=duration_ms,
        )

    async def prepare_incremental(
        self,
        audio_source: Any,
    ) -> PreparedTranscription:
        """Decode audio locally first, and call RunPod to run VAD/Diarization."""
        audio = load_audio(audio_source)
        wav_bytes = numpy_to_wav_bytes(audio)
        files = {"file": ("audio.wav", wav_bytes)}

        async with self._transcribe_lock:
            response = await self._post_with_retry(
                f"{self.runpod_url}/v1/prepare_incremental",
                files=files,
                timeout=600.0,
            )
            res_json = response.json()

        vad_segments = [
            Segment(start_ms=s["start_ms"], end_ms=s["end_ms"])
            for s in res_json["vad_segments"]
        ]
        speaker_turns = None
        if res_json.get("speaker_turns") is not None:
            speaker_turns = [
                SpeakerTurn(
                    start_ms=s["start_ms"],
                    end_ms=s["end_ms"],
                    speaker=s["speaker"]
                )
                for s in res_json["speaker_turns"]
            ]

        return PreparedTranscription(
            audio=audio,
            vad_segments=vad_segments,
            duration_ms=res_json["duration_ms"],
            speaker_turns=speaker_turns,
        )

    async def transcribe_segment(
        self,
        prepared: PreparedTranscription,
        segment: Segment,
        *,
        language: str = "auto",
        key: str | None = None,
    ) -> list[SentenceInfo]:
        """Slice local audio buffer and call RunPod to transcribe segment."""
        start = int(segment.start_ms / 1000 * 16000)
        end = int(segment.end_ms / 1000 * 16000)
        chunk = prepared.audio[start:end]
        if len(chunk) == 0:
            return []

        wav_bytes = numpy_to_wav_bytes(chunk)
        files = {"file": ("chunk.wav", wav_bytes)}
        data = {"language": language}
        if key:
            data["key"] = key

        async with self._transcribe_lock:
            response = await self._post_with_retry(
                f"{self.runpod_url}/v1/transcribe_segment",
                files=files,
                data=data,
                timeout=120.0,
            )
            res_json = response.json()

        return [
            SentenceInfo(
                text=s["text"],
                start=s["start"],
                end=s["end"],
                speaker=s.get("speaker"),
                char_timestamps=s.get("char_timestamps", []),
            )
            for s in res_json
        ]

    async def finalize_incremental(
        self,
        prepared: PreparedTranscription,
        sentences: list[SentenceInfo],
    ) -> list[SentenceInfo]:
        """Call RunPod to run final ASR-first diarization or preassigned turn grouping."""
        sentences_data = [
            {
                "text": s.text,
                "start": s.start,
                "end": s.end,
                "speaker": s.speaker,
                "char_timestamps": s.char_timestamps,
            }
            for s in sentences
        ]

        async with self._transcribe_lock:
            if prepared.speaker_turns is not None:
                payload = {"sentences": sentences_data}
                response = await self._post_with_retry(
                    f"{self.runpod_url}/v1/finalize_incremental",
                    json_payload=payload,
                    timeout=300.0,
                )
            else:
                wav_bytes = numpy_to_wav_bytes(prepared.audio)
                files = {"file": ("audio.wav", wav_bytes)}
                vad_data = [
                    {"start_ms": v.start_ms, "end_ms": v.end_ms}
                    for v in prepared.vad_segments
                ]
                data = {
                    "sentences_json": json.dumps(sentences_data),
                    "vad_segments_json": json.dumps(vad_data)
                }
                response = await self._post_with_retry(
                    f"{self.runpod_url}/v1/finalize_incremental",
                    files=files,
                    data=data,
                    timeout=300.0,
                )

            res_json = response.json()

        return [
            SentenceInfo(
                text=s["text"],
                start=s["start"],
                end=s["end"],
                speaker=s.get("speaker"),
                char_timestamps=s.get("char_timestamps", []),
            )
            for s in res_json
        ]

    async def recognize_partial(self, audio: np.ndarray) -> list[dict]:
        """Recognize partial audio chunk for realtime streaming."""
        wav_bytes = numpy_to_wav_bytes(audio)
        files = {"file": ("chunk.wav", wav_bytes)}
        async with self._transcribe_lock:
            response = await self._post_with_retry(
                f"{self.runpod_url}/v1/realtime/recognize",
                files=files,
                timeout=10.0,
                max_retries=3,
            )
            return response.json()

