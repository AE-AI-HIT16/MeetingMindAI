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
        self._is_serverless = "api.runpod.ai" in self.runpod_url

    async def _call_serverless(self, action: str, payload: dict, timeout: float = 600.0) -> dict:
        """Call RunPod Serverless /runsync and return the output dict."""
        url = f"{self.runpod_url}/runsync"
        body = {"input": {"action": action, **payload}}
        for attempt in range(3):
            async with httpx.AsyncClient(timeout=timeout, headers=self.headers) as client:
                response = await client.post(url, json=body)
            if response.status_code in (502, 503, 504):
                wait = 5 * (attempt + 1)
                logger.warning("RunPod %s returned %s, retry %d/3 in %ds", url, response.status_code, attempt + 1, wait)
                await asyncio.sleep(wait)
                continue
            if not response.is_success:
                logger.error("RunPod %s error: status=%s body=%s", url, response.status_code, response.text[:500])
            response.raise_for_status()
            result = response.json()
            status = result.get("status")
            if status == "FAILED":
                raise RuntimeError(f"RunPod job failed: {result.get('error', result)}")
            output = result.get("output", result)
            sentence_count = len(output.get("sentence_info", [])) if isinstance(output, dict) else "N/A"
            logger.info("RunPod /runsync response: status=%s action=%s sentence_info_count=%s", status, action, sentence_count)
            return output
        response.raise_for_status()

    def _ensure_local_port_8001(self) -> None:
        """Auto-spawn GPU ML Engine on port 8001 if calling outside Docker and port 8001 is closed."""
        if os.path.exists("/.dockerenv"):
            return
        try:
            import socket
            import subprocess
            import sys
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                if s.connect_ex(("127.0.0.1", 8001)) != 0:
                    logger.info("Port 8001 is closed. Auto-spawning local ASR model server on port 8001...")
                    subprocess.Popen(
                        [sys.executable, "-m", "uvicorn", "meetasr.runpod.app:app", "--host", "0.0.0.0", "--port", "8001"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
        except Exception as exc:
            logger.warning("Failed to auto-spawn local ASR server on port 8001: %s", exc)

    def _get_candidate_urls(self, target_url: str) -> list[str]:
        from urllib.parse import urlparse
        base = self.runpod_url.rstrip("/")

        # Extract only the path suffix relative to base to avoid duplication
        if target_url.startswith(base):
            path = target_url[len(base):]
        else:
            parsed = urlparse(target_url)
            path = parsed.path
            if parsed.query:
                path += f"?{parsed.query}"

        candidates = [base]
        is_local_url = any(
            h in self.runpod_url
            for h in ("localhost", "127.0.0.1", "host.docker.internal", "meetasr_runpod")
        )
        if is_local_url:
            fallbacks = [
                "http://localhost:8001",
                "http://127.0.0.1:8001",
                "http://host.docker.internal:8001",
                "http://meetasr_runpod:8001",
            ]
            for f in fallbacks:
                if f.rstrip("/") not in candidates:
                    candidates.append(f.rstrip("/"))

        return [f"{c}{path}" for c in candidates]

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
        candidate_urls = self._get_candidate_urls(url)
        last_exception: Exception | None = None

        for attempt in range(1, max_retries + 1):
            for candidate in candidate_urls:
                try:
                    async with httpx.AsyncClient(timeout=timeout, headers=self.headers) as client:
                        if json_payload is not None:
                            response = await client.post(candidate, json=json_payload)
                        else:
                            response = await client.post(candidate, files=files, data=data)

                        if response.status_code in (502, 503, 504):
                            logger.warning(
                                "ASR service at %s returned HTTP %s (attempt %d/%d). Retrying in %.1fs...",
                                candidate, response.status_code, attempt, max_retries, delay
                            )
                            last_exception = httpx.HTTPStatusError(
                                f"HTTP {response.status_code}",
                                request=response.request,
                                response=response,
                            )
                            break

                        response.raise_for_status()

                        # Update primary runpod_url if connected via a fallback
                        from urllib.parse import urlparse
                        base = candidate.rsplit(urlparse(candidate).path, 1)[0]
                        if base != self.runpod_url:
                            logger.info("Successfully connected to fallback ASR URL: %s", base)
                            self.runpod_url = base

                        return response
                except (httpx.ConnectError, httpx.NetworkError, httpx.TimeoutException) as exc:
                    last_exception = exc
                    continue
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in (502, 503, 504):
                        last_exception = exc
                        logger.warning(
                            "ASR service at %s returned HTTP %s (attempt %d/%d). Retrying in %.1fs...",
                            candidate, exc.response.status_code, attempt, max_retries, delay
                        )
                        break
                    else:
                        raise

            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 15.0)

        logger.error("ASR service connection failed after %d attempts: %s", max_retries, last_exception)
        raise RuntimeError(
            f"Cannot connect to ASR service after {max_retries} attempts (tried {candidate_urls}): {last_exception}. "
            f"Verify that ASR model server is running on port 8001."
        ) from last_exception

    async def transcribe(
        self,
        audio_source: Any,
        *,
        offset_ms: int = 0,
        key: str | None = None,
        on_chunk_complete: Any | None = None,
    ) -> ASRServiceResult:
        import base64

        # Prepare audio bytes
        if isinstance(audio_source, str) and os.path.exists(audio_source):
            with open(audio_source, "rb") as f:
                file_bytes = f.read()
        else:
            audio_arr = load_audio(audio_source)
            file_bytes = numpy_to_wav_bytes(audio_arr)

        if self._is_serverless:
            # Load as 16kHz mono numpy for chunking
            audio_arr = load_audio(file_bytes if not isinstance(audio_source, str) else audio_source)
            sample_rate = 16000
            # 4-minute chunks → ~7.7MB WAV → ~10MB base64, under 20MB limit
            chunk_samples = 4 * 60 * sample_rate
            chunks = [audio_arr[i:i + chunk_samples] for i in range(0, len(audio_arr), chunk_samples)]

            total_chunks = len(chunks)

            async def _process_chunk(chunk_idx: int, chunk: Any) -> None:
                chunk_offset_ms = offset_ms + chunk_idx * 4 * 60 * 1000
                chunk_bytes = numpy_to_wav_bytes(chunk, sample_rate)
                payload: dict = {"audio_base64": base64.b64encode(chunk_bytes).decode()}
                if key:
                    payload["key"] = f"{key}:chunk-{chunk_idx}"
                res_json = await self._call_serverless("transcribe", payload, timeout=600.0)

                chunk_segs: list[TranscriptSegmentPayload] = []
                for s in res_json.get("sentence_info", []):
                    if s["text"].strip():
                        chunk_segs.append(TranscriptSegmentPayload(
                            start_ms=chunk_offset_ms + int(s["start"] * 1000),
                            end_ms=chunk_offset_ms + int(s["end"] * 1000),
                            speaker=s.get("speaker"),
                            text=s["text"],
                        ))
                # Store results indexed so final assembly stays in order
                chunk_results[chunk_idx] = (
                    chunk_segs,
                    res_json.get("text", ""),
                    max(0, int(res_json.get("duration", 0) * 1000)),
                )
                if on_chunk_complete:
                    await on_chunk_complete(chunk_idx, total_chunks, chunk_segs)

            chunk_results: dict[int, tuple] = {}
            await asyncio.gather(*[_process_chunk(i, c) for i, c in enumerate(chunks)])

            all_segments: list[TranscriptSegmentPayload] = []
            all_text_parts: list[str] = []
            total_duration_ms = 0
            for i in range(total_chunks):
                segs, text, dur = chunk_results[i]
                all_segments.extend(segs)
                all_text_parts.append(text)
                total_duration_ms += dur

            if not all_segments and any(all_text_parts):
                all_segments.append(TranscriptSegmentPayload(
                    start_ms=offset_ms,
                    end_ms=offset_ms + total_duration_ms,
                    speaker=None,
                    text=" ".join(t for t in all_text_parts if t.strip()),
                ))

            return ASRServiceResult(
                segments=all_segments,
                text=" ".join(t for t in all_text_parts if t.strip()),
                duration_ms=total_duration_ms,
            )
        else:
            files = {"file": ("audio.wav", file_bytes)}
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

