"""RunPod FastAPI server — hosts ML model endpoints for the Backend to call.

This server loads the MeetPipeline (ASR + VAD + SPK + Punc) on startup and
exposes REST API endpoints that the Backend (on AWS) calls via HTTP.

Environment variables:
    RUNPOD_API_KEY  — Bearer token for authentication (optional for local dev).
    CONFIG_PATH     — Path to the YAML pipeline config (default: config.yaml).
    PORT            — Port to listen on (default: 8001).
"""

from __future__ import annotations

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from meetasr.runpod.auto.auto_pipeline import AutoPipeline
from meetasr.runpod.pipeline import MeetPipeline
from meetasr.runpod.pipeline_realtime import ASRPipeline
from meetasr.runpod.schemas import Segment, SentenceInfo, SpeakerTurn
from meetasr.runpod.utils.audio import load_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
EXPECTED_TOKEN = os.getenv("RUNPOD_API_KEY", "")

security = HTTPBearer(auto_error=False)

from dotenv import load_dotenv
load_dotenv()


# ------------------------------------------------------------------
# Auth dependency
# ------------------------------------------------------------------
async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> None:
    """Verify Bearer token if RUNPOD_API_KEY is configured."""
    if not EXPECTED_TOKEN:
        return  # No auth configured — allow all (local dev)
    if credentials is None or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
async def _read_audio_from_upload(file: UploadFile) -> np.ndarray:
    """Read an uploaded file and convert to a numpy audio array."""
    content = await file.read()
    return load_audio(content)


def _sentence_info_to_dict(si: SentenceInfo) -> dict:
    return {
        "text": si.text,
        "start": si.start,
        "end": si.end,
        "speaker": si.speaker,
        "char_timestamps": si.char_timestamps,
    }


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML pipeline on startup, release on shutdown."""
    logger.info("Loading ML pipeline from %s ...", CONFIG_PATH)
    try:
        pipeline = AutoPipeline.from_yaml(CONFIG_PATH)
        app.state.pipeline = pipeline
        logger.info("MeetPipeline loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load MeetPipeline: %s", exc, exc_info=True)
        app.state.pipeline = None

    # Build a realtime pipeline sharing the same ASR + VAD models
    if app.state.pipeline is not None:
        p = app.state.pipeline
        app.state.realtime_pipeline = ASRPipeline(
            asr_model=p.asr,
            vad_model=p.vad,
            device=p.device,
        )
        logger.info("Realtime ASRPipeline loaded successfully.")
    else:
        app.state.realtime_pipeline = None

    yield

    logger.info("Shutting down RunPod model server...")
    app.state.pipeline = None
    app.state.realtime_pipeline = None


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(
    title="MeetASR RunPod Model Server",
    description="GPU-accelerated ML model endpoints for MeetASR",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)


def _get_pipeline(request: Request) -> MeetPipeline:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return pipeline


def _get_realtime_pipeline(request: Request) -> ASRPipeline:
    pipeline = request.app.state.realtime_pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Realtime pipeline not loaded")
    return pipeline


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------
@app.get("/health")
@app.get("/ping")
async def health():
    if getattr(app.state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return {"status": "ok", "pipeline_loaded": True}


# ------------------------------------------------------------------
# POST /v1/transcribe — Full pipeline transcription
# ------------------------------------------------------------------
@app.post("/v1/transcribe", dependencies=[Depends(verify_token)])
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    key: Optional[str] = Form(None),
):
    """Run full pipeline (VAD + ASR + SPK + Punc) on uploaded audio."""
    pipeline = _get_pipeline(request)
    audio = await _read_audio_from_upload(file)

    import asyncio
    result = await asyncio.to_thread(pipeline.transcribe, audio, key=key)

    return {
        "key": result.key,
        "text": result.text,
        "duration": result.duration,
        "sentence_info": [_sentence_info_to_dict(s) for s in result.sentence_info],
    }


# ------------------------------------------------------------------
# POST /v1/prepare_incremental — VAD + optional diarization
# ------------------------------------------------------------------
@app.post("/v1/prepare_incremental", dependencies=[Depends(verify_token)])
async def prepare_incremental(
    request: Request,
    file: UploadFile = File(...),
):
    """Decode audio, run VAD (+ optional speaker diarization), return segments."""
    pipeline = _get_pipeline(request)
    audio = await _read_audio_from_upload(file)

    import asyncio

    if getattr(pipeline, "diarization_first", False):
        audio_out, vad_segments, speaker_turns, duration_ms = await asyncio.to_thread(
            pipeline.prepare_diarization_first_transcription, audio
        )
    else:
        audio_out, vad_segments, duration_ms = await asyncio.to_thread(
            pipeline.prepare_incremental_transcription, audio
        )
        speaker_turns = None

    response = {
        "vad_segments": [
            {"start_ms": s.start_ms, "end_ms": s.end_ms} for s in vad_segments
        ],
        "duration_ms": duration_ms,
        "speaker_turns": None,
    }
    if speaker_turns is not None:
        response["speaker_turns"] = [
            {"start_ms": t.start_ms, "end_ms": t.end_ms, "speaker": t.speaker}
            for t in speaker_turns
        ]

    return response


# ------------------------------------------------------------------
# POST /v1/transcribe_segment — Transcribe one VAD segment
# ------------------------------------------------------------------
@app.post("/v1/transcribe_segment", dependencies=[Depends(verify_token)])
async def transcribe_segment(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("auto"),
    key: Optional[str] = Form(None),
):
    """Transcribe a single audio chunk (one VAD segment)."""
    pipeline = _get_pipeline(request)
    audio = await _read_audio_from_upload(file)

    # The backend sends a pre-sliced chunk, so we create a synthetic
    # full-length segment covering the entire chunk.
    duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
    segment = Segment(0, duration_ms)

    import asyncio
    sentences = await asyncio.to_thread(
        pipeline.transcribe_vad_segment, audio, segment, language=language
    )

    return [_sentence_info_to_dict(s) for s in sentences]


# ------------------------------------------------------------------
# POST /v1/finalize_incremental — Final speaker + punc pass
# ------------------------------------------------------------------
@app.post("/v1/finalize_incremental", dependencies=[Depends(verify_token)])
async def finalize_incremental(
    request: Request,
    file: Optional[UploadFile] = File(None),
    sentences_json: Optional[str] = Form(None),
    vad_segments_json: Optional[str] = Form(None),
):
    """Run final diarization/punctuation on already-transcribed sentences.

    Two modes:
    1. Diarization-first (preassigned speakers): JSON body with ``sentences``.
    2. ASR-first (legacy): multipart with audio file + sentences + VAD segments.
    """
    pipeline = _get_pipeline(request)
    import asyncio

    # Try JSON body first (diarization-first mode)
    if sentences_json is None:
        body = await request.json()
        sentences_data = body.get("sentences", [])
    else:
        sentences_data = json.loads(sentences_json)

    sentences = [
        SentenceInfo(
            text=s["text"],
            start=s["start"],
            end=s["end"],
            speaker=s.get("speaker"),
            char_timestamps=s.get("char_timestamps", []),
        )
        for s in sentences_data
    ]

    # If file is present → ASR-first mode (needs audio + VAD segments)
    if file is not None:
        audio = await _read_audio_from_upload(file)
        vad_segments = [
            Segment(s["start_ms"], s["end_ms"])
            for s in json.loads(vad_segments_json or "[]")
        ]
        result = await asyncio.to_thread(
            pipeline.finalize_incremental_transcript, audio, sentences, vad_segments
        )
    else:
        # Diarization-first mode: speakers already assigned
        result = await asyncio.to_thread(
            pipeline.finalize_preassigned_transcript, sentences
        )

    return [_sentence_info_to_dict(s) for s in result]


# ------------------------------------------------------------------
# POST /v1/realtime/transcribe — Realtime full transcription
# ------------------------------------------------------------------
@app.post("/v1/realtime/transcribe", dependencies=[Depends(verify_token)])
async def realtime_transcribe(
    request: Request,
    file: UploadFile = File(...),
    key: Optional[str] = Form(None),
):
    """Transcribe a realtime audio chunk using the realtime pipeline."""
    pipeline = _get_realtime_pipeline(request)
    audio = await _read_audio_from_upload(file)

    import asyncio
    result = await asyncio.to_thread(pipeline.transcribe, audio, key=key)

    return {
        "key": result.key,
        "text": result.text,
        "duration": result.duration,
        "sentence_info": [_sentence_info_to_dict(s) for s in result.sentence_info],
    }


# ------------------------------------------------------------------
# POST /v1/realtime/recognize — Partial recognition
# ------------------------------------------------------------------
@app.post("/v1/realtime/recognize", dependencies=[Depends(verify_token)])
async def realtime_recognize(
    request: Request,
    file: UploadFile = File(...),
):
    """Recognize a partial audio chunk for streaming partial results."""
    pipeline = _get_realtime_pipeline(request)
    audio = await _read_audio_from_upload(file)

    import asyncio
    # Use the underlying ASR model directly for partial recognition
    results = await asyncio.to_thread(pipeline.asr.recognize, [audio])

    return results


# ------------------------------------------------------------------
# POST /v1/debug/vad — Debug VAD segments
# ------------------------------------------------------------------
@app.post("/v1/debug/vad", dependencies=[Depends(verify_token)])
async def debug_vad(
    request: Request,
    file: UploadFile = File(...),
):
    """Run VAD only and return detected segments for debugging."""
    pipeline = _get_pipeline(request)
    audio = await _read_audio_from_upload(file)

    import asyncio
    segments = await asyncio.to_thread(pipeline._run_vad, audio)

    return {
        "segments": [
            {"start_ms": s.start_ms, "end_ms": s.end_ms, "duration_ms": s.duration_ms}
            for s in segments
        ],
        "count": len(segments),
        "total_duration_ms": int(len(audio) / SAMPLE_RATE * 1000),
    }


# ------------------------------------------------------------------
# Global Exception Handler
# ------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": str(exc)}},
    )


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("meetasr.runpod.app:app", host="0.0.0.0", port=port, reload=False)
