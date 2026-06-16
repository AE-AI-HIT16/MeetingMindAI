"""FastAPI application for MeetASR REST API."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from meetasr import __version__
from meetasr.auto.auto_pipeline import AutoPipeline
from meetasr.pipeline import MeetPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="MeetASR API",
    description="Meeting Speech Recognition + LLM Summarization",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Pipeline instance — loaded once at startup
_pipeline: Optional[MeetPipeline] = None
_config_path = os.environ.get("MEETASR_CONFIG", "meeting_config.yaml")

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".webm"}
MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Load pipeline on server startup."""
    global _pipeline
    if os.path.exists(_config_path):
        logging.info(f"Loading pipeline from {_config_path}...")
        _pipeline = AutoPipeline.from_yaml(_config_path)
        logging.info("Pipeline ready.")
    else:
        logging.warning(
            f"Config file '{_config_path}' not found. "
            "Server starts without a pipeline — set MEETASR_CONFIG or POST a config."
        )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/v1/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": __version__,
        "pipeline_loaded": _pipeline is not None,
    }


@app.post("/v1/audio/transcriptions", tags=["ASR"])
async def transcribe(
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, mp4, flac)"),
    language: str = Form("auto", description="Language code: auto, vi, zh, en"),
    response_format: str = Form("verbose_json", description="json | text | verbose_json | srt"),
    speaker_diarization: bool = Form(False, description="Enable speaker detection"),
):
    """Transcribe an audio file to text.

    Returns transcript in the requested format.
    """
    _check_pipeline()
    audio_path = await _save_upload(file)

    try:
        result = _pipeline.transcribe(
            audio_path,
            language=language if language != "auto" else "auto",
        )
    finally:
        _safe_remove(audio_path)

    if response_format == "text":
        return PlainTextResponse(result.text)
    elif response_format == "srt":
        return PlainTextResponse(result.to_srt(), media_type="text/plain")
    elif response_format == "json":
        return {"text": result.text}
    else:  # verbose_json (default)
        return JSONResponse(result.to_dict())


@app.post("/v1/meeting/summarize", tags=["Meeting"])
async def summarize_meeting(
    file: UploadFile = File(..., description="Audio file"),
    language: str = Form("vi", description="Output language: vi | en"),
    response_format: str = Form("json", description="json | markdown"),
):
    """Full pipeline: transcribe + LLM summarization.

    Returns MeetingReport with summary, topics, action items, and decisions.
    """
    _check_pipeline()
    if _pipeline.summarizer is None:
        raise HTTPException(
            status_code=503,
            detail="LLM summarizer is not configured. Add 'llm' section to your config YAML.",
        )

    audio_path = await _save_upload(file)

    try:
        report = _pipeline.summarize_meeting(audio_path, language=language)
    finally:
        _safe_remove(audio_path)

    if response_format == "markdown":
        return PlainTextResponse(report.to_markdown(), media_type="text/markdown")

    return JSONResponse(report.to_dict())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _check_pipeline():
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Pipeline not loaded. "
                f"Create '{_config_path}' and restart the server."
            ),
        )


async def _save_upload(file: UploadFile) -> str:
    """Save uploaded file to a temp path and return the path."""
    ext = os.path.splitext(file.filename or "audio.wav")[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {sorted(SUPPORTED_FORMATS)}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // 1024 // 1024} MB). Max 500 MB.",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(content)
    tmp.close()
    return tmp.name


def _safe_remove(path: str):
    try:
        os.remove(path)
    except Exception:
        pass
