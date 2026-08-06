"""FastAPI application entry point for MeetASR.

Registers all routers and configures the application lifespan,
CORS middleware, and global exception handler.
"""
# Loading .env must happen before importing route modules because they read
# authentication and database variables at import time.
# ruff: noqa: E402

from __future__ import annotations

import logging
import os
import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load the repository-local .env without overriding variables supplied by the
# shell, container or deployment platform.
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from meetasr.backend import __version__
from meetasr.backend.api.dependencies import CONFIG_PATH, set_pipeline
from meetasr.backend.api.routes import (
    auth,
    db_routes,
    document,
    document_jobs,
    documents_phase2,
    health,
    jobs,
    realtime,
    sources,
    summarize,
    test_model,
    transcribe,
)
from meetasr.backend.pipeline import BackendPipeline
from meetasr.backend.db.connection import init_db
from meetasr.backend.realtime.document_generation import document_generation_queue
from meetasr.backend.realtime.job_worker import job_queue
from meetasr.backend.services.asr_service import ASRService
from meetasr.backend.services.realtime_asr_service import RealtimeASRService
from meetasr.backend.streaming.final_transcript_queue import FinalTranscriptQueue
from meetasr.backend.streaming.final_transcript_worker import FinalTranscriptWorker


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Lifecycle (modern FastAPI pattern replacing deprecated on_event)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifespan: load pipeline on startup, release on shutdown."""

    # --------------------------------------------------------------
    # Initialize application state
    # --------------------------------------------------------------
    app.state.pipeline = None
    app.state.realtime_pipeline = None

    app.state.asr_service = None
    app.state.realtime_asr_service = None

    init_db()
    logger.info("Database tables initialized.")

    # --------------------------------------------------------------
    # STARTUP
    # --------------------------------------------------------------
    if os.path.exists(CONFIG_PATH):
        logger.info(f"Loading LLM pipeline from {CONFIG_PATH}...")
        pipeline = BackendPipeline.from_yaml(CONFIG_PATH)
        set_pipeline(pipeline)
        app.state.pipeline = pipeline
    else:
        logger.warning(
            f"Config '{CONFIG_PATH}' not found. "
            "Server starts without LLM pipeline."
        )

    runpod_url = os.environ.get("RUNPOD_URL")
    if not runpod_url:
        endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
        base_url = os.environ.get("RUNPOD_API_BASE_URL", "https://api.runpod.ai/v2").rstrip("/")
        if endpoint_id:
            runpod_url = f"{base_url}/{endpoint_id}"
        else:
            runpod_url = "http://localhost:8001"

    runpod_api_key = os.environ.get("RUNPOD_API_KEY", "")
    logger.info(f"Using RunPod URL: {runpod_url}")

    app.state.asr_service = ASRService(runpod_url=runpod_url, api_key=runpod_api_key)
    app.state.realtime_asr_service = RealtimeASRService(runpod_url=runpod_url, api_key=runpod_api_key)

    print(logger.level)
    print(logger.getEffectiveLevel())

    yield

    # --------------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------------
    logger.info(
        "Shutting down... Cleaning up ML models and freeing VRAM."
    )

    set_pipeline(None)

    app.state.asr_service = None
    app.state.realtime_asr_service = None

    app.state.pipeline = None

# Set log cho api realtime

root = logging.getLogger()
root.setLevel(logging.INFO)

# ------------------------------------------------------------------
# App Initialization
# ------------------------------------------------------------------
app = FastAPI(
    title="MeetASR API",
    description="Meeting Speech Recognition + LLM Summarization",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Allow all origins for local dev — restrict origins on production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to actual domain on production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health.router)
app.include_router(auth.router)       # POST /v1/auth/google, GET /v1/auth/me
app.include_router(transcribe.router)
app.include_router(summarize.router)
app.include_router(db_routes.router)
app.include_router(document.router)
app.include_router(documents_phase2.router)
app.include_router(document_jobs.router)
app.include_router(realtime.router)
app.include_router(jobs.router)
app.include_router(test_model.router)
app.include_router(sources.router)   # Phase 2: /v1/sources — upload, library, media stream


# ------------------------------------------------------------------
# Global Exception Handler
# ------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return a safe 500 response.

    Args:
        request: The incoming HTTP request.
        exc: The unhandled exception.

    Returns:
        JSONResponse with status 500 and a generic error body.
    """
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Internal server error.",
            }
        },
    )
