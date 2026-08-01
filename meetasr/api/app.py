"""FastAPI application entry point for MeetASR.

Registers all routers and configures the application lifespan,
CORS middleware, and global exception handler.
"""
from __future__ import annotations

import logging
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from meetasr import __version__
from meetasr.api.dependencies import CONFIG_PATH, set_pipeline
from meetasr.api.routes import (
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
from meetasr.auto.auto_pipeline import AutoPipeline
from meetasr.db.connection import init_db
from meetasr.realtime.document_generation import document_generation_queue
from meetasr.realtime.job_worker import job_queue
from meetasr.services.asr_service import ASRService
from meetasr.services.realtime_asr_service import RealtimeASRService
from meetasr.pipeline_realtime import ASRPipeline
from meetasr.streaming.final_transcript_queue import FinalTranscriptQueue
from meetasr.streaming.final_transcript_worker import FinalTranscriptWorker


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
    app.state.final_transcript_queue = None
    app.state.final_transcript_worker = None
    app.state.final_transcript_task = None

    init_db()
    logger.info("Database tables initialized.")

    # --------------------------------------------------------------
    # STARTUP
    # --------------------------------------------------------------
    if os.path.exists(CONFIG_PATH):
        logger.info(f"Loading pipeline from {CONFIG_PATH}...")

        # Full pipeline:
        # ASR + VAD + Punctuation + Speaker + LLM + Document planner
        pipeline = AutoPipeline.from_yaml(CONFIG_PATH)

        # Lightweight realtime pipeline:
        # ASR + VAD only
        # Reuse loaded model instances from full pipeline
        realtime_pipeline = ASRPipeline(
            asr_model=pipeline.asr,
            vad_model=pipeline.vad,
            device=pipeline.device,
        )

        # Keep global full pipeline reference
        set_pipeline(pipeline)

        app.state.pipeline = pipeline
        app.state.realtime_pipeline = realtime_pipeline

        # ----------------------------------------------------------
        # Create independent services
        # ----------------------------------------------------------

        # Full processing service:
        # Used by upload jobs / transcription APIs
        app.state.asr_service = ASRService(
            pipeline
        )

        # Realtime websocket service:
        # Used by realtime streaming workers
        app.state.realtime_asr_service = RealtimeASRService(
            realtime_pipeline
        )

        logger.info("Full ASR pipeline ready.")
        logger.info("Realtime ASR pipeline ready.")

        # ----------------------------------------------------------
        # Final transcript worker
        # ----------------------------------------------------------

        app.state.final_transcript_queue = FinalTranscriptQueue()

        app.state.final_transcript_worker = FinalTranscriptWorker(
            queue=app.state.final_transcript_queue,
            pipeline=pipeline,
        )

        app.state.final_transcript_task = asyncio.create_task(
            app.state.final_transcript_worker.run()
        )

        logger.info("Final transcript worker started.")

    else:
        logger.warning(
            f"Config '{CONFIG_PATH}' not found. "
            "Server starts without pipeline."
        )

    # --------------------------------------------------------------
    # Start background workers
    # --------------------------------------------------------------
    await job_queue.start(
        app.state.asr_service,
        sources.get_storage_backend(),
    )

    await document_generation_queue.start(
        getattr(
            app.state.pipeline,
            "doc_planner",
            None,
        )
    )

    print(logger.level)
    print(logger.getEffectiveLevel())

    yield

    # --------------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------------
    logger.info(
        "Shutting down... Cleaning up ML models and freeing VRAM."
    )

    await document_generation_queue.stop()
    await job_queue.stop()

    set_pipeline(None)

    app.state.asr_service = None
    app.state.realtime_asr_service = None

    app.state.pipeline = None
    app.state.realtime_pipeline = None

    # --------------------------------------------------------------
    # Stop final transcript worker
    # --------------------------------------------------------------

    if app.state.final_transcript_task:

        app.state.final_transcript_task.cancel()

        try:

            await app.state.final_transcript_task

        except asyncio.CancelledError:
            pass

    if app.state.final_transcript_queue:
        await app.state.final_transcript_queue.clear()

    app.state.final_transcript_worker = None
    app.state.final_transcript_queue = None
    app.state.final_transcript_task = None

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
