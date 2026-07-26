"""FastAPI application entry point for MeetASR.

Registers all routers and configures the application lifespan,
CORS middleware, and global exception handler.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from meetasr import __version__
from meetasr.auto.auto_pipeline import AutoPipeline
from meetasr.api.dependencies import set_pipeline, CONFIG_PATH
from meetasr.api.routes import (
    db_routes,
    document,
    health,
    realtime,
    sources,
    summarize,
    test_model,
    transcribe,
)
from meetasr.db.connection import init_db


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Lifecycle (modern FastAPI pattern replacing deprecated on_event)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifespan: load pipeline on startup, release on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the running server between startup and shutdown.
    """
    app.state.pipeline = None
    init_db()
    logger.info("Database tables initialized.")
    # --- STARTUP ---
    if os.path.exists(CONFIG_PATH):
        logger.info(f"Loading pipeline from {CONFIG_PATH}...")
        pipeline = AutoPipeline.from_yaml(CONFIG_PATH)
        set_pipeline(pipeline)

        app.state.pipeline = pipeline
        logger.info("Pipeline ready.")
    else:
        logger.warning(
            f"Config '{CONFIG_PATH}' not found. "
            "Server starts without pipeline — set MEETASR_CONFIG."
        )

    print(logger.level)
    print(logger.getEffectiveLevel())

    yield  # server is now running and serving requests

    # --- SHUTDOWN ---
    logger.info("Shutting down... Cleaning up ML models and freeing VRAM.")
    set_pipeline(None)  # release reference so GC can free RAM/VRAM
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
app.include_router(transcribe.router)
app.include_router(summarize.router)
app.include_router(db_routes.router)
app.include_router(document.router)
app.include_router(realtime.router)
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
