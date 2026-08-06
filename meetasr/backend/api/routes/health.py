"""Health check endpoint — GET /v1/health."""
from __future__ import annotations

from fastapi import APIRouter

from meetasr.backend import __version__
from meetasr.backend.api import dependencies

router = APIRouter(tags=["System"])


@router.get("/v1/health")
async def health() -> dict:
    """Health check — reports server status and loaded models.
    Returns:
        JSON with status, version, models_loaded, llm_available.
    """
    pipeline = dependencies._pipeline
    models_loaded: list[str] = []
    if pipeline is not None:
        # Collect names of loaded model components
        if getattr(pipeline, "vad", None) is not None:
            vad_name = getattr(pipeline.vad, "model_name", None)
            models_loaded.append(
                vad_name
                if isinstance(vad_name, str)
                else type(pipeline.vad).__name__
            )
        if getattr(pipeline, "asr", None) is not None:
            models_loaded.append(getattr(pipeline.asr, "model_name", "asr"))
        if getattr(pipeline, "punc", None) is not None:
            models_loaded.append("ct-punc")
        if getattr(pipeline, "spk", None) is not None:
            models_loaded.append("cam++")

    return {
        "status": "ok",
        "version": __version__,
        "models_loaded": models_loaded,
        "llm_available": (
            pipeline is not None
            and getattr(pipeline, "summarizer", None) is not None
        ),
    }
