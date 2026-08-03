"""Health check endpoint — GET /v1/health."""
from __future__ import annotations

from fastapi import APIRouter, Request

from meetasr import __version__
from meetasr.api import dependencies

router = APIRouter(tags=["System"])


@router.get("/v1/metrics/inference")
async def inference_metrics(request: Request) -> dict:
    """Expose bounded, non-sensitive scheduling metrics for operations."""
    coordinator = getattr(
        request.app.state,
        "inference_coordinator",
        None,
    )
    if coordinator is None:
        return {"available": False}
    snapshot = coordinator.snapshot()
    return {
        "available": True,
        "queue_depth": snapshot.queue_depth,
        "max_queue_depth": snapshot.max_queue_depth,
        "asr_call_count": snapshot.asr_call_count,
        "fallback_count": snapshot.fallback_count,
        "partial_drop_count": snapshot.partial_drop_count,
        "failure_count": snapshot.failure_count,
        "average_wait_ms": snapshot.average_wait_ms,
        "average_run_ms": snapshot.average_run_ms,
        "average_wait_ms_by_kind": {
            kind.value: value
            for kind, value in snapshot.average_wait_ms_by_kind.items()
        },
        "average_run_ms_by_kind": {
            kind.value: value
            for kind, value in snapshot.average_run_ms_by_kind.items()
        },
        "completed_by_kind": {
            kind.value: count
            for kind, count in snapshot.completed_by_kind.items()
        },
    }


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
