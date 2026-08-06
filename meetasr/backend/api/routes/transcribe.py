"""Transcription endpoint — POST /v1/audio/transcriptions."""
from __future__ import annotations
import logging
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse
from meetasr.backend.api.dependencies import get_pipeline, save_upload, safe_remove

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ASR"])


@router.post("/v1/audio/transcriptions",response_model=None)
async def transcribe(
    pipeline=Depends(get_pipeline),
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, mp4, flac)"),
    model: str = Form("sensevoice", description="sensevoice (default) | paraformer | fun-asr-nano"),
    language: str = Form("auto", description="Language code: auto, vi, zh, en"),
    response_format: str = Form("json", description="json | text | verbose_json | srt"),
    speaker_diarization: bool = Form(False, description="Enable speaker detection"),
    timestamp_granularity: str = Form("segment", description="segment (default) | word"),
) -> JSONResponse | PlainTextResponse | dict:
    """Transcribe an audio file to text.

    Accepts multipart form data with an audio file and optional parameters.
    Returns transcript in the requested format.
    """
    audio_path = await save_upload(file)

    try:
        # NOTE: `model` and `timestamp_granularity` are accepted per spec
        # but not yet passed to pipeline (pending model selection support).
        result = pipeline.transcribe(
            audio_path,
            language=language if language != "auto" else "auto",
            speaker_diarization=speaker_diarization,
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "transcription_failed",
                    "message": str(e),
                }
            },
        )
    finally:
        safe_remove(audio_path)

    match response_format:
        case "text":
            return PlainTextResponse(result.text)
        case "srt":
            return PlainTextResponse(result.to_srt(), media_type="text/srt")
        case "verbose_json":
            return JSONResponse(result.to_dict())
        case _:
            return {"text": result.text}
