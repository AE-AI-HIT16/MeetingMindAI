"""MeetASR — Meeting Speech Recognition + LLM Summarization."""

from meetasr.runpod.auto.auto_model import AutoModel
from meetasr.runpod.auto.auto_pipeline import AutoPipeline
from meetasr.runpod.register import tables

__version__ = "0.1.0"
__all__ = ["AutoModel", "AutoPipeline", "tables"]


def _register_all_models() -> None:
    """Import all model/component modules to trigger @tables.register() decorators.

    Called explicitly before AutoPipeline.from_yaml() — not at package import time —
    so a failing optional dependency doesn't crash the handler on startup.
    """
    import logging as _log
    _logger = _log.getLogger(__name__)

    # Frontend (required for feature extraction)
    from meetasr.runpod.frontends import fbank  # noqa: F401

    # LLM clients are optional — don't crash if groq/ollama not installed
    try:
        from meetasr.runpod.llm import groq_client, ollama_client, openai_client  # noqa: F401
    except Exception as exc:
        _logger.warning("LLM clients not available (non-fatal): %s", exc)

    # ASR models
    from meetasr.runpod.models.asr import (  # noqa: F401
        faster_whisper_asr,
        paraformer,
        qwen3_asr,
        sense_voice,
        zipformer_vi,
    )

    # Punctuation, speaker, VAD models
    from meetasr.runpod.models.punc import ct_transformer, vibert_capu  # noqa: F401
    from meetasr.runpod.models.spk import campplus  # noqa: F401
    from meetasr.runpod.models.vad import fsmn_vad, silero_vad  # noqa: F401

    # Tokenizer
    from meetasr.runpod.tokenizer import char_tokenizer  # noqa: F401
