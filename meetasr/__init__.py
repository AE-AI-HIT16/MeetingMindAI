"""MeetASR — Meeting Speech Recognition + LLM Summarization."""

from meetasr.auto.auto_model import AutoModel
from meetasr.auto.auto_pipeline import AutoPipeline
from meetasr.frontends import fbank  # noqa: F401
from meetasr.llm import groq_client, ollama_client, openai_client  # noqa: F401
from meetasr.models.asr import (  # noqa: F401
    faster_whisper_asr,
    paraformer,
    qwen3_asr,
    sense_voice,
    zipformer_vi,
)
from meetasr.models.punc import ct_transformer, vibert_capu  # noqa: F401
from meetasr.models.spk import campplus  # noqa: F401
from meetasr.models.vad import fsmn_vad, silero_vad  # noqa: F401
from meetasr.register import tables
from meetasr.tokenizer import char_tokenizer  # noqa: F401

__version__ = "0.1.0"
__all__ = ["AutoModel", "AutoPipeline", "tables"]
