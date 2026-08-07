"""MeetASR — Meeting Speech Recognition + LLM Summarization."""

from meetasr.runpod.auto.auto_model import AutoModel
from meetasr.runpod.auto.auto_pipeline import AutoPipeline
from meetasr.runpod.frontends import fbank  # noqa: F401
from meetasr.runpod.llm import groq_client, ollama_client, openai_client  # noqa: F401
from meetasr.runpod.models.asr import (  # noqa: F401
    faster_whisper_asr,
    paraformer,
    qwen3_asr,
    sense_voice,
    zipformer_vi,
)
from meetasr.runpod.models.punc import ct_transformer, vibert_capu  # noqa: F401
from meetasr.runpod.models.spk import campplus  # noqa: F401
from meetasr.runpod.models.vad import fsmn_vad, silero_vad  # noqa: F401
from meetasr.runpod.register import tables
from meetasr.runpod.tokenizer import char_tokenizer  # noqa: F401

__version__ = "0.1.0"
__all__ = ["AutoModel", "AutoPipeline", "tables"]
