"""MeetASR — Meeting Speech Recognition + LLM Summarization."""

from meetasr.register import tables
from meetasr.auto.auto_model import AutoModel
from meetasr.auto.auto_pipeline import AutoPipeline

# Trigger registration of all components
from meetasr.models.vad import fsmn_vad  # noqa: F401
from meetasr.models.asr import sense_voice, paraformer  # noqa: F401
from meetasr.models.punc import ct_transformer  # noqa: F401
from meetasr.models.spk import campplus  # noqa: F401
from meetasr.frontends import fbank  # noqa: F401
from meetasr.tokenizer import char_tokenizer, sentencepiece_tokenizer  # noqa: F401

__version__ = "0.1.0"
__all__ = ["AutoModel", "AutoPipeline", "tables"]
