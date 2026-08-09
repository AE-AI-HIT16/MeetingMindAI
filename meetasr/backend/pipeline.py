from __future__ import annotations
import os
import logging
from typing import Any
from omegaconf import OmegaConf
from meetasr.backend.llm.summarizer import MeetingSummarizer
from meetasr.backend.llm.planner import DocumentPlanner
from meetasr.backend.register import tables
from meetasr.backend.llm import groq_client, ollama_client, openai_client

logger = logging.getLogger(__name__)

class BackendPipeline:
    def __init__(self, summarizer=None, doc_planner=None):
        self.summarizer = summarizer
        self.doc_planner = doc_planner
        self.asr = "runpod_asr"
        self.vad = "runpod_vad"
        self.punc = "runpod_punc"
        self.spk = "runpod_spk"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> BackendPipeline:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        cfg = OmegaConf.load(yaml_path)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        return cls.from_config(cfg_dict)

    @classmethod
    def from_config(cls, config: dict) -> BackendPipeline:
        summarizer = None
        doc_planner = None
        if "llm" in config and config["llm"]:
            summarizer = cls._build_llm(config["llm"])
            doc_planner = cls._build_doc_planner(config["llm"])
        return cls(summarizer=summarizer, doc_planner=doc_planner)

    @staticmethod
    def _build_doc_planner(llm_cfg: dict) -> Any:
        provider = llm_cfg.get("provider", "openai")
        llm_class = tables.llm_classes.get(provider)
        if llm_class is None:
            return None

        client_kwargs = {
            k: v for k, v in llm_cfg.items()
            if k not in ("provider", "language", "temperature", "max_tokens", "use_planner")
        }
        if "api_key" in client_kwargs:
            key_val = client_kwargs["api_key"]
            if isinstance(key_val, str) and key_val.startswith("${"):
                client_kwargs["api_key"] = os.environ.get(key_val[2:-1], "")

        client = llm_class(**client_kwargs)
        return DocumentPlanner(
            client=client,
            language=llm_cfg.get("language", "vi"),
            temperature=llm_cfg.get("temperature", 0.3),
            max_tokens=llm_cfg.get("max_tokens", 4096),
        )

    @staticmethod
    def _build_llm(llm_cfg: dict) -> Any:
        provider = llm_cfg.get("provider", "openai")
        llm_class = tables.llm_classes.get(provider)
        if llm_class is None:
            registered = tables.list_registered("llm_classes")
            raise ValueError(
                f"LLM provider '{provider}' not registered. "
                f"Available: {registered}"
            )

        client_kwargs = {
            k: v for k, v in llm_cfg.items()
            if k not in (
                "provider", "language", "temperature", "max_tokens", "use_planner"
            )
        }
        if "api_key" in client_kwargs:
            key_val = client_kwargs["api_key"]
            if isinstance(key_val, str) and key_val.startswith("${"):
                env_name = key_val[2:-1]
                client_kwargs["api_key"] = os.environ.get(env_name, "")

        client = llm_class(**client_kwargs)

        return MeetingSummarizer(
            client=client,
            language=llm_cfg.get("language", "vi"),
            temperature=llm_cfg.get("temperature", 0.3),
            max_tokens=llm_cfg.get("max_tokens", 4096),
        )
