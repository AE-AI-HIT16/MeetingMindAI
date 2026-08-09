"""AutoPipeline — build a full MeetPipeline from a config dict or YAML file."""

from __future__ import annotations

import logging
import os
from typing import Any

from omegaconf import OmegaConf

from meetasr.runpod.auto.auto_model import AutoModel
from meetasr.runpod.pipeline import MeetPipeline


class AutoPipeline:
    """Factory for building MeetPipeline from config.

    Example — from dict:
        >>> pipeline = AutoPipeline.from_config({
        ...     "asr": {"model": "sensevoice-small", "device": "cpu"},
        ...     "vad": {"model": "silero-vad"},
        ...     "punc": {"model": "ct-punc"},
        ...     "spk": {"model": "cam++"},
        ...     "llm": {
        ...         "provider": "ollama",
        ...         "model": "llama3.2",
        ...         "language": "vi",
        ...     },
        ... })
        >>> result = pipeline.transcribe("meeting.wav")

    Example — from YAML file:
        >>> pipeline = AutoPipeline.from_yaml("meeting_config.yaml")
    """

    @classmethod
    def from_config(cls, config: dict) -> MeetPipeline:
        """Build a MeetPipeline from a configuration dictionary.

        Args:
            config: Dict with keys: "asr" (required), "vad", "punc", "spk",
                "llm", and optional pipeline behavior settings.

        Returns:
            Configured MeetPipeline instance.
        """
        if "asr" not in config:
            raise ValueError("Config must have an 'asr' section with a 'model' key.")

        device = config.get("device", "cpu")

        # Build ASR (required)
        asr_cfg = dict(config["asr"])
        asr_cfg.setdefault("device", device)
        asr_model = AutoModel(**asr_cfg)

        # Build optional components
        vad_model = cls._build_optional(config, "vad", device)
        punc_model = cls._build_optional(config, "punc", device)
        spk_model = cls._build_optional(config, "spk", device)

        # Build LLM summarizer
        summarizer = None
        doc_planner = None
        if "llm" in config and config["llm"]:
            summarizer = cls._build_llm(config["llm"])
            doc_planner = cls._build_doc_planner(config["llm"])

        pipeline_cfg = config.get("pipeline") or {}
        gap_rescue_cfg = pipeline_cfg.get("gap_rescue") or {}
        diarization_first_cfg = pipeline_cfg.get("diarization_first") or {}

        return MeetPipeline(
            asr_model=asr_model,
            vad_model=vad_model,
            punc_model=punc_model,
            spk_model=spk_model,
            llm_summarizer=summarizer,
            doc_planner=doc_planner,
            device=device,
            enable_gap_rescue=gap_rescue_cfg.get("enabled", False),
            diarization_first=diarization_first_cfg.get("enabled", False),
            speaker_turn_max_chunk_ms=diarization_first_cfg.get(
                "max_chunk_ms",
                15000,
            ),
            speaker_turn_boundary_search_ms=diarization_first_cfg.get(
                "boundary_search_ms",
                2000,
            ),
            speaker_turn_min_chunk_ms=diarization_first_cfg.get(
                "min_chunk_ms",
                1000,
            ),
            transcription_language=pipeline_cfg.get(
                "transcription_language",
                "auto",
            ),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> MeetPipeline:
        """Build a MeetPipeline from a YAML config file.

        Args:
            yaml_path: Path to YAML config file.

        Returns:
            Configured MeetPipeline instance.

        Raises:
            FileNotFoundError: If yaml_path does not exist.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        cfg = OmegaConf.load(yaml_path)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        return cls.from_config(cfg_dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_optional(config: dict, key: str, device: str) -> Any:
        """Build an optional model component from config[key]."""
        if key not in config or not config[key]:
            return None
        cfg = dict(config[key])
        cfg.setdefault("device", device)
        logging.info(f"Building {key} model: {cfg.get('model')}")
        return AutoModel(**cfg)

    @staticmethod
    def _build_doc_planner(llm_cfg: dict) -> Any:
        """Build a DocumentPlanner from LLM config."""
        from meetasr.llm.planner import DocumentPlanner
        from meetasr.runpod.register import tables

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
        """Build a MeetingSummarizer from LLM config."""
        from meetasr.llm.summarizer import MeetingSummarizer
        from meetasr.runpod.register import tables

        provider = llm_cfg.get("provider", "openai")
        llm_class = tables.llm_classes.get(provider)
        if llm_class is None:
            registered = tables.list_registered("llm_classes")
            raise ValueError(
                f"LLM provider '{provider}' not registered. "
                f"Available: {registered}"
            )

        # Build client kwargs — strip non-client keys
        client_kwargs = {
            k: v for k, v in llm_cfg.items()
            if k not in (
                "provider", "language", "temperature", "max_tokens", "use_planner"
            )
        }
        # Resolve env vars in api_key
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
