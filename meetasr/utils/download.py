"""Model download from ModelScope and HuggingFace hubs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def download_model(
    model: str,
    hub: str = "ms",
    model_revision: str = "master",
    **kwargs: Any,
) -> dict:
    """Download model and return its resolved config dict.

    Args:
        model: Model name/ID (e.g. "fsmn-vad", "iic/SenseVoiceSmall").
        hub: "ms" for ModelScope or "hf" for HuggingFace.
        model_revision: Branch/tag/commit. Default "master".
        **kwargs: Extra kwargs passed through to caller.

    Returns:
        Config dict with "model_path" injected and all YAML keys available.

    Raises:
        RuntimeError: If model cannot be found or downloaded.
    """
    model_path = _resolve_model_path(model, hub, model_revision)
    config = _load_config(model_path)
    config["model_path"] = model_path
    config["hub"] = hub
    return config


def _resolve_model_path(model: str, hub: str, revision: str) -> str:
    """Download model files and return local cache path."""
    # Check if model is already a local path
    if os.path.isdir(model):
        logging.info(f"Using local model directory: {model}")
        return model

    if hub == "ms":
        return _download_from_modelscope(model, revision)
    elif hub == "hf":
        return _download_from_huggingface(model, revision)
    else:
        raise ValueError(f"Unknown hub '{hub}'. Expected 'ms' or 'hf'.")


def _download_from_modelscope(model_id: str, revision: str) -> str:
    """Download from ModelScope and return local path."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        # Resolve shorthand aliases (same as FunASR)
        model_id = _resolve_model_alias(model_id, hub="ms")
        logging.info(f"Downloading '{model_id}' from ModelScope (revision={revision})")
        return snapshot_download(model_id, revision=revision)
    except ImportError:
        raise RuntimeError(
            "modelscope is not installed. Run: pip install modelscope"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download '{model_id}' from ModelScope: {e}")


def _download_from_huggingface(model_id: str, revision: str) -> str:
    """Download from HuggingFace and return local path."""
    try:
        from huggingface_hub import snapshot_download
        model_id = _resolve_model_alias(model_id, hub="hf")
        logging.info(f"Downloading '{model_id}' from HuggingFace (revision={revision})")
        return snapshot_download(model_id, revision=revision)
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download '{model_id}' from HuggingFace: {e}")


def _load_config(model_path: str) -> dict:
    """Load config.yaml from model directory."""
    config_path = os.path.join(model_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found in model directory: {model_path}"
        )
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


# Shorthand aliases — same as FunASR for compatibility
_MS_ALIASES = {
    "fsmn-vad": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "damo/punc_ct-transformer_cn-en-common-vocab471067-large",
    "cam++": "iic/speech_campplus_sv_zh-cn_16k-common",
    "paraformer-zh": "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-zh-streaming": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "sensevoice-small": "iic/SenseVoiceSmall",
}

_HF_ALIASES = {
    "fsmn-vad": "funasr/fsmn-vad",
    "ct-punc": "funasr/ct-punc",
    "cam++": "funasr/campplus",
    "paraformer-zh": "funasr/paraformer-zh",
    "sensevoice-small": "FunAudioLLM/SenseVoiceSmall",
}


def _resolve_model_alias(model_id: str, hub: str) -> str:
    """Resolve shorthand to full model ID."""
    aliases = _MS_ALIASES if hub == "ms" else _HF_ALIASES
    return aliases.get(model_id, model_id)
