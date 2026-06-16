"""AutoModel — load a single model component from name or local path."""

from __future__ import annotations

import logging
from typing import Any

from meetasr.register import tables
from meetasr.utils.download import download_model as _download


class AutoModel:
    """Load any registered MeetASR model by name.

    Downloads model weights if not already cached, reads config.yaml,
    looks up the model class in the registry, and returns an instance.

    Example:
        >>> vad = AutoModel(model="fsmn-vad", device="cpu")
        >>> asr = AutoModel(model="sensevoice-small", device="cuda:0")
        >>> asr.recognize("audio.wav")
    """

    def __new__(cls, model: str, hub: str = "ms", device: str = "cpu", **kwargs):
        """Build and return the model instance (not an AutoModel wrapper).

        Args:
            model: Model name (e.g. "fsmn-vad") or full ID or local path.
            hub: "ms" (ModelScope) or "hf" (HuggingFace).
            device: Torch device string.
            **kwargs: Config overrides (merged with config.yaml values).

        Returns:
            Instantiated model object (AbsVAD, AbsASR, AbsPunc, or AbsSpk).

        Raises:
            RuntimeError: If model is not registered or weights not found.
        """
        config = _download(model=model, hub=hub)
        config.update(kwargs)
        config["device"] = device

        model_key = config.get("model", model)
        model_class = tables.model_classes.get(model_key)

        if model_class is None:
            registered = tables.list_registered("model_classes")
            raise RuntimeError(
                f"Model '{model_key}' is not registered.\n"
                f"Registered models: {registered}\n"
                "Make sure the model's module is imported in meetasr/__init__.py"
            )

        logging.info(f"Building model '{model_key}' ({model_class.__name__}) on {device}")
        instance = model_class(**config)
        return instance
