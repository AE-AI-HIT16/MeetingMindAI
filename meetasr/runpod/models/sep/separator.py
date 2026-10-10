"""Recover one speaker's voice where two people talk at once.

Qwen3-ASR on overlapped audio transcribes the louder voice (or a blend), so a
"dạ vâng" said over someone else is lost. MossFormer2 splits the overlapped
region into two streams; the stream whose cam++ embedding is closest to the
target speaker's voice replaces the mixture in that region only.
"""

from __future__ import annotations

import logging
import types
from typing import Callable

import numpy as np

from meetasr.runpod.models.abs_models import AbsSeparator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_ARGS = types.SimpleNamespace(
    encoder_embedding_dim=512,
    mossformer_sequence_dim=512,
    num_mossformer_layer=24,
    encoder_kernel_size=16,
    num_spks=2,
)


class SpeechSeparator(AbsSeparator):
    """MossFormer2 2-speaker separation applied to selected regions."""

    def __init__(
        self,
        model: str = "alibabasglab/MossFormer2_SS_16K",
        filename: str = "last_best_checkpoint.pt",
        device: str = "cpu",
        window_s: float = 4.0,
        context_s: float = 0.3,
        min_region_s: float = 0.2,
        crossfade_s: float = 0.02,
        verify_speakers: bool = True,
        **_: object,
    ):
        """
        Args:
            model: Local checkpoint path or Hugging Face repo id.
            filename: Checkpoint file inside the repo.
            window_s: Longest piece separated at once; longer regions are
                split and the target stream is chosen per piece (stream order
                is not stable across pieces).
            context_s: Audio kept around a region so the model hears both
                voices before they overlap.
            min_region_s: Shorter overlaps are left as they are.
            crossfade_s: Fade between the mixture and the separated stream.
            verify_speakers: Re-check who speaks in mostly-overlapped turns
                from the separated voices (MeetPipeline).
        """
        self.model = model
        self.filename = filename
        self.device = device
        self.window_s = window_s
        self.context_s = context_s
        self.min_region_s = min_region_s
        self.crossfade_s = crossfade_s
        self.verify_speakers = verify_speakers
        self._net = None

    def _ensure_loaded(self) -> None:
        if self._net is not None:
            return
        import os

        import torch

        from meetasr.runpod.models.sep.mossformer2.mossformer2 import MossFormer2_SS_16K

        path = self.model
        if not os.path.isfile(path):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self.model, self.filename)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint = checkpoint.get("model", checkpoint)
        net = MossFormer2_SS_16K(_ARGS).model
        state = net.state_dict()
        loaded = 0
        for key in state:
            for candidate in (key, key.replace("module.", ""), "module." + key):
                if candidate in checkpoint and checkpoint[candidate].shape == state[key].shape:
                    state[key] = checkpoint[candidate]
                    loaded += 1
                    break
        if loaded != len(state):
            raise RuntimeError(f"MossFormer2 checkpoint matched {loaded}/{len(state)} tensors")
        net.load_state_dict(state)
        self._net = net.eval().to(self.device)
        logger.info("Speech separator loaded: %s on %s", path, self.device)

    def separate(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split mixed speech into two streams.

        Args:
            audio: Float32 mono audio at 16kHz.

        Returns:
            Two streams with the input's length and loudness.
        """
        import torch

        self._ensure_loaded()
        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None].to(self.device)
            outputs = self._net(tensor)
        level = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
        streams = []
        for output in outputs:
            stream = output[0].detach().float().cpu().numpy()[: len(audio)]
            stream = stream / float(np.sqrt(np.mean(np.square(stream)) + 1e-12)) * level
            streams.append(stream.astype(np.float32))
        return streams

    def extract(
        self,
        chunk: np.ndarray,
        regions_ms: list[tuple[int, int]],
        target_embedding: np.ndarray,
        embed_batch: Callable[[list[np.ndarray]], "object"],
    ) -> np.ndarray:
        """Replace each overlapped region of ``chunk`` with the target voice.

        Args:
            chunk: Float32 mono audio at 16kHz (one speaker turn).
            regions_ms: Overlapped ``(start_ms, end_ms)`` ranges in ``chunk``.
            target_embedding: The turn speaker's mean cam++ embedding.
            embed_batch: cam++ batch embedder used to pick the stream.

        Returns:
            A copy of ``chunk`` with the target voice in those regions.
        """
        output = chunk.astype(np.float32, copy=True)
        target = np.asarray(target_embedding, dtype=np.float32)
        target /= max(float(np.linalg.norm(target)), 1e-8)
        window = int(self.window_s * SAMPLE_RATE)
        context = int(self.context_s * SAMPLE_RATE)
        fade = int(self.crossfade_s * SAMPLE_RATE)

        for start_ms, end_ms in regions_ms:
            start = max(0, int(start_ms * SAMPLE_RATE / 1000))
            end = min(len(chunk), int(end_ms * SAMPLE_RATE / 1000))
            if end - start < self.min_region_s * SAMPLE_RATE:
                continue
            piece_start = start
            while piece_start < end:
                piece_end = min(end, piece_start + window)
                a = max(0, piece_start - context)
                b = min(len(chunk), piece_end + context)
                streams = self.separate(chunk[a:b])
                embeddings = embed_batch(streams)
                if hasattr(embeddings, "detach"):
                    embeddings = embeddings.detach().float().cpu().numpy()
                embeddings = np.asarray(embeddings, dtype=np.float32)
                embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
                best = streams[int(np.argmax(embeddings @ target))]
                voice = best[piece_start - a:piece_end - a]
                _blend(output, voice, piece_start, fade)
                piece_start = piece_end
        return output


def _blend(output: np.ndarray, voice: np.ndarray, start: int, fade: int) -> None:
    """Write ``voice`` at ``start`` with linear fades into the mixture."""
    length = len(voice)
    if length == 0:
        return
    weight = np.ones(length, dtype=np.float32)
    ramp = min(fade, length // 2)
    if ramp > 0:
        weight[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        weight[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
    segment = output[start:start + length]
    output[start:start + length] = segment * (1.0 - weight) + voice * weight
