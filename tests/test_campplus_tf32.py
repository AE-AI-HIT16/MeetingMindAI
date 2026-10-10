"""CAM++ embeddings must be computed without cuDNN TF32 (diarization stability)."""

from __future__ import annotations

import numpy as np
import torch

from meetasr.runpod.models.spk.campplus import CAMPlusPlus


def test_embedding_runs_with_tf32_disabled_and_restores_flag():
    seen = []

    class FakeInner:
        def generate(self, **kwargs):
            seen.append(torch.backends.cudnn.allow_tf32)
            n = len(kwargs["input"]) if isinstance(kwargs["input"], list) else 1
            return [{"spk_embedding": torch.zeros(n, 192)}]

    model = CAMPlusPlus(model_path="unused", device="cpu")
    model._inner = FakeInner()
    torch.backends.cudnn.allow_tf32 = True

    model.embed(np.zeros(24000, dtype=np.float32))
    embeddings = model.embed_batch([np.zeros(24000, dtype=np.float32)] * 3, batch_size=2)

    assert seen == [False, False, False]
    assert torch.backends.cudnn.allow_tf32 is True
    assert embeddings.shape == (3, 192)
