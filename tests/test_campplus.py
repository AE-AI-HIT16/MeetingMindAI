"""Safety tests for CAM++ clustering failures."""

from __future__ import annotations

import pytest
import torch

from meetasr.models.spk import cluster as cluster_module
from meetasr.models.spk.campplus import CAMPlusPlus


def test_clustering_failure_does_not_claim_everyone_is_speaker_zero(
    monkeypatch,
) -> None:
    class FailingClustering:
        def __init__(self, **kwargs):
            del kwargs

        def __call__(self, embeddings, **kwargs):
            del embeddings, kwargs
            raise RuntimeError("clustering unavailable")

    monkeypatch.setattr(cluster_module, "CommonClustering", FailingClustering)
    model = CAMPlusPlus()

    with pytest.raises(RuntimeError, match="Speaker clustering failed"):
        model.cluster(torch.ones(3, 192))
