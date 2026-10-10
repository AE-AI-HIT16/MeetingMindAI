"""An optional model that fails to load must not stop the RunPod worker."""

from types import SimpleNamespace


class _Component:
    def __init__(self, fail: bool):
        self.fail = fail
        self.loaded = False

    def _ensure_loaded(self) -> None:
        if self.fail:
            raise RuntimeError("download failed")
        self.loaded = True


def test_failed_optional_model_is_disabled_not_fatal():
    from meetasr.runpod import handler

    pipe = SimpleNamespace(punc=_Component(False), segmenter=_Component(True), separator=None)

    handler._load_optional_components(pipe)

    assert pipe.punc.loaded
    assert pipe.segmenter is None
    assert "download failed" in handler._load_stats["segmenter_load_error"]


def test_only_unused_qwen_asr_caches_are_stale(tmp_path):
    from meetasr.runpod import handler

    for name in (
        "models--Qwen--Qwen3-ASR-0.6B",
        "models--Qwen--Qwen3-ASR-1.7B",
        "models--dragonSwing--vibert-capu",
        "models--onnx-community--pyannote-segmentation-3.0",
    ):
        (tmp_path / name).mkdir()

    stale = handler._stale_asr_caches(str(tmp_path), "models--Qwen--Qwen3-ASR-1.7B")

    assert [p.rsplit("/", 1)[-1] for p in stale] == ["models--Qwen--Qwen3-ASR-0.6B"]
    assert handler._stale_asr_caches(str(tmp_path / "missing"), "x") == []
