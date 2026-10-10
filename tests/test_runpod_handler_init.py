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
