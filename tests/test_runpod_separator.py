"""Overlap separation splices the target speaker's stream into the chunk."""

import numpy as np

from meetasr.runpod.models.sep.separator import SpeechSeparator


class FakeSeparator(SpeechSeparator):
    def separate(self, audio):
        # stream 0 = constant 1.0 ("other"), stream 1 = constant 2.0 ("target")
        return [np.full(len(audio), 1.0, np.float32), np.full(len(audio), 2.0, np.float32)]


def _embed(streams):
    # The stream with value 2.0 looks like the target voice [0, 1].
    return np.array([[1.0, 0.0] if s[0] < 1.5 else [0.0, 1.0] for s in streams])


def test_only_overlapped_region_is_replaced_by_target_stream():
    chunk = np.zeros(16000 * 3, dtype=np.float32)
    separator = FakeSeparator(crossfade_s=0.0, window_s=0.5)

    out = separator.extract(chunk, [(1000, 2000)], np.array([0.0, 1.0]), _embed)

    assert np.all(out[:16000] == 0)
    assert np.allclose(out[16000:32000], 2.0)
    assert np.all(out[32000:] == 0)


def test_tiny_overlaps_are_left_alone():
    chunk = np.zeros(16000, dtype=np.float32)
    separator = FakeSeparator(min_region_s=0.2)
    out = separator.extract(chunk, [(100, 200)], np.array([0.0, 1.0]), _embed)
    assert np.all(out == 0)
