"""Run-level punctuation keeps words and segment count."""

from meetasr.runpod.schemas import SentenceInfo
from meetasr.runpod.utils.punctuation import needs_punctuation, punctuate_speaker_runs


def _s(text, speaker):
    return SentenceInfo(text=text, start=0.0, end=1.0, speaker=speaker)


def fake_restore(text):
    # Capitalize the first word and end with a full stop, like ViBERT.
    words = text.split()
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def test_same_speaker_segments_are_punctuated_together():
    out = punctuate_speaker_runs(
        [_s("xin chào các bạn", 0), _s("hôm nay mình họp", 0), _s("vâng", 1)],
        fake_restore,
    )
    # One call for the run: the cut between segments gets no full stop.
    assert [s.text for s in out] == ["Xin chào các bạn", "hôm nay mình họp.", "Vâng."]


def test_well_punctuated_runs_are_left_alone():
    calls = []
    out = punctuate_speaker_runs([_s("Chào anh.", 0)], lambda t: calls.append(t) or t)
    assert calls == [] and out[0].text == "Chào anh."


def test_changed_words_fall_back_to_original_text():
    out = punctuate_speaker_runs([_s("xin chào", 0)], lambda t: "Xin chao.")
    assert out[0].text == "xin chào"


def test_qwen_name_capitals_are_kept():
    out = punctuate_speaker_runs([_s("gặp anh Tuấn ở hà nội", 0)], fake_restore)
    assert out[0].text == "Gặp anh Tuấn ở hà nội."


def test_needs_punctuation():
    assert needs_punctuation("xin chào")
    assert needs_punctuation("xin chào.")
    assert not needs_punctuation("Xin chào.")
    assert not needs_punctuation("")
