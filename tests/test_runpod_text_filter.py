"""Non-speech ASR output is dropped or cut."""

from meetasr.runpod.utils.text_filter import clean_transcript_text


def test_sound_tags_and_laughter_are_dropped():
    for text in ["Applause.", "applause", "haha", "Hahaha!", "Tiếng vỗ tay"]:
        assert clean_transcript_text(text, 2.0) == ""


def test_normal_speech_is_kept():
    text = "Chúng ta bắt đầu họp nhé."
    assert clean_transcript_text(text, 2.0) == text
    assert clean_transcript_text("Dạ.", 0.4) == "Dạ."


def test_repetition_loops_are_cut():
    assert clean_transcript_text("bố bố bố bố bố bố", 3.0) == "bố"
    looped = "cảm ơn " * 10
    assert clean_transcript_text(looped, 10.0) == "cảm ơn"
    assert clean_transcript_text("vâng vâng vâng", 1.5) == "vâng vâng vâng"


def test_impossible_speaking_rate_is_dropped():
    text = " ".join(f"từ{i}" for i in range(40))
    assert clean_transcript_text(text, 2.0) == ""
