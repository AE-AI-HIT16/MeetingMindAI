"""Overlapped turns that only repeat another speaker are detected."""

from meetasr.backend.utils.overlap_echo import is_overlap_echo, overlap_ratio


def test_overlap_ratio():
    assert overlap_ratio(1000, 2000, [(1000, 1500)]) == 0.5
    assert overlap_ratio(1000, 2000, [(0, 5000)]) == 1.0
    assert overlap_ratio(1000, 2000, []) == 0.0


def test_echo_of_concurrent_speaker_is_detected():
    main = "theo phó chủ tịch ủy ban an toàn giao thông quốc gia"
    assert is_overlap_echo("ủy ban an toàn giao thông", 0.9, [main])


def test_real_interjection_is_kept():
    main = "theo phó chủ tịch ủy ban an toàn giao thông quốc gia"
    assert not is_overlap_echo("dạ vâng ạ", 0.9, [main])


def test_mostly_clean_turn_is_never_dropped():
    main = "ủy ban an toàn giao thông"
    assert not is_overlap_echo("ủy ban an toàn giao thông", 0.3, [main])


def test_crosstalk_flag_needs_300ms_inside_the_sentence():
    from meetasr.backend.api.schemas_phase2 import SpeakerTurn
    from meetasr.backend.realtime.job_worker import _has_crosstalk
    from meetasr.backend.schemas import SentenceInfo

    turn = SpeakerTurn(0, 10000, 0, overlaps=[(2000, 2500), (8000, 8100)])
    assert _has_crosstalk(SentenceInfo(text="a", start=1.0, end=3.0), turn)
    assert not _has_crosstalk(SentenceInfo(text="b", start=7.0, end=9.0), turn)
    assert not _has_crosstalk(SentenceInfo(text="c", start=1.0, end=3.0), None)
