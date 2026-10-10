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
