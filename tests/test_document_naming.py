"""Tests for human-friendly document and realtime source names."""

from datetime import datetime, timezone

from meetasr.utils.document_naming import (
    export_title,
    source_display_name,
    source_document_label,
    source_export_label,
)

CREATED_AT = datetime(2026, 8, 4, 0, 23, tzinfo=timezone.utc)


def test_uploaded_source_keeps_original_display_name_and_drops_extension_on_export():
    assert source_display_name("6 người.wav", CREATED_AT) == "6 người.wav"
    assert source_export_label("6 người.wav", CREATED_AT) == "6 người"
    assert source_document_label("6 người.wav", CREATED_AT) == "6 người"
    assert export_title("6 người.wav", CREATED_AT, "full_text") == (
        "Toàn văn - 6 người"
    )


def test_realtime_source_uses_local_human_friendly_names():
    filename = "realtime_2026-08-04T00:23:14.123456.wav"

    assert source_display_name(filename, CREATED_AT) == (
        "Ghi âm trực tiếp · 04/08/2026 07:23"
    )
    assert source_export_label(filename, CREATED_AT) == (
        "Ghi âm 04-08-2026 07-23"
    )
    assert source_document_label(filename, CREATED_AT) == (
        "Ghi âm trực tiếp · 04/08/2026 07:23"
    )
    assert export_title(filename, CREATED_AT, "full_text") == (
        "Toàn văn - Ghi âm 04-08-2026 07-23"
    )
