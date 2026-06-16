"""Tests for schemas serialization."""

import json
from meetasr.schemas import (
    Segment, SentenceInfo, TranscriptResult,
    Topic, ActionItem, Decision, MeetingReport
)


def make_transcript() -> TranscriptResult:
    return TranscriptResult(
        key="test_meeting",
        text="Xin chào. Hôm nay chúng ta thảo luận về kế hoạch Q3.",
        duration=10.5,
        language="vi",
        sentence_info=[
            SentenceInfo(text="Xin chào.", start=0.5, end=2.0, speaker=0),
            SentenceInfo(text="Hôm nay chúng ta thảo luận về kế hoạch Q3.", start=2.5, end=8.0, speaker=1),
        ],
    )


class TestSegment:

    def test_duration(self):
        seg = Segment(1000, 3500)
        assert seg.duration_ms == 2500

    def test_seconds_conversion(self):
        seg = Segment(1000, 3500)
        assert seg.start_s == 1.0
        assert seg.end_s == 3.5


class TestTranscriptResult:

    def test_to_dict_is_json_serializable(self):
        result = make_transcript()
        d = result.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        assert "Xin chào" in json_str

    def test_to_srt_format(self):
        result = make_transcript()
        srt = result.to_srt()
        assert "00:00:00,500 --> 00:00:02,000" in srt
        assert "[Speaker 0]" in srt
        assert "Xin chào." in srt


class TestMeetingReport:

    def test_to_dict_is_json_serializable(self):
        transcript = make_transcript()
        report = MeetingReport(
            transcript=transcript,
            summary="Cuộc họp bàn về kế hoạch Q3.",
            topics=[Topic(title="Kế hoạch Q3", description="Thảo luận roadmap", start_time=0, end_time=10)],
            action_items=[ActionItem(task="Thiết kế UI", assignee="Nam", priority="high")],
            decisions=[Decision(content="Dùng React Native", made_by="Nhóm")],
        )
        d = report.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        assert "Kế hoạch Q3" in json_str
        assert "Thiết kế UI" in json_str

    def test_to_markdown_contains_sections(self):
        transcript = make_transcript()
        report = MeetingReport(
            transcript=transcript,
            summary="Tóm tắt ngắn gọn.",
            action_items=[ActionItem(task="Viết tài liệu", priority="medium")],
        )
        md = report.to_markdown()
        assert "# Báo cáo Cuộc họp" in md
        assert "Tóm tắt" in md
        assert "Action Items" in md
