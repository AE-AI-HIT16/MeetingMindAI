"""Data schemas for MeetASR inputs and outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# ASR Output
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A speech segment detected by VAD."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def start_s(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_s(self) -> float:
        return self.end_ms / 1000.0


@dataclass
class SentenceInfo:
    """A single sentence with timing and speaker information."""

    text: str
    start: float          # seconds
    end: float            # seconds
    speaker: Optional[int] = None
    # char-level timestamps [[start_ms, end_ms], ...]
    char_timestamps: list[list[int]] = field(default_factory=list)


@dataclass
class TranscriptResult:
    """Full transcription result for one audio file."""

    key: str
    text: str
    duration: float       # seconds
    language: str = "vi"
    sentence_info: list[SentenceInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-serializable dict."""
        return asdict(self)

    def to_srt(self) -> str:
        """Format as SRT subtitle string."""
        lines = []
        for i, s in enumerate(self.sentence_info, start=1):
            start = _seconds_to_srt(s.start)
            end = _seconds_to_srt(s.end)
            spk = f"[Speaker {s.speaker}] " if s.speaker is not None else ""
            lines.append(f"{i}\n{start} --> {end}\n{spk}{s.text}\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Meeting Report
# ---------------------------------------------------------------------------

@dataclass
class Topic:
    """A main topic/agenda item discussed in the meeting."""

    title: str
    description: str
    start_time: float = 0.0   # seconds
    end_time: float = 0.0     # seconds


@dataclass
class ActionItem:
    """A task assigned during the meeting."""

    task: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    priority: str = "medium"           # high | medium | low
    mentioned_by: Optional[str] = None
    timestamp: Optional[float] = None  # seconds


@dataclass
class Decision:
    """A decision made during the meeting."""

    content: str
    made_by: str = "Team"
    timestamp: Optional[float] = None  # seconds


@dataclass
class MeetingReport:
    """Full meeting report combining transcript and LLM analysis."""

    transcript: TranscriptResult
    summary: str = ""
    topics: list[Topic] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    language: str = "vi"
    asr_model: str = ""
    llm_model: str = ""
    processing_time: float = 0.0      # seconds

    def to_dict(self) -> dict:
        """Serialize to JSON-serializable dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to formatted JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        """Format as a human-readable Markdown meeting report."""
        from meetasr.utils.misc import seconds_to_human

        lines = [
            f"# Báo cáo Cuộc họp",
            f"",
            f"**Thời lượng:** {seconds_to_human(self.transcript.duration)}  ",
            f"**Ngôn ngữ:** {self.language}  ",
            f"**Mô hình ASR:** {self.asr_model}  ",
            f"**Mô hình LLM:** {self.llm_model}  ",
            f"",
            f"---",
            f"",
        ]

        if self.summary:
            lines += ["## Tóm tắt", "", self.summary, "", "---", ""]

        if self.topics:
            lines += ["## Đề mục chính", ""]
            for i, t in enumerate(self.topics, 1):
                ts = f"({seconds_to_human(t.start_time)} - {seconds_to_human(t.end_time)})"
                lines.append(f"### {i}. {t.title} {ts}")
                lines.append(t.description)
                lines.append("")
            lines += ["---", ""]

        if self.action_items:
            lines += ["## Action Items", ""]
            lines.append("| # | Công việc | Người phụ trách | Deadline | Ưu tiên |")
            lines.append("|---|---|---|---|---|")
            prio_icon = {"high": "🔴 Cao", "medium": "🟡 TB", "low": "🟢 Thấp"}
            for i, a in enumerate(self.action_items, 1):
                lines.append(
                    f"| {i} | {a.task} | {a.assignee or '-'} "
                    f"| {a.deadline or '-'} | {prio_icon.get(a.priority, a.priority)} |"
                )
            lines += ["", "---", ""]

        if self.decisions:
            lines += ["## Quyết định", ""]
            for i, d in enumerate(self.decisions, 1):
                ts = f" *(tại {seconds_to_human(d.timestamp)})*" if d.timestamp else ""
                lines.append(f"{i}. **{d.content}** — {d.made_by}{ts}")
            lines += ["", "---", ""]

        if self.transcript.sentence_info:
            lines += ["## Transcript", ""]
            for s in self.transcript.sentence_info:
                ts = f"`[{seconds_to_human(s.start)} → {seconds_to_human(s.end)}]`"
                spk = f"Speaker {s.speaker}: " if s.speaker is not None else ""
                lines.append(f"{ts} {spk}{s.text}  ")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seconds_to_srt(seconds: float) -> str:
    """Convert seconds float to SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
