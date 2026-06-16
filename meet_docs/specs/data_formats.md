# Data Formats — MeetASR

## 1. Input Formats

### Audio Files
- **Supported Extensions:** `.wav`, `.mp3`, `.m4a`, `.mp4`, `.flac`, `.ogg`, `.webm`
- **Sample Rate:** Any (automatically resampled to 16kHz)
- **Channels:** Mono or stereo (automatically downmixed to mono)
- **Bit Depth:** 16-bit or 32-bit float

### Batch Input (for CLI)
```
# file.scp (space-separated: key path)
meeting_001 /path/to/meeting1.wav
meeting_002 /path/to/meeting2.mp3

# file.jsonl (one JSON object per line)
{"key": "meeting_001", "source": "/path/to/meeting1.wav"}
{"key": "meeting_002", "source": "https://example.com/audio.wav"}
```

---

## 2. TranscriptResult (ASR Output)

```json
{
  "key": "meeting_001",
  "text": "Hello everyone. Today we are discussing the Q3 plan.",
  "duration": 125.4,
  "language": "en",
  "sentence_info": [
    {
      "text": "Hello everyone.",
      "start": 0.5,
      "end": 3.2,
      "speaker": 0,
      "timestamp": [[500, 800], [900, 1200], [1300, 1800], [1900, 2200], [2300, 3200]]
    },
    {
      "text": "Today we are discussing the Q3 plan.",
      "start": 3.8,
      "end": 7.5,
      "speaker": 0,
      "timestamp": [[3800, 4100], [4200, 4800], ...]
    }
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `key` | string | ID of the audio file (typically filename without extension) |
| `text` | string | Full transcript text containing punctuation |
| `duration` | float | Audio length in seconds |
| `language` | string | Detected or specified language |
| `sentence_info` | array | List of sentences with associated metadata |
| `sentence_info[].text` | string | Sentence text content |
| `sentence_info[].start` | float | Start timestamp (seconds) |
| `sentence_info[].end` | float | End timestamp (seconds) |
| `sentence_info[].speaker` | int\|null | Speaker index (0, 1, 2...) or null if diarization is disabled |
| `sentence_info[].timestamp` | array | Character-level timestamps [[start_ms, end_ms], ...] |

---

## 3. MeetingReport (Full Output)

```json
{
  "meeting_id": "meet_20240616_093045",
  "created_at": "2024-06-16T09:30:45+07:00",
  "language": "en",
  "duration": 3654.2,
  "processing_time": {
    "asr": 23.4,
    "llm": 12.1,
    "total": 35.5
  },
  "asr_model": "iic/SenseVoiceSmall",
  "llm_model": "gpt-4o-mini",
  "speakers": {
    "count": 3,
    "labels": ["Speaker 0", "Speaker 1", "Speaker 2"]
  },
  "transcript": { ... },
  "summary": "The meeting on 2024-06-16 discussed Q3 plans...",
  "topics": [
    {
      "title": "Q3 2024 Planning",
      "description": "Discussion regarding roadmap and milestones",
      "start_time": 0.0,
      "end_time": 900.0
    }
  ],
  "action_items": [
    {
      "task": "Design mockup for Feature A",
      "assignee": "John Doe",
      "deadline": "2024-07-15",
      "priority": "high",
      "mentioned_by": "Speaker 0",
      "timestamp": 456.2
    }
  ],
  "decisions": [
    {
      "content": "Use React Native for mobile app development",
      "made_by": "Team",
      "timestamp": 1234.5
    }
  ]
}
```

---

## 4. SRT Subtitle Format

```srt
1
00:00:00,500 --> 00:00:03,200
[Speaker 0] Hello everyone.

2
00:00:03,800 --> 00:00:07,500
[Speaker 0] Today we are discussing the Q3 plan.

3
00:00:08,200 --> 00:00:11,800
[Speaker 1] I have a few points to share.
```

---

## 5. Markdown Report Format

```markdown
# Meeting Report — 2024-06-16

**Duration:** 60 minutes 54 seconds  
**Number of Speakers:** 3  
**Language:** English

---

## Executive Summary

The meeting on 2024-06-16 outlined plans for Q3 development. The team agreed...

---

## Key Topics

### 1. Q3 2024 Planning (0:00 - 15:00)
Discussion regarding the roadmap and milestones for Q3.

### 2. Work Delegation (15:00 - 35:00)
...

---

## Action Items

| # | Task | Assignee | Deadline | Priority |
|---|---|---|---|---|
| 1 | Design mockup for Feature A | John Doe | 2024-07-15 | 🔴 High |
| 2 | Review backend code | Jane Smith | 2024-07-20 | 🟡 Medium |

---

## Key Decisions

1. **Use React Native for mobile app development** — Team agreement (at 20:34)
2. **Increase Q3 budget by 20%** — Director Tuấn (at 45:12)

---

## Complete Transcript

`[00:00:00 → 00:00:03] Speaker 0: Hello everyone.`  
`[00:00:04 → 00:00:08] Speaker 1: Thank you for inviting me.`  
...
```

---

## 6. Internal Data Structures (Python)

```python
# Internal Timestamp Format
# All timestamps are stored as integer milliseconds in ASR processing.
# Converted to float seconds when exporting to files/APIs.

# VAD Segments Format (from fsmn-vad)
vad_segments: list[list[int]]  # [[start_ms, end_ms], ...]
# Example: [[500, 3200], [4100, 7500], ...]

# ASR result per segment (from model.inference())
{
  "text": "Hello",
  "timestamp": [[0, 200], [300, 500], ...],   # char-level, relative to segment
  "raw_text": "hello",                          # before punctuation (optional)
}

# Speaker embedding (from cam++)
spk_embedding: torch.Tensor  # shape [1, 192]
```
