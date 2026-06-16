# API Specification — MeetASR

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### 1. Health Check

```
GET /v1/health
```

**Response 200:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "models_loaded": ["sensevoice-small", "fsmn-vad", "ct-punc"],
  "llm_available": true
}
```

---

### 2. Transcribe Audio

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

**Request Body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | ✅ | Audio file (wav, mp3, m4a, mp4, flac) |
| `model` | string | ❌ | `sensevoice` (default) \| `paraformer` \| `fun-asr-nano` |
| `language` | string | ❌ | `auto` (default) \| `vi` \| `zh` \| `en` |
| `response_format` | string | ❌ | `json` (default) \| `text` \| `srt` \| `verbose_json` |
| `speaker_diarization` | bool | ❌ | `false` (default) — enables speaker detection |
| `timestamp_granularity` | string | ❌ | `segment` (default) \| `word` |

**Response `json`:**
```json
{
  "text": "Hello everyone. Today we will discuss the Q3 plan."
}
```

**Response `verbose_json`:**
```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 125.4,
  "text": "Hello everyone...",
  "segments": [
    {
      "id": 0,
      "start": 0.5,
      "end": 3.2,
      "text": "Hello everyone.",
      "speaker": 0,
      "tokens": [...]
    }
  ]
}
```

**Response `srt`:**
```
1
00:00:00,500 --> 00:00:03,200
[Speaker 0] Hello everyone.

2
00:00:04,100 --> 00:00:08,500
[Speaker 1] Thank you for inviting me.
```

---

### 3. Meeting Summarize (Full Pipeline)

```
POST /v1/meeting/summarize
Content-Type: multipart/form-data
```

**Request Body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | ✅ | Audio file |
| `language` | string | ❌ | `vi` (default) \| `en` \| `zh` |
| `llm_model` | string | ❌ | `gpt-4o-mini` (default), or any specified model |
| `asr_model` | string | ❌ | `sensevoice` (default) |
| `include_transcript` | bool | ❌ | `true` (default) |
| `include_topics` | bool | ❌ | `true` (default) |
| `include_actions` | bool | ❌ | `true` (default) |
| `include_decisions` | bool | ❌ | `true` (default) |

**Response 200:**
```json
{
  "meeting_id": "meet_20240616_093045",
  "language": "en",
  "duration": 3654.2,
  "processing_time": 45.3,
  "transcript": {
    "text": "Hello everyone...",
    "sentence_info": [
      {
        "text": "Hello everyone.",
        "start": 0.5,
        "end": 3.2,
        "speaker": 0
      }
    ]
  },
  "summary": "The meeting discussed product development plans for Q3 2024. The team agreed to focus on three core features: Feature A, B, and C. Deadlines for each feature were clearly defined.",
  "topics": [
    {
      "title": "Q3 2024 Planning",
      "description": "Discussion regarding the product roadmap and task delegation",
      "start_time": 0.0,
      "end_time": 900.0
    }
  ],
  "action_items": [
    {
      "task": "Design UI for feature A",
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

### 4. Errors

```json
// 400 Bad Request
{
  "error": {
    "code": "invalid_file_format",
    "message": "File format not supported. Supported: wav, mp3, m4a, mp4, flac"
  }
}

// 413 Payload Too Large
{
  "error": {
    "code": "file_too_large",
    "message": "File size exceeds maximum limit of 500MB"
  }
}

// 503 Service Unavailable
{
  "error": {
    "code": "model_not_loaded",
    "message": "ASR model is not loaded. Call GET /v1/health to check status."
  }
}
```

---

## Rate Limits (Phase 2)

- Single-user local: Unlimited
- Multi-tenant: 10 requests/minute per API key

## File Size Limits

| Type | Limit |
|---|---|
| Max file size | 500 MB |
| Max audio duration | 4 hours |
| Recommended max | 60 minutes (for optimal LLM summary quality) |
