# Technical Constraints — MeetASR

## 1. Runtime Requirements

### Python & Dependencies

```
Python >= 3.10         (uses match statements, modern typing syntax)
PyTorch >= 2.0         (compile API, better performance)
CUDA >= 11.8           (optional — for GPU acceleration)
```

### Hardware Requirements

| Tier | RAM | VRAM | Processing Speed |
|---|---|---|---|
| **Minimum** (CPU only) | 8 GB | N/A | ~1-3x real-time |
| **Recommended** (CPU) | 16 GB | N/A | ~17x real-time |
| **GPU** | 16 GB | 4 GB | ~170x real-time |

### Supported OS

- Windows 10/11 (x64)
- Ubuntu 20.04+ (x64, ARM64)
- macOS 12+ (Apple Silicon via MPS)

---

## 2. Model Weight Constraints

### Model Weight Sources

- ✅ ModelScope (ms) — preferred due to faster download speeds in Vietnam
- ✅ HuggingFace (hf) — fallback source
- ❌ No model re-training from scratch (out of scope)
- ❌ No model fine-tuning (out of scope for MVP)

### Cache Location

```
Windows: %USERPROFILE%/.cache/modelscope/hub/
Linux:   ~/.cache/modelscope/hub/
         or set MODELSCOPE_CACHE=/custom/path
```

### Model Sizes

| Model | Size on Disk | RAM Usage (Loaded) |
|---|---|---|
| fsmn-vad | ~2 MB | ~50 MB |
| SenseVoiceSmall | ~234 MB | ~500 MB |
| Paraformer-zh | ~220 MB | ~600 MB |
| ct-punc | ~290 MB | ~700 MB |
| cam++ | ~7 MB | ~100 MB |
| **Total (SenseVoice stack)** | **~533 MB** | **~1.3 GB** |

---

## 3. Audio Processing Constraints

```python
# System invariants that must hold true:

SAMPLE_RATE = 16000        # Hz — all models require 16kHz
MAX_AUDIO_DURATION = 14400 # seconds (4 hours)
MIN_AUDIO_DURATION = 0.1   # seconds
DTYPE = np.float32         # audio array dtype
CHANNEL = 1                # mono only (auto-convert)

# VAD constraints
VAD_MAX_SEGMENT_MS = 60000   # max 60s per segment (configurable)
VAD_MIN_SEGMENT_MS = 200     # ignore segments < 200ms
VAD_PADDING_MS = 100         # padding before/after each segment
```

---

## 4. LLM Constraints

### Context Window Management

```
Max input characters per LLM call: 8000 chars (~6000 tokens)
→ 1 hour audio ≈ 8000-15000 chars transcript
→ Meetings longer than 30 minutes require chunking
```

### Supported LLM Providers

| Provider | Requires | Notes |
|---|---|---|
| OpenAI | API key | Best performance, lowest latency |
| Azure OpenAI | API key + endpoint | Enterprise deployments |
| Ollama | Local install | Free, privacy-first |
| LM Studio | Local install | Windows-friendly local runner |
| Any OpenAI-compatible | API key | Groq, Together, OpenRouter, etc. |

### LLM Output Constraints

```python
# All LLM calls must return valid JSON or plain text.
# The system must never crash if the LLM returns an invalid format.
# Always fallback to default values on failure:

DEFAULT_FALLBACK = {
    "summary": "[Unable to generate summary]",
    "topics": [],
    "action_items": [],
    "decisions": [],
}
```

---

## 5. API Constraints

```python
MAX_FILE_SIZE = 500 * 1024 * 1024   # 500 MB
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".webm"}
REQUEST_TIMEOUT = 600               # 10 minutes (for long meetings)
MAX_CONCURRENT_REQUESTS = 3         # local single-user limit
```

---

## 6. Performance Targets

| Metric | Target |
|---|---|
| Audio load time | < 0.5s for a 60-minute file |
| VAD processing | < 2s for 60-minute audio |
| ASR (CPU, SenseVoice) | ≤ 4x real-time (60 mins → ≤ 4 mins execution) |
| ASR (GPU, SenseVoice) | ≤ 0.5x real-time (60 mins → ≤ 30s execution) |
| LLM summarize (60 mins) | ≤ 30s execution |
| Total E2E (CPU) | ≤ 6 mins execution for a 60-minute meeting |

---

## 7. Error Handling Constraints

```python
# All public APIs must:
# 1. Never crash with RuntimeError or segfault.
# 2. Return a MeetingReport with whatever information is successfully recovered.
# 3. Log a warning rather than raise an exception for input edge cases.

# Edge cases that must be handled:
# - Empty audio file (0 bytes)
# - Audio containing only silence (no speech detected)
# - Audio with extremely high noise (no segments detected by VAD)
# - LLM timeout or rate limit
# - Corrupt model weights
# - Disk full during model download
```

---

## 8. Threading & Concurrency

```python
# Inference: single-threaded per request (torch is not thread-safe with model state)
# Multiple requests: process pool or request queue
# LLM calls: asyncio-friendly (use httpx async clients)

# Thread safety rules:
# - Model objects MUST NOT be shared across requests
# - Registry (tables) is read-only after startup → thread-safe
# - Config dicts are copied per request
```
