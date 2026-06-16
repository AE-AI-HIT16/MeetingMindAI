# Coding Style & Conventions — MeetASR

> Based on `claude.md`: Think Before Coding · Simplicity First · Surgical Changes · Goal-Driven

---

## 1. Python Style

### Type Hints — Mandatory

```python
# ✅ CORRECT
def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    ...

def detect_segments(audio: np.ndarray) -> list[Segment]:
    ...

# ❌ INCORRECT — missing type hints
def load_audio(path, sr=16000):
    ...
```

### Docstrings — Mandatory for Public APIs

```python
# ✅ CORRECT
def transcribe(self, audio: str | np.ndarray, **kwargs) -> TranscriptResult:
    """Transcribe audio to text.

    Args:
        audio: File path (str) or audio samples (np.ndarray at 16kHz).
        **kwargs: language (str), batch_size (int).

    Returns:
        TranscriptResult with text, timestamps, and sentence_info.

    Raises:
        FileNotFoundError: If audio path does not exist.
        ValueError: If audio array has wrong dtype or sample rate.
    """
```

### Naming Conventions

```python
# Classes: PascalCase
class MeetPipeline: ...
class FsmnVAD: ...

# Functions/methods: snake_case
def load_audio(): ...
def detect_segments(): ...

# Constants: UPPER_SNAKE_CASE
SAMPLE_RATE = 16000
MAX_SEGMENT_MS = 60000

# Private: prefix with underscore
def _format_transcript(self): ...
self._model = None

# File names: snake_case.py
# abs_vad.py, fsmn_vad.py, auto_model.py
```

---

## 2. File Organization Rules

### Single Responsibility Rule

```
# ✅ CORRECT
fsmn_vad.py      → contains only FsmnVAD class
abs_vad.py       → contains only AbsVAD abstract class
cluster.py       → contains only clustering logic

# ❌ INCORRECT
models.py        → contains VAD + ASR + Punc + Spk (bloated)
utils.py         → dump-all file for uncategorized logic
```

### Max File Length

```
Model files:    ≤ 300 lines
Utility files:  ≤ 200 lines
Pipeline:       ≤ 400 lines
Tests:          unlimited
```

---

## 3. Abstract Base Classes

Every major component must implement an abstract base class:

```python
# meetasr/models/vad/abs_vad.py
from abc import ABC, abstractmethod
import numpy as np
from meetasr.schemas import Segment

class AbsVAD(ABC):
    """Abstract Voice Activity Detection."""

    @abstractmethod
    def detect(self, audio: np.ndarray, **kwargs) -> list[Segment]:
        """Detect speech segments.

        Args:
            audio: Audio samples at 16kHz, float32.
            **kwargs: Model-specific parameters.

        Returns:
            List of (start_ms, end_ms) segments.
        """
        ...
```

---

## 4. Config Pattern

```python
# ✅ CORRECT — config-driven, no hardcoding
class FsmnVAD(AbsVAD):
    def __init__(
        self,
        max_single_segment_time: int = 60000,
        min_duration_on: int = 200,
        **kwargs,
    ):
        self.max_segment_ms = max_single_segment_time
        self.min_duration_on = min_duration_on

# ❌ INCORRECT — hardcoded in logic
class FsmnVAD(AbsVAD):
    def detect(self, audio):
        if len(audio) > 960000:   # hardcoded ← BAD
            ...
```

---

## 5. Error Handling

```python
# ✅ CORRECT — specific exception, helpful message
def load_audio(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.endswith(SUPPORTED_FORMATS):
        raise ValueError(f"Unsupported format. Got: {path!r}, expected: {SUPPORTED_FORMATS}")

# ✅ CORRECT — log warning for recoverable errors
def _safe_llm_call(self, prompt: str) -> dict:
    try:
        return self._llm.chat(prompt)
    except Exception as e:
        logging.warning(f"LLM call failed: {e}. Using empty result.")
        return {}

# ❌ INCORRECT — silent catch-all exception
try:
    result = model.inference(audio)
except:
    pass   # ← NEVER DO THIS
```

---

## 6. Testing Conventions

### File Naming

```
tests/
├── test_audio_utils.py      ← tests utils/audio.py
├── test_fsmn_vad.py         ← tests models/vad/fsmn_vad.py
├── test_pipeline.py         ← tests pipeline.py
├── test_llm_summarizer.py   ← tests llm/summarizer.py
├── test_api.py              ← tests api/
└── fixtures/
    ├── sample_vi.wav        ← 10s Vietnamese sample
    ├── sample_2spk.wav      ← 2-speaker sample
    └── sample_silent.wav    ← silent audio (edge case)
```

### Test Pattern

```python
# tests/test_fsmn_vad.py

import pytest
import numpy as np

def test_detect_returns_segments(vad_model, sample_audio):
    """VAD must return list of segments for audio with speech."""
    segments = vad_model.detect(sample_audio)
    assert isinstance(segments, list)
    assert len(segments) > 0
    assert all(s.start_ms < s.end_ms for s in segments)

def test_detect_silent_audio(vad_model):
    """VAD must return empty list for silent audio."""
    silent = np.zeros(16000, dtype=np.float32)
    segments = vad_model.detect(silent)
    assert segments == []

@pytest.fixture
def vad_model():
    from meetasr.models.vad.fsmn_vad import FsmnVAD
    return FsmnVAD()   # uses pre-cached model weights
```

---

## 7. Import Order

```python
# Import ordering (PEP 8 + isort):
# 1. Standard library
import os
import json
import logging
from typing import Optional
from dataclasses import dataclass

# 2. Third-party
import numpy as np
import torch
from omegaconf import OmegaConf

# 3. Local (meetasr)
from meetasr.register import tables
from meetasr.schemas import TranscriptResult
from meetasr.utils.audio import load_audio
```

---

## 8. Logging

```python
# Use standard logging, not print()
import logging

logger = logging.getLogger(__name__)

# Levels:
logger.debug("Detailed debug info")          # internal state
logger.info("VAD: found 12 segments")        # progress info
logger.warning("LLM call failed, retrying")  # recoverable
logger.error("Model weights not found")       # non-recoverable

# ❌ Do not use print() in library code
print("Processing...")   # ← BAD
```

---

## 9. Git Conventions

```
# Branch naming
feature/add-ollama-client
fix/vad-empty-audio-crash
docs/update-api-spec

# Commit messages (imperative, English)
feat: add OllamaClient for local LLM support
fix: handle empty audio in FsmnVAD.detect()
docs: add LLM integration design doc
test: add unit tests for MeetingSummarizer
refactor: split AutoModel into Pipeline + Loader
```
