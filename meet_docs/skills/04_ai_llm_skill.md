# AI LLM Integration Skill — MeetASR

This document outlines the guidelines and technical patterns for integrating Large Language Models (LLMs) into MeetASR. It covers structured prompting, map-reduce architectures for handling long transcripts, and robust fallback strategies.

---

## 1. Principles of Prompt Engineering

To guarantee reliable outputs conforming to structured JSON schemas, prompt designs must be handled systematically.

### 1.1 Structured Prompting Specifications
Prompts stored in the `meetasr/llm/prompts/` directory must adhere to a three-part structure:
1. **Role & Context:** Define the LLM as an expert meeting analyst and transcription assistant.
2. **Task Instructions:** Specify rules for extraction (e.g., forbid inventing information not present in the transcript, preserve original timestamps).
3. **Response Schema:** Enforce raw JSON responses by providing an explicit JSON schema and warning the model against returning introductory or conversational text blocks.

### 1.2 Prompt File Management
Never embed prompt templates directly in Python files.
- Store templates as independent plain text files (`.txt`) in the `meetasr/llm/prompts/` folder.
- Read files dynamically at runtime and format inputs using Python's `.format()` method.

---

## 2. Long Transcript Chunking (Map-Reduce)

Long meeting transcripts can exceed LLM context windows or maximum generation limits. MeetASR implements a Map-Reduce pipeline to handle large files.

```
[Full Transcript] -> [Split into Chunks with Overlap] -> [Summarize per Chunk (Map)] -> [Merge & Re-summarize (Reduce)] -> [Final Report]
```

### 2.1 Overlapping Chunking Strategy
- **Chunk Size:** Divide transcripts into blocks of **8,000 to 12,000 characters** (approx. 2,000 - 3,000 words).
- **Context Overlap:** Maintain an overlap of **1,000 to 1,500 characters** between adjacent blocks to prevent sentence fragmentation at split boundaries.

### 2.2 Map-Reduce Execution
- **Map Step:** Send each chunk independently to the LLM to extract local summaries, action items, and key decisions.
- **Reduce Step:** Aggregate the extracted local reports into a single prompt context. Execute a final consolidation call to synthesize a unified, non-redundant `MeetingReport`.

---

## 3. API Resilience & Error Handling

Design the connection logic to tolerate transient API failures or limits:
- **Exponential Backoff:** If the LLM client encounters Rate Limits (HTTP 429) or Server Errors (HTTP 500), retry the request with an exponentially increasing delay (e.g., 1s, 2s, 4s, 8s).
- **Model Fallback:** If the primary high-tier model (e.g., `gpt-4o`) fails repeatedly, automatically route the request to a fallback lightweight model (e.g., `gpt-4o-mini`).

---

## 4. Reference Test Configurations

### 4.1 Mock-based Summarizer Test
To prevent API costs and ensure fast test execution during CI/CD pipelines, AI Agents **must mock** LLM API calls:

```python
# tests/test_llm_summarizer.py
import pytest
from unittest.mock import MagicMock
from meetasr.llm.summarizer import MeetingSummarizer
from meetasr.schemas import TranscriptResult, SentenceInfo

def test_summarizer_map_reduce_logic():
    # 1. Initialize mock client
    mock_client = MagicMock()
    mock_client.complete.return_value = '{"summary": "Mocked summary", "topics": [], "action_items": [], "decisions": []}'
    
    # 2. Bind summarizer
    summarizer = MeetingSummarizer(client=mock_client, language="vi")
    
    # 3. Build test transcript
    mock_transcript = TranscriptResult(
        key="test_meeting",
        text="Mock meeting text content.",
        duration=10.0,
        sentence_info=[SentenceInfo(text="Mock meeting", start=0.0, end=10.0, speaker=0)]
    )
    
    # 4. Execute summarization
    report = summarizer.summarize(mock_transcript)
    
    # 5. Assertions
    assert report.summary == "Mocked summary"
    assert mock_client.complete.called
```

---

## 5. AI Verification Checklist

When building or auditing LLM integration modules, the Agent must verify:
- [ ] Are prompt templates stored separately as text files instead of hardcoded strings?
- [ ] Is there active chunking and map-reduce logic for transcripts exceeding length limits?
- [ ] Does chunking configure an overlap window to preserve semantic continuity?
- [ ] Is there an exponential backoff retry handler in place?
- [ ] Are LLM connections mocked during test executions to prevent network overhead?
- [ ] Is the LLM output parsed and validated using a structured schema (Pydantic/SQLModel)?
