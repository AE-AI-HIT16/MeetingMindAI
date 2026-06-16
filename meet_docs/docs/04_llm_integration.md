# LLM Integration Design — MeetASR

## 1. LLM Layer Overview

The LLM Layer is a **completely new** component compared to the original FunASR. It receives the **transcript** from the ASR pipeline and generates the **MeetingReport**.

```
TranscriptResult → LLMSummarizer → MeetingReport
```

---

## 2. Abstract LLM Client

```python
# meetasr/llm/abs_llm.py

from abc import ABC, abstractmethod
from typing import Optional

class AbsLLMClient(ABC):
    """Abstract base for any LLM provider."""

    @abstractmethod
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Send a prompt, return text response."""
        ...
```

---

## 3. Supported Providers

### 3.1 OpenAI / OpenAI-compatible (Primary)

```python
# meetasr/llm/openai_client.py
from openai import OpenAI

class OpenAIClient(AbsLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, prompt, system=None, temperature=0.3, max_tokens=4096) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message.content
```

### 3.2 Ollama (Local LLM — No API Key Required)

```python
# meetasr/llm/ollama_client.py
# Uses OpenAIClient with Ollama's base_url
# Ollama exposes an OpenAI-compatible API at localhost:11434

def make_ollama_client(model: str = "llama3.2") -> OpenAIClient:
    return OpenAIClient(
        api_key="ollama",          # placeholder
        model=model,
        base_url="http://localhost:11434/v1"
    )
```

### 3.3 Anthropic / Google (Phase 2)

To be added later if needed; all will implement `AbsLLMClient`.

---

## 4. Summarizer Design

```python
# meetasr/llm/summarizer.py

class MeetingSummarizer:
    """Orchestrates LLM calls to produce a full MeetingReport."""

    def __init__(self, client: AbsLLMClient, language: str = "vi"):
        self.client = client
        self.language = language
        self._load_prompts()

    def summarize(self, transcript: TranscriptResult) -> MeetingReport:
        text = self._format_transcript(transcript)

        # Parallel or sequential LLM calls
        summary = self._get_summary(text)
        topics = self._get_topics(text)
        actions = self._get_action_items(text)
        decisions = self._get_decisions(text)

        return MeetingReport(
            transcript=transcript,
            summary=summary,
            topics=topics,
            action_items=actions,
            decisions=decisions,
        )

    def _format_transcript(self, result: TranscriptResult) -> str:
        """Convert transcript to readable format for LLM."""
        lines = []
        for sent in result.sentence_info:
            ts = f"[{sent.start:.1f}s → {sent.end:.1f}s]"
            spk = f"Speaker {sent.speaker}" if sent.speaker is not None else "Unknown"
            lines.append(f"{ts} {spk}: {sent.text}")
        return "\n".join(lines)
```

---

## 5. Prompt Templates (Vietnamese Preferred)

### 5.1 Summary Prompt (vi)

```
# meetasr/llm/prompts/summarize_vi.txt

Bạn là trợ lý phân tích cuộc họp chuyên nghiệp.

Dưới đây là transcript cuộc họp với timestamp và người nói:

{transcript}

Hãy viết một đoạn tóm tắt súc tích (3-5 câu) bằng tiếng Việt, nêu rõ:
- Chủ đề chính của cuộc họp
- Những điểm quan trọng được thảo luận
- Kết quả/kết luận chính

Chỉ trả về đoạn tóm tắt, không có tiêu đề hay định dạng khác.
```

### 5.2 Topics Prompt (vi)

```
# meetasr/llm/prompts/topics_vi.txt

Dưới đây là transcript cuộc họp:

{transcript}

Hãy trích xuất các đề mục/chủ đề chính được thảo luận trong cuộc họp.
Trả về dạng JSON theo format sau, KHÔNG có markdown code block:
[
  {
    "title": "Tên đề mục ngắn gọn",
    "description": "Mô tả ngắn nội dung",
    "start_time": 0.0,
    "end_time": 120.5
  }
]

Giới hạn tối đa 8 đề mục. Sắp xếp theo thứ tự thời gian.
```

### 5.3 Action Items Prompt (vi)

```
# meetasr/llm/prompts/action_items_vi.txt

Dưới đây là transcript cuộc họp:

{transcript}

Hãy trích xuất TẤT CẢ action items (việc cần làm) được đề cập trong cuộc họp.
Trả về dạng JSON, KHÔNG có markdown code block:
[
  {
    "task": "Mô tả công việc cụ thể",
    "assignee": "Tên người được giao (hoặc null nếu không rõ)",
    "deadline": "Deadline nếu có (hoặc null)",
    "priority": "high | medium | low",
    "mentioned_by": "Speaker X",
    "timestamp": 145.2
  }
]

Nếu không có action item nào, trả về mảng rỗng [].
```

### 5.4 Decisions Prompt (vi)

```
# meetasr/llm/prompts/decisions_vi.txt

Dưới đây là transcript cuộc họp:

{transcript}

Hãy trích xuất các QUYẾT ĐỊNH chính thức được đưa ra trong cuộc họp.
Trả về dạng JSON, KHÔNG có markdown code block:
[
  {
    "content": "Nội dung quyết định",
    "made_by": "Speaker X (hoặc 'Nhóm' nếu đồng thuận chung)",
    "timestamp": 230.0
  }
]

Nếu không có quyết định nào, trả về mảng rỗng [].
```

---

## 6. Output Schema — MeetingReport

```python
# meetasr/schemas.py (using dataclass or pydantic)

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SentenceInfo:
    text: str
    start: float        # seconds
    end: float          # seconds
    speaker: Optional[int] = None

@dataclass
class TranscriptResult:
    key: str
    text: str
    duration: float
    sentence_info: list[SentenceInfo] = field(default_factory=list)

@dataclass
class Topic:
    title: str
    description: str
    start_time: float
    end_time: float

@dataclass
class ActionItem:
    task: str
    assignee: Optional[str]
    deadline: Optional[str]
    priority: str           # high | medium | low
    mentioned_by: Optional[str]
    timestamp: Optional[float]

@dataclass
class Decision:
    content: str
    made_by: str
    timestamp: Optional[float]

@dataclass
class MeetingReport:
    transcript: TranscriptResult
    summary: str
    topics: list[Topic]
    action_items: list[ActionItem]
    decisions: list[Decision]
    language: str = "vi"
    llm_model: str = ""
    processing_time: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to JSON-serializable dict."""
        ...

    def to_markdown(self) -> str:
        """Format as human-readable Markdown report."""
        ...
```

---

## 7. LLM Error Handling Strategy

```python
class MeetingSummarizer:

    def _safe_llm_call(self, prompt: str, fallback: any) -> any:
        """Call LLM with retries and fallback."""
        for attempt in range(3):
            try:
                response = self.client.chat(prompt)
                return self._parse_json(response)
            except json.JSONDecodeError:
                # LLM returned non-JSON → retry
                continue
            except Exception as e:
                logging.warning(f"LLM call failed attempt {attempt}: {e}")
                time.sleep(2 ** attempt)   # exponential backoff
        return fallback   # return default value instead of crashing
```

---

## 8. Strategy for Long Transcripts

For a 60-minute meeting, the transcript can be very long and exceed the LLM's context window.

### Chunking Strategy

```
If transcript ≤ 8000 tokens → call LLM once (direct)
If transcript > 8000 tokens → map-reduce approach:
    1. Split transcript into 10-15 minute chunks
    2. Summarize each chunk
    3. Aggregate chunk summaries into the final summary
```

```python
class MeetingSummarizer:
    MAX_TOKENS_DIRECT = 8000   # characters (approx. tokens)

    def _get_summary(self, text: str) -> str:
        if len(text) <= self.MAX_TOKENS_DIRECT:
            return self._summarize_direct(text)
        else:
            return self._summarize_mapreduce(text)
```

---

## 9. LLM Config

```yaml
# In meeting_config.yaml
llm:
  provider: openai               # openai | ollama
  model: gpt-4o-mini             # model name
  base_url: null                 # null = use default OpenAI url
  api_key: ${OPENAI_API_KEY}     # from env var
  language: vi                   # output language
  max_tokens: 4096
  temperature: 0.3
  max_chunk_chars: 8000          # chunk size for long transcripts
  retry_attempts: 3
  timeout: 60                    # seconds
```
