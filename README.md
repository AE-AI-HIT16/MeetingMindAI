# MeetASR

> **Meeting Speech Recognition + LLM Summarization** — Rebuilt from FunASR, simplified, and integrated with AI analysis.

## Quick Install

```bash
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from meetasr.auto.auto_pipeline import AutoPipeline

pipeline = AutoPipeline.from_config({
    "asr": {"model": "sensevoice-small", "device": "cpu"},
    "vad": {"model": "fsmn-vad"},
    "punc": {"model": "ct-punc"},
    "llm": {
        "provider": "ollama",
        "model": "llama3.2",
        "language": "vi",
    },
})

# Transcription only
result = pipeline.transcribe("meeting.wav")
print(result.text)
print(result.to_srt())

# Transcription + LLM summary
report = pipeline.summarize_meeting("meeting.wav")
print(report.summary)
print(report.to_markdown())
```

### CLI

```bash
# Transcription
meetasr transcribe meeting.wav --language vi -f srt -o ./output/

# Full meeting report (requires config containing 'llm' block)
meetasr summarize meeting.wav --config meeting_config.yaml -f markdown -o ./reports/

# Start the REST API server
meetasr server --config meeting_config.yaml --port 8000
```

### REST API

```bash
# Transcription
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@meeting.wav" -F "language=vi" -F "response_format=verbose_json"

# Meeting Summarization
curl -X POST http://localhost:8000/v1/meeting/summarize \
  -F "file=@meeting.wav" -F "language=vi"
```

## Project Directory Structure

```
meetasr/                  ← Core package
├── auto/                 ← AutoModel + AutoPipeline (entry points)
├── models/               ← VAD, ASR, Punc, Speaker wrappers
│   ├── vad/              ← fsmn_vad.py
│   ├── asr/              ← sense_voice.py, paraformer.py
│   ├── punc/             ← ct_transformer.py
│   └── spk/              ← campplus.py
├── llm/                  ← LLM integration
│   ├── prompts/          ← Prompt templates (vi/en)
│   ├── openai_client.py
│   ├── ollama_client.py
│   └── summarizer.py
├── frontends/            ← Audio feature extraction
├── tokenizer/            ← Text tokenizers
├── utils/                ← Audio loading, timestamps, download
├── api/                  ← FastAPI REST endpoints
├── bin/                  ← CLI (meetasr command)
├── register.py           ← Component registry
├── pipeline.py           ← MeetPipeline orchestrator
└── schemas.py            ← TranscriptResult, MeetingReport

meet_docs/                ← Design docs & plans
├── docs/                 ← Goals, Architecture, Analysis, LLM design, Roadmap, Database, Pipelines, Tasks
├── specs/                ← API spec, Data formats, Registry spec
└── constraints/          ← Technical constraints, Coding style

tests/                    ← Unit tests
```

## Configuration

Copy `meeting_config.example.yaml` → `meeting_config.yaml` and modify:

```yaml
asr:
  model: sensevoice-small
  device: cpu

vad:
  model: fsmn-vad

punc:
  model: ct-punc

llm:
  provider: openai          # or ollama
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  language: vi
```

## Running Tests

```bash
pytest tests/ -v
```

## Supported Models

| Model | Key | Purpose |
|---|---|---|
| SenseVoiceSmall | `sensevoice-small` | ASR (vi/zh/en/ja/ko) |
| Paraformer-zh | `paraformer-zh` | ASR (zh/en, fastest) |
| FSMN-VAD | `fsmn-vad` | Voice Activity Detection |
| CT-Punc | `ct-punc` | Punctuation Restoration |
| CAM++ | `cam++` | Speaker Diarization |

## LLM Providers

| Provider | Config | Notes |
|---|---|---|
| OpenAI | `provider: openai` | Requires API key |
| Ollama | `provider: ollama` | Local and free |
| Any OpenAI-compatible | `base_url: ...` | Groq, Together, LM Studio |
