# Development Roadmap — MeetASR

## Phase Overview

```
Phase 0: Foundation     (1-2 weeks) ← Scaffolding + core infrastructure
Phase 1: ASR Core       (2-3 weeks) ← VAD + ASR + Punc pipeline operational
Phase 2: Speaker        (1-2 weeks) ← Speaker diarization
Phase 3: LLM Layer      (1-2 weeks) ← Summarization, action items
Phase 4: API + CLI      (1 week)    ← REST API, CLI, output formats
Phase 5: Polish         (ongoing)   ← Tests, docs, perf optimization
```

---

## Phase 0 — Foundation (Weeks 1-2)

### Objective
Create the skeleton project, model registry, config system, and model downloader.

### Tasks

```
[ ] 0.1 Create meetasr/ package directory structure
[ ] 0.2 register.py — Registry system (copy + adapt from FunASR)
[ ] 0.3 utils/audio.py — load_audio() supporting wav/mp3/m4a/mp4
[ ] 0.4 utils/download.py — download model from ModelScope + HuggingFace
[ ] 0.5 setup.py / pyproject.toml
[ ] 0.6 tests/ — test framework setup (pytest)
[ ] 0.7 Verify: able to download fsmn-vad weights and load config.yaml
```

### Verification Criteria

```python
# Test: download and load model config
from meetasr.utils.download import download_model
config = download_model(model="fsmn-vad", hub="ms")
assert "model" in config
assert "frontend" in config
```

---

## Phase 1 — ASR Core Pipeline (Weeks 2-4)

### Objective
Achieve end-to-end VAD → ASR → Punctuation pipeline execution.

### Tasks

```
[ ] 1.1 frontends/fbank.py — WavFrontend: audio → fbank features
    Verify: features shape [T, 80], dtype float32

[ ] 1.2 models/vad/fsmn_vad.py — FSMN-VAD wrapper
    Verify: returns List[Segment(start_ms, end_ms)]

[ ] 1.3 models/asr/sense_voice.py — SenseVoice wrapper
    Verify: {"text": "...", "timestamp": [...]}

[ ] 1.4 models/asr/paraformer.py — Paraformer wrapper
    Verify: matches SenseVoice output format

[ ] 1.5 models/punc/ct_transformer.py — CT-Punc wrapper
    Verify: text input → punctuated text output

[ ] 1.6 pipeline.py — MeetPipeline orchestrator
    Verify: audio file → TranscriptResult JSON

[ ] 1.7 auto/auto_model.py — AutoModel (simplified)
    Verify: AutoModel(model="sensevoice-small") loads correctly

[ ] 1.8 Integration test: real audio file → transcript
```

### Example Test

```python
# tests/test_pipeline.py
def test_asr_pipeline_end_to_end():
    pipeline = MeetPipeline.from_config({
        "asr": {"model": "iic/SenseVoiceSmall"},
        "vad": {"model": "fsmn-vad"},
        "punc": {"model": "ct-punc"},
    })
    result = pipeline.transcribe("tests/fixtures/sample_vi.wav")
    assert isinstance(result, TranscriptResult)
    assert len(result.text) > 0
    assert result.duration > 0
```

---

## Phase 2 — Speaker Diarization (Weeks 4-5)

### Objective
Determine who spoke what, and add speaker labels to the transcript.

### Tasks

```
[ ] 2.1 models/spk/campplus.py — CAM++ speaker embedding
    Verify: audio chunk → embedding vector [192]

[ ] 2.2 models/spk/cluster.py — UMAP + AgglomerativeClustering
    Verify: N embeddings → N speaker labels (int)

[ ] 2.3 Pipeline integration: merge speaker labels into sentence_info
    Verify: sentence_info[i].speaker in {0, 1, 2, ...}

[ ] 2.4 Integration test with 2-speaker audio
```

### Example Output

```json
{
  "sentence_info": [
    {"text": "Chào mừng mọi người", "start": 0.5, "end": 2.1, "speaker": 0},
    {"text": "Cảm ơn anh", "start": 2.5, "end": 3.2, "speaker": 1}
  ]
}
```

---

## Phase 3 — LLM Summarization (Weeks 5-6)

### Objective
Integrate LLM to summarize the meeting and extract agenda/action items.

### Tasks

```
[ ] 3.1 llm/abs_llm.py — AbsLLMClient interface
[ ] 3.2 llm/openai_client.py — OpenAI API client
[ ] 3.3 llm/ollama_client.py — Ollama local client
[ ] 3.4 llm/prompts/ — Vietnamese prompt templates
    [ ] summarize_vi.txt
    [ ] topics_vi.txt
    [ ] action_items_vi.txt
    [ ] decisions_vi.txt
[ ] 3.5 llm/summarizer.py — MeetingSummarizer class
    Verify: TranscriptResult → MeetingReport
[ ] 3.6 Chunking logic for long transcripts (>8000 chars)
[ ] 3.7 Retry + fallback logic
[ ] 3.8 schemas.py — MeetingReport.to_dict() + to_markdown()
```

### Verification Criteria

```python
def test_llm_summarizer():
    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    summarizer = MeetingSummarizer(client, language="vi")
    report = summarizer.summarize(transcript_result)

    assert len(report.summary) > 50
    assert isinstance(report.topics, list)
    assert isinstance(report.action_items, list)
    assert isinstance(report.decisions, list)
    assert report.to_dict()  # JSON serializable
```

---

## Phase 4 — API & CLI (Weeks 6-7)

### Objective
Expose REST API and CLI entrypoints for production usage.

### Tasks

```
[ ] 4.1 api/app.py — FastAPI app
[ ] 4.2 api/routes/transcribe.py — POST /v1/audio/transcriptions
[ ] 4.3 api/routes/summarize.py — POST /v1/meeting/summarize
[ ] 4.4 api/routes/db_routes.py — CRUD endpoints for meetings
[ ] 4.5 bin/cli.py — CLI tool supporting 'transcribe', 'summarize', 'server'
[ ] 4.6 Output formats: JSON, Markdown, SRT
[ ] 4.7 Automated OpenAPI documentation (Swagger)
```

### API Test

```bash
# Transcribe
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "model=sensevoice" \
  -F "language=vi"

# Full meeting report
curl -X POST http://localhost:8000/v1/meeting/summarize \
  -F "file=@meeting.wav" \
  -F "language=vi" \
  -F "llm_model=gpt-4o-mini"
```

---

## Phase 5 — Polish & Optimization (Ongoing)

```
[ ] 5.1 Comprehensive unit tests (target: >80% coverage)
[ ] 5.2 Performance benchmarks (RTF measurement)
[ ] 5.3 Streaming transcription support
[ ] 5.4 Web UI (Gradio or simple HTML dashboard)
[ ] 5.5 Docker packaging
[ ] 5.6 Documentation (mkdocs)
[ ] 5.7 English prompt templates
[ ] 5.8 Multi-language meeting support (auto-detect)
```

---

## Phase Dependencies

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 4
                │                      ▲
                └──► Phase 3 ──────────┘
                
Phase 5 (runs in parallel with Phase 4+)
```

---

## Milestone Checklist

### MVP (Phase 0-1 Completed)
- [ ] Vietnamese audio transcription functional
- [ ] Output includes text with global timestamps

### Alpha (Phase 0-3 Completed)  
- [ ] Full pipeline: VAD + ASR + Punc + Spk + LLM summary
- [ ] Runs locally using Ollama

### Beta (Phase 0-4 Completed)
- [ ] REST API endpoints operational
- [ ] CLI `meetasr` executable
- [ ] OpenAI + Ollama model integrations
