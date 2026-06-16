# Overall Architecture — MeetASR

## 1. High-level Overview

```
Audio Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    MeetPipeline                          │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │   VAD    │───▶│   ASR    │───▶│   Punctuation    │  │
│  │ (fsmn)   │    │(paraform │    │   (ct-punc)      │  │
│  └──────────┘    │ /sensev) │    └──────────────────┘  │
│                  └──────────┘              │            │
│  ┌──────────┐                             │            │
│  │ Speaker  │─────────────────────────────┤            │
│  │ Diarize  │                             │            │
│  │(cam++)   │                             ▼            │
│  └──────────┘              ┌──────────────────────┐   │
│                             │   Transcript Merger  │   │
│                             │  (text + timestamps  │   │
│                             │   + speaker labels)  │   │
│                             └──────────────────────┘   │
└─────────────────────────────────────┬───────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │      LLM Summarizer      │
                        │                          │
                        │  • Full summary          │
                        │  • Key topics / agenda   │
                        │  • Action items          │
                        │  • Decisions made        │
                        │  • Sentiment per speaker │
                        └─────────────────────────┘
                                      │
                                      ▼
                              MeetingReport JSON
```

---

## 2. Package Structure

```
meetasr/                          ← main package
├── __init__.py
├── version.txt
├── register.py                   ← Registry system (similar to FunASR)
├── pipeline.py                   ← MeetPipeline - main orchestrator
│
├── auto/                         ← Highest-level entry point
│   ├── __init__.py
│   ├── auto_model.py             ← AutoModel: load + build model from config
│   └── auto_pipeline.py          ← AutoPipeline: build full meeting pipeline
│
├── models/                       ← Model components
│   ├── __init__.py
│   ├── vad/                      ← Voice Activity Detection
│   │   ├── __init__.py
│   │   ├── fsmn_vad.py           ← FSMN-VAD model wrapper
│   │   └── abs_vad.py            ← Abstract base class
│   ├── asr/                      ← Automatic Speech Recognition
│   │   ├── __init__.py
│   │   ├── paraformer.py         ← Paraformer model wrapper
│   │   ├── sense_voice.py        ← SenseVoice model wrapper
│   │   └── abs_asr.py            ← Abstract base class
│   ├── punc/                     ← Punctuation Restoration
│   │   ├── __init__.py
│   │   ├── ct_transformer.py     ← CT-Transformer wrapper
│   │   └── abs_punc.py
│   └── spk/                      ← Speaker Diarization
│       ├── __init__.py
│       ├── campplus.py           ← CAM++ speaker embedding
│       ├── cluster.py            ← Clustering backend
│       └── abs_spk.py
│
├── frontends/                    ← Audio feature extraction
│   ├── __init__.py
│   ├── abs_frontend.py
│   ├── fbank.py                  ← Log-Mel filterbank (primary)
│   └── wav_frontend.py           ← WaveForm → feature
│
├── tokenizer/                    ← Text tokenization
│   ├── __init__.py
│   ├── abs_tokenizer.py
│   ├── char_tokenizer.py
│   └── sentencepiece_tokenizer.py
│
├── llm/                          ← LLM Integration Layer (NEW)
│   ├── __init__.py
│   ├── abs_llm.py                ← Abstract LLM client
│   ├── openai_client.py          ← OpenAI / OpenAI-compatible
│   ├── ollama_client.py          ← Ollama local LLM
│   ├── summarizer.py             ← Meeting summarization logic
│   ├── extractor.py              ← Action items, decisions extractor
│   └── prompts/                  ← Prompt templates
│       ├── summarize_vi.txt      ← Vietnamese summary prompt
│       ├── summarize_en.txt      ← English summary prompt
│       ├── action_items_vi.txt
│       └── topics_vi.txt
│
├── utils/                        ← Shared utilities
│   ├── __init__.py
│   ├── audio.py                  ← Audio loading, resampling
│   ├── timestamp.py              ← Timestamp merging utilities
│   ├── vad_utils.py              ← VAD segment processing
│   ├── download.py               ← Model download from hub
│   └── misc.py
│
├── download/                     ← Model hub download
│   ├── __init__.py
│   ├── modelscope.py             ← Download from ModelScope
│   └── huggingface.py            ← Download from HuggingFace
│
├── datasets/                     ← Data loading (for training/eval)
│   ├── __init__.py
│   └── audio_dataset.py
│
├── api/                          ← REST API layer
│   ├── __init__.py
│   ├── app.py                    ← FastAPI application
│   ├── routes/
│   │   ├── transcribe.py         ← POST /v1/audio/transcriptions
│   │   └── summarize.py          ← POST /v1/meeting/summarize
│   └── schemas.py                ← Pydantic models
│
└── bin/                          ← CLI entrypoints
    ├── __init__.py
    ├── transcribe.py             ← meetasr transcribe audio.wav
    └── server.py                 ← meetasr-server
```

---

## 3. Detailed Data Flow

### 3.1 ASR Pipeline (without LLM)

```python
input: audio file path / numpy array / bytes

1. load_audio(input)           → np.ndarray [16kHz, float32]
2. vad.detect(audio)           → List[Segment(start_ms, end_ms)]
3. for each segment:
   a. slice audio
   b. frontend.extract(audio)  → fbank features [T, 80]
   c. asr.decode(features)     → {"text": str, "timestamp": [...]}
4. merge_results(segments)     → full transcript with global timestamps
5. punc.restore(text)          → punctuated text
6. spk.diarize(audio, segs)    → speaker labels per segment

output: TranscriptResult {
    key: str,
    text: str,              # full transcript
    sentence_info: [        # per-sentence breakdown
        {text, start, end, speaker, timestamp}
    ],
    duration: float         # audio length in seconds
}
```

### 3.2 LLM Summarization (after ASR)

```python
input: TranscriptResult

1. format_transcript(result)   → formatted text with speaker labels
2. llm.summarize(text)         → SummaryResult
3. llm.extract_topics(text)    → List[Topic]
4. llm.extract_actions(text)   → List[ActionItem]
5. llm.extract_decisions(text) → List[Decision]

output: MeetingReport {
    transcript: TranscriptResult,
    summary: str,           # summary paragraph
    topics: [               # key topics/agenda
        {title, description, timestamp_range}
    ],
    action_items: [         # things to do
        {assignee, task, deadline, priority}
    ],
    decisions: [            # decisions made
        {content, made_by, timestamp}
    ],
    sentiment: {            # per speaker
        "Speaker 0": "positive",
        "Speaker 1": "neutral"
    }
}
```

---

## 4. Registry Pattern

```python
# Register models
from meetasr.register import tables

@tables.register("model_classes", key="fsmn-vad")
class FsmnVAD(AbsVAD):
    ...

@tables.register("model_classes", key="paraformer-zh")
class Paraformer(AbsASR):
    ...

# Usage
vad_class = tables.model_classes["fsmn-vad"]
vad = vad_class(**config)
```

---

## 5. Config System

```yaml
# meeting_config.yaml
asr:
  model: paraformer-zh
  device: cpu
  hub: ms          # modelscope or hf

vad:
  model: fsmn-vad
  max_single_segment_time: 60000   # ms

punc:
  model: ct-punc

spk:
  model: cam++
  spk_mode: punc_segment

llm:
  provider: openai              # openai | ollama | anthropic
  model: gpt-4o-mini
  base_url: http://localhost:11434/v1   # for Ollama
  api_key: ${OPENAI_API_KEY}
  language: vi                  # output language
  max_tokens: 4096
  temperature: 0.3
```

---

## 6. API Design

```
POST /v1/audio/transcriptions
  Content-Type: multipart/form-data
  Body: file (audio), model, language, response_format
  Response: { text, segments, words }

POST /v1/meeting/summarize
  Content-Type: multipart/form-data
  Body: file (audio), language, llm_model
  Response: MeetingReport JSON

GET /v1/health
  Response: { status, models_loaded }
```

---

## 7. Dependency Stack

```
meetasr
  ├── torch >= 2.0           # model inference
  ├── torchaudio             # audio loading
  ├── librosa                # audio processing
  ├── soundfile              # audio I/O
  ├── numpy                  # array ops
  ├── scipy                  # signal processing
  ├── omegaconf              # config management
  ├── modelscope             # model download
  ├── huggingface_hub        # HF model download
  ├── transformers           # tokenizers, model utils
  ├── sentencepiece          # subword tokenizer
  ├── openai                 # LLM API client
  ├── fastapi                # REST API
  ├── uvicorn                # ASGI server
  ├── python-multipart       # file upload
  └── pydantic >= 2.0        # data validation
```
