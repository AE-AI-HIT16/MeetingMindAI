# Team Task Breakdown — MeetASR

The MeetASR project is built upon the existing skeleton codebase. The development team consists of **4 members (all beginners)**: 3 AI Engineers and 1 Data Engineer. This document maps out detailed tasks for each role, preventing overlap and ensuring clear implementation paths.

---

## 1. Project Dependencies Map

To ensure a smooth workflow, team members must align on the following development dependencies:

```
[AI Engineer 1: Model Wrappers] ──┐
                                  ├───► [AI Engineer 3: Pipeline & API Integration]
[AI Engineer 2: LLM Layer] ───────┤
                                  │
[Data Engineer: DB & CRUD] ───────┘
```

- **AI Engineer 3** cannot integrate the orchestrator pipeline until **AI Engineer 1** completes the core model wrappers.
- **AI Engineer 3** cannot integrate the summarization feature into the REST API until **AI Engineer 2** finishes the `summarizer.py` module.
- **AI Engineer 3** cannot save meeting results or query report states until the **Data Engineer** completes the Database CRUD layer.

---

## 2. Role-by-Role Task Breakdown

### 2.1 AI Engineer 1: Model Wrappers & Audio Features
**Role:** Ensure all machine learning models (VAD, ASR, Punc, Speaker) are wrapped correctly, load weights from ModelScope/HuggingFace, and run inference reliably.

#### Responsible Files:
- [WavFrontend](file:///d:/FunASR/meetasr/frontends/fbank.py) (`meetasr/frontends/fbank.py`)
- [FSMN-VAD](file:///d:/FunASR/meetasr/models/vad/fsmn_vad.py) (`meetasr/models/vad/fsmn_vad.py`)
- [SenseVoice ASR](file:///d:/FunASR/meetasr/models/asr/sense_voice.py) (`meetasr/models/asr/sense_voice.py`)
- [Paraformer ASR](file:///d:/FunASR/meetasr/models/asr/paraformer.py) (`meetasr/models/asr/paraformer.py`)
- [CT-Punc](file:///d:/FunASR/meetasr/models/punc/ct_transformer.py) (`meetasr/models/punc/ct_transformer.py`)
- [CAM++ Speaker](file:///d:/FunASR/meetasr/models/spk/campplus.py) (`meetasr/models/spk/campplus.py`)

#### Specific Tasks:
1. **Implement WavFrontend:** Write the Fbank feature extraction logic. The output must be a tensor of shape `[Time_Frames, 80]`. Apply Low Frame Rate (LFR) stacking to convert the features to `[New_Frames, 560]`.
2. **Implement VAD Wrapper:** Load `fsmn-vad` from ModelScope, write the `extract_segments` method that takes a waveform and returns a list of `Segment(start_ms, end_ms)` objects.
3. **Implement ASR Wrappers (SenseVoice & Paraformer):** Implement the SenseVoice model (prioritizing Vietnamese) and Paraformer (fallback). The `transcribe_segment` method must take an audio segment and return text alongside character-level timestamps.
4. **Implement Punctuation Wrapper:** Load `ct-punc`. The `add_punc` method should accept plain text and return text with automatically inserted punctuation marks.
5. **Implement Speaker Embedding Wrapper:** Load `cam++` to extract speaker embedding vectors (dimension size 192) from audio segments to support speaker identification.

#### Verification:
- Write isolated unit tests in the `tests/` directory (e.g., passing a sample WAV through `fsmn_vad.py` and asserting that it yields a valid list of segment timestamps).

---

### 2.2 AI Engineer 2: LLM Summarization Layer
**Role:** Build the LLM integration layer to connect with external APIs or local models, handle text chunking for long meetings, and extract summaries, key topics, decisions, and action items.

#### Responsible Files:
- [AbsLLMClient](file:///d:/FunASR/meetasr/llm/abs_llm.py) (`meetasr/llm/abs_llm.py`)
- [OpenAIClient](file:///d:/FunASR/meetasr/llm/openai_client.py) (`meetasr/llm/openai_client.py`)
- [OllamaClient](file:///d:/FunASR/meetasr/llm/ollama_client.py) (`meetasr/llm/ollama_client.py`)
- [Prompt Templates](file:///d:/FunASR/meetasr/llm/prompts/) (`meetasr/llm/prompts/*`)
- [MeetingSummarizer](file:///d:/FunASR/meetasr/llm/summarizer.py) (`meetasr/llm/summarizer.py`)

#### Specific Tasks:
1. **Implement Clients:** Write connection and response extraction logic in `openai_client.py` and `ollama_client.py` extending `AbsLLMClient`. Ensure robust handling of environment variables, custom base URLs, and timeout retry handlers.
2. **Design Prompt Templates:** Author structured templates in the `prompts/` directory (in Vietnamese) to ensure the LLM returns well-formed JSON. Target four extraction tasks: summary paragraph, timeline-indexed topics, action items, and key decisions.
3. **Build MeetingSummarizer:** Orchestrate LLM calls.
   - **Chunking Logic:** Automatically segment long transcripts exceeding token windows (e.g., >8000 characters) into overlapping chunks.
   - **Map-Reduce:** Run local summarization on each chunk, then aggregate summaries into a final coherent `MeetingReport`.

#### Verification:
- Run the summarizer pipeline using mock transcript results and verify that `MeetingReport` exports valid Markdown and JSON structures.

---

### 2.3 AI Engineer 3: Pipeline & API/CLI Integration
**Role:** Act as the system integrator. Combine the core models from **AI Engineer 1** and the LLM summarizer from **AI Engineer 2** into a cohesive end-to-end flow. Expose the workflow via a FastAPI REST API and CLI commands.

#### Responsible Files:
- [Registry System](file:///d:/FunASR/meetasr/register.py) (`meetasr/register.py`)
- [MeetPipeline](file:///d:/FunASR/meetasr/pipeline.py) (`meetasr/pipeline.py`)
- [AutoModel & AutoPipeline](file:///d:/FunASR/meetasr/auto/auto_model.py) (`meetasr/auto/auto_model.py` and `auto_pipeline.py`)
- [FastAPI App](file:///d:/FunASR/meetasr/api/app.py) (`meetasr/api/app.py`)
- [CLI Command](file:///d:/FunASR/meetasr/bin/cli.py) (`meetasr/bin/cli.py`)

#### Specific Tasks:
1. **Integrate MeetPipeline:** Chain the execution steps:
   `Audio File` -> `Load Waveform` -> `VAD Segment Detection` -> `ASR Transcription per Segment` -> `Punctuation Restoration` -> `Speaker Diarization` -> `TranscriptResult`.
2. **Implement Registry & Auto classes:** Support dynamic model lookup and pipeline loading from YAML configuration files using the `@register` decorators.
3. **Build REST API (FastAPI):**
   - Endpoint `/v1/audio/transcriptions`: Accepts uploaded audio, executes ASR pipeline, and returns text or SRT.
   - Endpoint `/v1/meeting/summarize`: Accepts uploaded audio, runs transcription, calls `MeetingSummarizer`, and returns a structured `MeetingReport`.
   - Integrate with the database CRUD layer to register meeting sessions and track processing states (`pending`, `processing`, `completed`, `failed`).
4. **Develop CLI:** Create entrypoints for terminal execution:
   - `meetasr transcribe --audio file.wav --format srt`
   - `meetasr server --port 8000`

#### Verification:
- Spin up the FastAPI server locally and send test requests using `curl` or Postman to verify end-to-end behavior.

---

### 2.4 Data Engineer: Database & Data Pipeline
**Role:** Manage persistent storage, optimize database read/write queries, finalize raw audio reading/resampling, and build dataset utilities for batch offline jobs.

#### Responsible Files:
- [Audio Utils](file:///d:/FunASR/meetasr/utils/audio.py) (`meetasr/utils/audio.py`)
- [Database Connection](file:///d:/FunASR/meetasr/db/connection.py) (New file `meetasr/db/connection.py`)
- [Database Models](file:///d:/FunASR/meetasr/db/models.py) (New file `meetasr/db/models.py`)
- [Database Repository](file:///d:/FunASR/meetasr/db/repository.py) (New file `meetasr/db/repository.py`)
- [Dataset Loaders](file:///d:/FunASR/meetasr/datasets/) (New files `meetasr/datasets/scp_dataset.py`, `jsonl_dataset.py`)

#### Specific Tasks:
1. **Finalize Audio Utils:** Implement robust loading of compressed files (`.mp3`, `.m4a`, etc.) using `soundfile` or `torchaudio`. Ensure audio is converted to single-channel (mono) at a 16kHz sampling rate.
2. **Define Database Models:** Implement tables `meetings`, `transcripts`, `sentences`, `reports`, `topics`, `action_items`, and `decisions` according to [06_database.md](file:///d:/FunASR/meet_docs/docs/06_database.md) using SQLModel.
3. **Implement CRUD Repository:** Develop repository functions to:
   - Create a session (`create_meeting`).
   - Update state (`update_meeting_status`).
   - Store full reports (`save_meeting_result`).
   - Query meeting history and reports (`get_meeting_report`).
4. **Build Dataset Loaders:** Implement `SCPDataset` and `JSONLDataset` classes following [07_data_pipeline.md](file:///d:/FunASR/meet_docs/docs/07_data_pipeline.md) for batch training and feature extraction pipelines.

#### Verification:
- Write a database sanity script that creates a local SQLite instance, saves a mock `MeetingReport`, and reads it back to assert schema alignment.

---

## 3. Collaboration & Branching Strategy

To maintain codebase health and prevent merge conflicts:
- **Git Branching:** Each developer must work on a dedicated branch branched off `main` (e.g., `feature/model-wrappers`, `feature/llm-layer`, `feature/database-crud`).
- **Integration Workflow:** AI Engineer 1, 2, and the Data Engineer must deliver their modules first. AI Engineer 3 will pull these changes into their branch to assemble and verify the final application.
- **Coding Conventions:** Strictly follow [coding_style.md](file:///d:/FunASR/meet_docs/constraints/coding_style.md) regarding naming conventions, docstrings, and typing.
