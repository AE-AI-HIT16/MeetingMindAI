# Goals & Constraints — MeetASR

## 1. Goal Statement

Build **MeetASR** — an end-to-end meeting processing system:

1. **ASR Pipeline**: Receive audio → transcribe text (speech-to-text)
2. **Speaker Diarization**: Determine who spoke what and when
3. **LLM Summarization**: Summarize meeting content, extract key topics, and action items

### Specific Goals (Verifiable)

| Goal | Measurement Criteria |
|---|---|
| Accurate Vietnamese transcription | WER ≤ 10% on real meeting audio |
| Speaker Diarization | DER ≤ 15% |
| Summarize a 60-minute meeting | ≤ 30 seconds of LLM processing |
| Action items extraction | Accuracy ≥ 80% compared to gold standard |
| API response time | ≤ 5 seconds for audio ≤ 10 minutes |

---

## 2. Scope

### In Scope

- ✅ ASR: Offline transcription for audio files (wav, mp3, m4a, mp4)
- ✅ VAD: Voice activity detection, silence segmenting
- ✅ Speaker diarization: Distinguish speakers
- ✅ Punctuation restoration: Auto-insert punctuation marks
- ✅ LLM summary: Summarize the entire meeting
- ✅ LLM extraction: Agenda, action items, decisions, key topics
- ✅ REST API: Endpoint to upload audio → receive JSON results
- ✅ Output formats: JSON, Markdown, SRT subtitles
- ✅ Vietnamese support is priority #1

### Out of Scope (MVP)

- ❌ Real-time streaming (can be added later)
- ❌ Training models from scratch (inference only)
- ❌ Web UI (API-first, UI is phase 2)
- ❌ Multi-tenant / auth (local/self-hosted single-user tool first)
- ❌ Cloud deployment (local/self-hosted first)

---

## 3. Constraints

### Technical

- **Python ≥ 3.10** — use modern typing, match statements
- **PyTorch ≥ 2.0** — compile support, better performance
- **Runs on CPU** — GPU is not mandatory (GPU is a bonus)
- **Reuse model weights from FunASR/ModelScope** — no re-training
- **LLM API-based** — call OpenAI / Ollama / any OpenAI-compatible endpoint
- **No dependency on the FunASR package** — code from scratch, only borrow model weights

### Architecture

- **Modular design** — each component (VAD, ASR, SPK, LLM) is independent and swappable
- **Registry pattern** — similar to FunASR, use decorators to register model classes
- **Config-driven** — all hyperparameters via YAML/dict, no hardcoding
- **Pipeline pattern** — sequential processing steps are clearly defined

### Code Quality (per claude.md)

- **Simplicity first** — do not over-engineer, do not add features before they are needed
- **Surgical changes** — each file does exactly one thing
- **Goal-driven** — each function has clear unit tests
- **Type hints mandatory** — all public APIs must have type annotations

---

## 4. User Personas

### Persona 1: Developer / Researcher

- Upload audio file → get transcript + summary in JSON format
- Integrate into their pipeline via REST API
- Needs: Clear API documentation, standardized output formats

### Persona 2: Non-technical User (Phase 2)

- Wants to drag-and-drop audio files → view summary on a web page
- Needs: Simple UI (not MVP)

---

## 5. Definition of "Done"

A feature is considered completed when:

1. Unit tests pass
2. Full type hints are implemented
3. Docstrings clearly describe input/output
4. Runs end-to-end with sample audio files
5. Output JSON is valid according to `data_formats.md`
