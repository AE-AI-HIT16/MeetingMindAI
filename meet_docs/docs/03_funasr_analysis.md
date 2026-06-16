# Original FunASR Analysis — Lessons Learned & Redesign Decisions

## 1. FunASR Overview

FunASR is an end-to-end ASR toolkit by Alibaba DAMO Academy. Key highlights include:

- **170x real-time** on GPU with SenseVoice-Small
- **17x real-time** on CPU
- 50+ languages supported
- Pipeline: VAD → ASR → Punctuation → Speaker Diarization
- Model Zoo: Paraformer, SenseVoice, Fun-ASR-Nano, Qwen3-ASR, CAM++, FSMN-VAD, CT-Punc

---

## 2. Original FunASR Architecture — Component Analysis

### 2.1 Package Structure

```
funasr/
├── __init__.py         ← Auto-imports all submodules on startup
├── register.py         ← Registry: RegisterTables dataclass
├── auto/
│   ├── auto_model.py   ← AutoModel class (1035 lines) — primary entry point
│   └── auto_model_vllm.py
├── models/             ← 53 model folders!
├── frontends/          ← Audio feature extraction (fbank, wav)
├── tokenizer/          ← Char, SentencePiece, Whisper, HF tokenizers
├── datasets/           ← 8 dataset loader types
├── train_utils/        ← Trainer, trainer_ds (deepspeed)
├── utils/              ← Load audio, timestamps, VAD utils
├── download/           ← ModelScope + HuggingFace download
└── bin/                ← CLI, server, train entrypoints
```

### 2.2 Registry Pattern — RETAINED

```python
# funasr/register.py
@dataclass
class RegisterTables:
    model_classes = {}
    frontend_classes = {}
    tokenizer_classes = {}
    # ... 13 registry tables

    def register(self, register_tables_key, key=None):
        def decorator(target_class):
            registry[registry_key] = target_class
            return target_class
        return decorator

tables = RegisterTables()
```

**Evaluation:** ✅ This pattern is highly effective. Decorator-based registration, lazy loading, and easy model swapping. **Retained exactly as is.**

### 2.3 AutoModel — REWRITTEN (Simplified)

```python
# Issues with original AutoModel:
# - 1035 lines in a single file
# - build_model() handles too many responsibilities
# - inference_with_vad() is complex and hard to read
# - Mixes training and inference logic

# MeetASR Solution:
# - Split into AutoModel + MeetPipeline
# - AutoModel only loads individual models
# - MeetPipeline orchestrates the full flow
```

### 2.4 Frontend (Audio Features) — RETAINED & Simplified

FunASR uses:
- `WavFrontend` — computes FBANK features, CMVN normalization
- `DefaultFrontend` — more generalized
- `WhisperFrontend` — for Whisper models

**Retained:** `WavFrontend` (Mel filterbank, CMVN) as it is the standard for Paraformer/SenseVoice models.

### 2.5 VAD Pipeline — REWRITTEN (Cleaned Up)

FunASR flow:
```
inference_with_vad():
  1. vad.inference() → vad segments
  2. Sort segments by length (for efficient batching)
  3. slice_padding_audio_samples()
  4. asr.inference() per batch
  5. Restore original order
  6. Merge timestamps
  7. punc.inference()
  8. spk.inference() per segment
```

**Issues:** Complex logic, nested loops, difficult to test.  
**Solution:** Split into individual `VADProcessor`, `ASRProcessor`, `PuncProcessor`, and `SPKProcessor` with clear, clean interfaces.

### 2.6 Download System — RETAINED LOGIC

FunASR downloads from:
- ModelScope: `modelscope.hub.snapshot_download()`
- HuggingFace: `huggingface_hub.snapshot_download()`

After downloading, it reads `config.yaml` to obtain:
- `model` class name
- `frontend` class name
- `tokenizer` class name
- hyperparameters

**Retained this logic** — to reuse model weights from the FunASR ecosystem.

---

## 3. Models Used in MeetASR

### Recommended Model Stack

| Task | Model | Reason for Selection |
|---|---|---|
| VAD | `fsmn-vad` | Small (0.4M), fast, accurate |
| ASR (Vietnamese) | `SenseVoiceSmall` | Supports 5 languages incl. Vietnamese, emotion detection |
| ASR (Multilingual) | `paraformer-zh` | Production-grade, 170x real-time |
| Punctuation | `ct-punc` | Standard, zh/en support, 290M |
| Speaker Diarization | `cam++` | Lightweight 7.2M, good quality |

### Model Config Format (Retained from FunASR)

```yaml
# Each model folder has config.yaml:
model: Paraformer          # class name in the registry
frontend: WavFrontend
tokenizer: CharTokenizer
frontend_conf:
  fs: 16000
  window: hamming
  n_mels: 80
  frame_length: 25
  frame_shift: 10
model_conf:
  encoder_conf: {...}
  decoder_conf: {...}
```

---

## 4. What FunASR Does Well → RETAINED

| Pattern | Description |
|---|---|
| Registry + decorator | `@tables.register("model_classes", key="paraformer-zh")` |
| Config YAML | `omegaconf.OmegaConf.load(config_path)` |
| Model hub download | ModelScope + HuggingFace fallback |
| Frontend abstraction | `abs_frontend.py` → concrete implementations |
| `torch.no_grad()` inference | All inferences run inside the context manager |
| CPU fallback | Auto fallback to CPU if CUDA is unavailable |
| `load_audio_text_image_video()` | Multi-format audio loading function |

---

## 5. What FunASR Does Poorly → REWRITTEN

| Issue | Solution in MeetASR |
|---|---|
| AutoModel is 1035 lines, bloated | Split into `AutoModel` + `MeetPipeline` + `Processor` classes |
| `inference_with_vad()` is overly complex | `VADPipeline` class with clear, distinct steps |
| No LLM integration layer | Added a complete `meetasr/llm/` layer |
| No built-in REST API | `meetasr/api/` built with FastAPI |
| Hard to test individual components | Each Processor has its own unit tests |
| Imports all models on startup | Lazy importing, only loads models when needed |
| Training and inference mixed | Split out: inference-only package |
| No structured meeting output format | Defined the `MeetingReport` schema |

---

## 6. Conclusion — Strategy

```
Original FunASR:   Model zoo + training platform
MeetASR:           Production inference + LLM layer

Reused:            Model weights (no re-training)
                   Registry pattern
                   Config YAML format
                   Frontend/tokenizer logic

Newly Written:     Clean pipeline architecture
                   LLM summarization layer
                   REST API
                   Meeting output schema
                   Vietnamese-first design
```
