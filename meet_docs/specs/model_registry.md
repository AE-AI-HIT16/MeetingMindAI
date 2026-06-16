# Model Registry Specification — MeetASR

## 1. Registry Design

The Registry is a centralized registration system for all components.
The pattern mimics FunASR but is simplified, retaining only the necessary registries.

```python
# meetasr/register.py

from dataclasses import dataclass, field
import inspect
import logging
import re

@dataclass
class RegisterTables:
    """Central registry for all MeetASR components."""

    # Core component registries
    model_classes: dict = field(default_factory=dict)   # VAD, ASR, Punc, Spk models
    frontend_classes: dict = field(default_factory=dict) # Audio feature extractors
    tokenizer_classes: dict = field(default_factory=dict) # Text tokenizers
    llm_classes: dict = field(default_factory=dict)     # LLM clients

    def register(self, registry_name: str, key: str = None):
        """Decorator to register a class into a named registry."""
        def decorator(cls):
            registry = getattr(self, registry_name)
            reg_key = key if key is not None else cls.__name__
            registry[reg_key] = cls
            return cls
        return decorator

tables = RegisterTables()
```

---

## 2. Model Registry Keys

### VAD Models

| Key | Class | Source Model |
|---|---|---|
| `fsmn-vad` | `FsmnVAD` | `damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` |

### ASR Models

| Key | Class | Source Model |
|---|---|---|
| `sensevoice-small` | `SenseVoice` | `iic/SenseVoiceSmall` |
| `paraformer-zh` | `Paraformer` | `damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| `paraformer-zh-streaming` | `ParaformerStreaming` | `damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` |

### Punctuation Models

| Key | Class | Source Model |
|---|---|---|
| `ct-punc` | `CTTransformerPunc` | `damo/punc_ct-transformer_cn-en-common-vocab471067-large` |

### Speaker Models

| Key | Class | Source Model |
|---|---|---|
| `cam++` | `CAMPlusPlus` | `iic/speech_campplus_sv_zh-cn_16k-common` |

### Frontend Classes

| Key | Class | Description |
|---|---|---|
| `WavFrontend` | `WavFrontend` | Standard wav → fbank (used for Paraformer, SenseVoice) |
| `WhisperFrontend` | `WhisperFrontend` | Whisper-compatible frontend |

### Tokenizer Classes

| Key | Class | Description |
|---|---|---|
| `CharTokenizer` | `CharTokenizer` | Character-level (Chinese/Vietnamese) |
| `SentencePieceTokenizer` | `SPTokenizer` | SentencePiece subword tokenizer |
| `WhisperTokenizer` | `WhisperTokenizer` | Whisper tiktoken-based tokenizer |

### LLM Client Classes

| Key | Class | Description |
|---|---|---|
| `openai` | `OpenAIClient` | OpenAI API + compatible endpoints |
| `ollama` | `OllamaClient` | Ollama local (wraps OpenAIClient) |

---

## 3. Registration Pattern

```python
# meetasr/models/vad/fsmn_vad.py

from meetasr.register import tables
from meetasr.models.vad.abs_vad import AbsVAD

@tables.register("model_classes", key="fsmn-vad")
class FsmnVAD(AbsVAD):
    """FSMN Voice Activity Detection."""

    def __init__(self, **kwargs):
        super().__init__()
        # Initialize from config

    def detect(self, audio: np.ndarray, **kwargs) -> list[Segment]:
        """Detect speech segments in audio."""
        ...
```

```python
# meetasr/frontends/fbank.py

from meetasr.register import tables
from meetasr.frontends.abs_frontend import AbsFrontend

@tables.register("frontend_classes", key="WavFrontend")
class WavFrontend(AbsFrontend):
    def __init__(self, fs: int = 16000, n_mels: int = 80, **kwargs):
        ...

    def forward(self, audio: np.ndarray) -> torch.Tensor:
        """audio [N] → fbank [T, n_mels]"""
        ...
```

---

## 4. Model Config YAML Format

Each model folder (after being downloaded) must contain a `config.yaml`:

```yaml
# Example config.yaml for paraformer-zh
model: Paraformer              # Key in model_classes registry
frontend: WavFrontend          # Key in frontend_classes registry
tokenizer: CharTokenizer       # Key in tokenizer_classes registry

frontend_conf:
  fs: 16000
  window: hamming
  n_mels: 80
  frame_length: 25             # ms
  frame_shift: 10              # ms
  dither: 0.0
  lfr_m: 7
  lfr_n: 6

tokenizer_conf:
  token_list: tokens.txt

model_conf:
  encoder: SANMEncoder
  encoder_conf:
    output_size: 512
    attention_heads: 4
    ...
  decoder: ParaformerDecoderSAN
  decoder_conf:
    ...
```

---

## 5. Auto-registration

FunASR uses `import_submodules()` to automatically import all submodules when the package loads, executing all `@tables.register()` decorators.

MeetASR uses explicit imports in `__init__.py` to maintain better control:

```python
# meetasr/__init__.py

from meetasr.models.vad.fsmn_vad import FsmnVAD
from meetasr.models.asr.sense_voice import SenseVoice
from meetasr.models.asr.paraformer import Paraformer
from meetasr.models.punc.ct_transformer import CTTransformerPunc
from meetasr.models.spk.campplus import CAMPlusPlus
from meetasr.frontends.fbank import WavFrontend
from meetasr.tokenizer.char_tokenizer import CharTokenizer
from meetasr.llm.openai_client import OpenAIClient
from meetasr.llm.ollama_client import OllamaClient

# After importing, tables.model_classes is populated with all registered models
```

**Reasons for choosing explicit imports over auto-import:**
- Easier to debug (we know exactly what has been imported)
- Faster startup time (avoids walking the entire package tree)
- Clearer import errors (does not swallow failures silently)
