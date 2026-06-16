# Audio Data Processing Skill — MeetASR

This document outlines the design standards and technical implementations of the audio processing pipeline in MeetASR, aligned with the patterns established in **FunASR**. This serves as a skill guide for AI Agents and developers to build and evaluate audio features.

---

## 1. Principles of Audio Processing in ASR

To achieve optimal speech recognition accuracy, raw audio waveforms must be standardized and transformed into log-mel filterbank (Fbank) representations.

```
[Raw Audio] -> [Resample & Mono] -> [VAD Segmenting] -> [Fbank Extraction] -> [LFR Stacking]
```

### 1.1 Audio Standardization (Resampling)
- **Mandatory Requirement:** All input audio files must be resampled to **16kHz**, single-channel (**mono**), and represented as **16-bit PCM**.
- **Reasoning:** Pre-trained acoustic models in FunASR (SenseVoice, Paraformer, FSMN-VAD) were optimized on datasets conforming to these specs. Mismatched sample rates (e.g., 44.1kHz or 48kHz) will lead to highly inaccurate model decoding.

### 1.2 Voice Activity Detection (VAD Chunking)
- **Standard:** Long audio streams must be segmented into smaller chunks based on voice activity.
- **Principles:**
  - Minimum silence duration to split two sentences is typically set to **800ms - 1000ms**.
  - Maximum segment length is capped at **10s - 30s** to prevent GPU Out of Memory (OOM) errors during inference.
  - The VAD model yields a list of segment timestamps: `[[start_ms, end_ms], ...]`.

---

## 2. Fbank Extraction & LFR (Low Frame Rate) Stacking

These are the primary acoustic features fed into ASR models in the FunASR ecosystem.

### 2.1 Log-Mel Filterbank Extraction
- Compute Short-Time Fourier Transform (STFT) using the following parameters:
  - `frame_length` = 25ms (window size).
  - `frame_shift` = 10ms (hop size).
  - `num_mel_bins` = 80 (Mel filter banks).
- Compute log energy to mimic human auditory perception. This yields a feature matrix of shape $[T, 80]$ where $T$ represents the time frames (roughly 100 frames per second of audio).

### 2.2 Low Frame Rate (LFR) Stacking
To compress the time dimension of the feature matrix (reducing computational costs by 3-4x), FunASR stacks adjacent frames:
- **Standard Parameters:** `lfr_m` = 7 (stack 7 frames), `lfr_n` = 6 (hop size of 6 frames).
- **Execution:** For each frame index $t$ at stride $6$, retrieve $3$ preceding and $3$ succeeding frames, concatenating them into a single $80 \times 7 = 560$ dimensional vector.
- **Output Shape:** $[T', 560]$, where $T' \approx T/6$.

---

## 3. Reference Implementation & Test Cases

### 3.1 LFR Computation (`meetasr/frontends/fbank.py`)
AI Agents must implement LFR stacking conforming to the following structure:

```python
import numpy as np

def compute_lfr_features(fbank_feats: np.ndarray, lfr_m: int = 7, lfr_n: int = 6) -> np.ndarray:
    """Stack Fbank features according to FunASR's LFR specifications."""
    t, d = fbank_feats.shape
    lfr_feats = []
    
    for i in range(0, t, lfr_n):
        # Determine indices to stack
        frame_idx = []
        for j in range(i - (lfr_m - 1) // 2, i + (lfr_m - 1) // 2 + 1):
            # Border padding via index clamping
            idx = max(0, min(j, t - 1))
            frame_idx.append(idx)
        
        # Concatenate 7 frames into a 560-dimensional vector
        stacked_vector = np.concatenate([fbank_feats[idx] for idx in frame_idx], axis=0)
        lfr_feats.append(stacked_vector)
        
    return np.array(lfr_feats, dtype=np.float32)
```

### 3.2 Feature Shape Unit Test
AI Agents must write corresponding tests to validate audio processing outputs:

```python
# tests/test_data_processing.py
import pytest
import numpy as np
from meetasr.frontends.fbank import compute_lfr_features

def test_lfr_stacking_shape():
    # Simulate Fbank features of 100 frames (approx. 1s of audio) with 80 channels
    mock_fbank = np.random.randn(100, 80)
    
    # Process LFR stacking
    lfr_feats = compute_lfr_features(mock_fbank, lfr_m=7, lfr_n=6)
    
    # Expected frame count: ceil(100 / 6) = 17
    expected_frames = int(np.ceil(100 / 6))
    
    assert lfr_feats.shape[0] == expected_frames, f"Unexpected frame count: {lfr_feats.shape[0]}"
    assert lfr_feats.shape[1] == 560, f"Unexpected feature dimension: {lfr_feats.shape[1]}"
    assert lfr_feats.dtype == np.float32, "Data type must be float32"
```

---

## 4. AI Verification Checklist

When assessing or developing audio processing modules, the Agent must verify:
- [ ] Has the input audio sample rate been downsampled to exactly $16000\text{Hz}$?
- [ ] Has the audio channel been mixed down to mono?
- [ ] Does VAD chunking prevent voice cut-offs or truncation at segment borders?
- [ ] Does LFR stacking handle short files (where total frames are less than 7) without crashing?
