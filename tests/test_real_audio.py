# tests/test_real_audio.py — Test SenseVoice WITH timestamps enabled
from meetasr.auto.auto_pipeline import AutoPipeline
import numpy as np
from meetasr.utils.audio import load_audio
from meetasr.utils.timestamp import merge_vad_segments, build_sentence_info

pipe = AutoPipeline.from_config({
    "asr": {"model": "sensevoice-small"},
    "vad": {"model": "fsmn-vad"},
    "spk": {"model": "cam++"},
})

audio = load_audio("tests/data/test1.wav")
duration = len(audio) / 16000
print(f"Audio: {duration:.1f}s")

# VAD
raw_segs = pipe.vad.detect(audio)
merged = merge_vad_segments(raw_segs)
print(f"VAD: {len(merged)} segments")

# Slice audio
chunks = []
pad_ms = 100
total_ms = int(duration * 1000)
for seg in merged:
    start_ms = max(0, seg.start_ms - pad_ms)
    end_ms = min(total_ms, seg.end_ms + pad_ms)
    chunk = audio[int(start_ms/1000*16000):int(end_ms/1000*16000)]
    if len(chunk) > 0:
        chunks.append(chunk)

# Test SenseVoice with output_timestamp=True (bypass wrapper)
import torch
pipe.asr._ensure_loaded()
results_ts = []
with torch.no_grad():
    for chunk in chunks:
        inference_kwargs = dict(pipe.asr._inference_kwargs)
        raw = pipe.asr._model.inference(
            data_in=[chunk],
            language="auto",
            use_itn=True,
            output_timestamp=True,  # ← KEY FIX
            **inference_kwargs,
        )
        if isinstance(raw, tuple):
            raw = raw[0]
        if isinstance(raw, list):
            results_ts.extend(raw)

print(f"\n=== ASR output_timestamp=True ===")
for i, r in enumerate(results_ts):
    text = r.get("text", "")
    ts = r.get("timestamp", [])
    words = r.get("words", "")
    print(f"  [{i}] text='{text[:100]}'")
    print(f"       ts_count={len(ts)} words_count={len(words) if isinstance(words, list) else 'N/A'}")
    if ts:
        print(f"       ts sample: {ts[:5]}")
    if isinstance(words, list) and words:
        print(f"       words sample: {words[:5]}")
