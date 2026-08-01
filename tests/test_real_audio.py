"""Optional live SenseVoice test using a real audio fixture."""

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("MEETASR_RUN_LIVE_TESTS") != "1",
    reason="downloads ASR models; set MEETASR_RUN_LIVE_TESTS=1",
)
def test_sensevoice_real_audio_timestamps():
    import torch

    from meetasr.auto.auto_pipeline import AutoPipeline
    from meetasr.utils.audio import load_audio
    from meetasr.utils.timestamp import merge_vad_segments

    pipe = AutoPipeline.from_config(
        {
            "asr": {"model": "sensevoice-small"},
            "vad": {"model": "fsmn-vad"},
            "spk": {"model": "cam++"},
        }
    )

    audio = load_audio("tests/data/test1.wav")
    duration = len(audio) / 16000
    raw_segments = pipe.vad.detect(audio)
    merged = merge_vad_segments(raw_segments)

    chunks = []
    pad_ms = 100
    total_ms = int(duration * 1000)
    for segment in merged:
        start_ms = max(0, segment.start_ms - pad_ms)
        end_ms = min(total_ms, segment.end_ms + pad_ms)
        chunk = audio[int(start_ms / 1000 * 16000):int(end_ms / 1000 * 16000)]
        if len(chunk) > 0:
            chunks.append(chunk)

    pipe.asr._ensure_loaded()
    results = []
    with torch.no_grad():
        for chunk in chunks:
            inference_kwargs = dict(pipe.asr._inference_kwargs)
            raw = pipe.asr._model.inference(
                data_in=[chunk],
                language="auto",
                use_itn=True,
                output_timestamp=True,
                **inference_kwargs,
            )
            if isinstance(raw, tuple):
                raw = raw[0]
            if isinstance(raw, list):
                results.extend(raw)

    assert results
    assert any(result.get("timestamp") for result in results)
