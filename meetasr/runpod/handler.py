import logging
import sys

# Configure logging before any other import so startup errors are captured
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
    force=True,
)
logger = logging.getLogger(__name__)

logger.info("Handler starting — importing dependencies...")

# Cache models on Network Volume if mounted, so cold starts skip HuggingFace download
import os as _os
_NV = "/runpod-volume"
if _os.path.isdir(_NV):
    _os.environ.setdefault("HF_HOME", f"{_NV}/hf_cache")
    _os.environ.setdefault("MODELSCOPE_CACHE", f"{_NV}/ms_cache")
    logger.info("Network Volume detected — using %s/hf_cache for model cache.", _NV)
else:
    logger.info("No Network Volume — models will be downloaded to container cache.")

try:
    import runpod
    import base64
    import asyncio
    import os
    logger.info("Core imports OK.")
except Exception as exc:
    logger.exception("Fatal: core import failed: %s", exc)
    sys.exit(1)

# Global instances for model caching (loaded on first job request)
pipeline = None
realtime_pipeline = None

# Cold-start diagnostics, reported by the "warmup" action.
_load_stats: dict = {}


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def init_models():
    global pipeline, realtime_pipeline
    if pipeline is not None:
        return

    import time
    hf_home = os.environ.get("HF_HOME", "")
    # Cache already populated before loading → models come from the volume.
    _load_stats["hf_cache_bytes_before_load"] = _dir_size(hf_home) if hf_home else 0
    _load_stats["load_started_at"] = time.time()
    t0 = time.monotonic()

    logger.info("First job received — loading ML pipeline...")
    try:
        from meetasr.runpod.auto.auto_pipeline import AutoPipeline
        from meetasr.runpod.pipeline_realtime import ASRPipeline
        import meetasr.runpod as _mrp
        _mrp._register_all_models()

        config_path = os.getenv("CONFIG_PATH", "config.yaml")
        logger.info("Loading pipeline from config: %s", config_path)
        pipeline = AutoPipeline.from_yaml(config_path)
        realtime_pipeline = ASRPipeline(
            asr_model=pipeline.asr,
            vad_model=pipeline.vad,
            device=pipeline.device,
        )
        _load_stats["model_load_seconds"] = round(time.monotonic() - t0, 1)
        logger.info("Pipeline loaded successfully in %.1fs.", _load_stats["model_load_seconds"])
    except Exception as exc:
        logger.exception("Fatal: pipeline init failed: %s", exc)
        raise


def _sentence_info_to_dict(si) -> dict:
    return {
        "text": si.text,
        "start": si.start,
        "end": si.end,
        "speaker": si.speaker,
        "char_timestamps": si.char_timestamps,
    }


async def run_handler(job):
    init_models()
    job_input = job.get('input', {})
    action = job_input.get('action')
    logger.info("Handling action: %s", action)

    if action == "transcribe":
        audio_bytes = base64.b64decode(job_input['audio_base64'])
        from meetasr.runpod.utils.audio import load_audio
        audio = load_audio(audio_bytes)
        key = job_input.get('key')

        result = await asyncio.to_thread(pipeline.transcribe, audio, key=key)
        return {
            "key": result.key,
            "text": result.text,
            "duration": result.duration,
            "sentence_info": [_sentence_info_to_dict(s) for s in result.sentence_info],
        }

    elif action == "prepare_incremental":
        audio_bytes = base64.b64decode(job_input['audio_base64'])
        from meetasr.runpod.utils.audio import load_audio
        audio = load_audio(audio_bytes)
        if getattr(pipeline, "diarization_first", False):
            audio_out, vad_segments, speaker_turns, duration_ms = await asyncio.to_thread(
                pipeline.prepare_diarization_first_transcription, audio
            )
        else:
            audio_out, vad_segments, duration_ms = await asyncio.to_thread(
                pipeline.prepare_incremental_transcription, audio
            )
            speaker_turns = None

        response = {
            "vad_segments": [
                {"start_ms": s.start_ms, "end_ms": s.end_ms} for s in vad_segments
            ],
            "duration_ms": duration_ms,
            "speaker_turns": None,
        }
        if speaker_turns is not None:
            response["speaker_turns"] = [
                {"start_ms": t.start_ms, "end_ms": t.end_ms, "speaker": t.speaker}
                for t in speaker_turns
            ]
        return response

    elif action == "transcribe_segment":
        audio_bytes = base64.b64decode(job_input['audio_base64'])
        from meetasr.runpod.utils.audio import load_audio
        from meetasr.runpod.schemas import Segment
        audio = load_audio(audio_bytes)
        language = job_input.get('language', 'auto')
        duration_ms = int(len(audio) / 16000 * 1000)
        segment = Segment(0, duration_ms)
        sentences = await asyncio.to_thread(
            pipeline.transcribe_vad_segment, audio, segment, language=language
        )
        return [_sentence_info_to_dict(s) for s in sentences]

    elif action == "finalize_incremental":
        from meetasr.runpod.schemas import SentenceInfo
        sentences_data = job_input.get('sentences', [])
        sentences = [
            SentenceInfo(
                text=s["text"],
                start=s["start"],
                end=s["end"],
                speaker=s.get("speaker"),
                char_timestamps=s.get("char_timestamps", []),
            )
            for s in sentences_data
        ]

        if 'audio_base64' in job_input:
            audio_bytes = base64.b64decode(job_input['audio_base64'])
            from meetasr.runpod.utils.audio import load_audio
            from meetasr.runpod.schemas import Segment
            audio = load_audio(audio_bytes)
            vad_segments = [
                Segment(s["start_ms"], s["end_ms"])
                for s in job_input.get('vad_segments', [])
            ]
            result = await asyncio.to_thread(
                pipeline.finalize_incremental_transcript, audio, sentences, vad_segments
            )
        else:
            result = await asyncio.to_thread(
                pipeline.finalize_preassigned_transcript, sentences
            )
        return [_sentence_info_to_dict(s) for s in result]

    elif action == "realtime_transcribe":
        audio_bytes = base64.b64decode(job_input['audio_base64'])
        from meetasr.runpod.utils.audio import load_audio
        audio = load_audio(audio_bytes)
        key = job_input.get('key')
        result = await asyncio.to_thread(realtime_pipeline.transcribe, audio, key=key)
        return {
            "key": result.key,
            "text": result.text,
            "duration": result.duration,
            "sentence_info": [_sentence_info_to_dict(s) for s in result.sentence_info],
        }

    elif action == "warmup":
        # Sent by the backend when a realtime session opens so the cold start
        # overlaps with the user speaking. init_models() above did the loading;
        # report where models came from so the Network Volume can be verified.
        import shutil
        import time
        hf_home = os.environ.get("HF_HOME", "")
        mounted = os.path.isdir(_NV)
        info = {
            "status": "ready",
            "worker_id": os.environ.get("RUNPOD_POD_ID"),
            "network_volume_mounted": mounted,
            "hf_home": hf_home,
            "modelscope_cache": os.environ.get("MODELSCOPE_CACHE"),
            "hf_cache_bytes": _dir_size(hf_home) if hf_home else 0,
            "hf_cached_repos": sorted(
                d for d in (os.listdir(os.path.join(hf_home, "hub")) if os.path.isdir(os.path.join(hf_home, "hub")) else [])
                if d.startswith("models--")
            ),
            **_load_stats,
            "loaded_seconds_ago": round(time.time() - _load_stats.get("load_started_at", time.time()), 1),
        }
        if mounted:
            usage = shutil.disk_usage(_NV)
            info["volume_free_gb"] = round(usage.free / 1e9, 1)
            info["volume_total_gb"] = round(usage.total / 1e9, 1)
        return info

    elif action == "realtime_recognize":
        audio_bytes = base64.b64decode(job_input['audio_base64'])
        from meetasr.runpod.utils.audio import load_audio
        audio = load_audio(audio_bytes)
        results = await asyncio.to_thread(realtime_pipeline.asr.recognize, [audio])
        return results

    else:
        return {"error": f"Unknown action: {action}"}


async def handler(job):
    return await run_handler(job)


logger.info("Handler module loaded — starting RunPod serverless worker.")

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
