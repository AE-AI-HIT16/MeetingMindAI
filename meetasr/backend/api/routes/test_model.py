from fastapi import APIRouter, UploadFile, File, Request
import logging

from meetasr.backend.api.dependencies import save_upload, safe_remove
from meetasr.backend.utils.audio import load_audio


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/debug",
    tags=["debug"]
)


@router.post("/vad")
async def test_vad(
    request: Request,
    file: UploadFile = File(...)
):
    # lấy pipeline thật đang chạy
    pipeline = request.app.state.pipeline

    if pipeline is None:
        return {
            "error": "Pipeline is not initialized"
        }

    vad = pipeline.vad

    if vad is None:
        return {
            "error": "VAD is not enabled"
        }


    audio_path = await save_upload(file)

    try:
        # dùng chung preprocessing với ASR
        audio = load_audio(audio_path)


        # debug sau normalize
        print(
            "audio shape:",
            audio.shape,
            "dtype:",
            audio.dtype,
            "duration:",
            len(audio) / 16000,
            "sec"
        )


        # gọi đúng hàm detect của instance hiện tại
        segments = vad.detect(
            audio,
            cache={},
            is_final=True,
        )


        result = []

        for i, seg in enumerate(segments):
            result.append(
                {
                    "index": i,
                    "start_sec": round(
                        seg.start_ms / 1000,
                        3
                    ),
                    "end_sec": round(
                        seg.end_ms / 1000,
                        3
                    ),
                    "duration_sec": round(
                        (seg.end_ms - seg.start_ms) / 1000,
                        3,
                    ),
                }
            )


        return {
            "vad_type": str(type(vad)),
            "audio_duration_sec": round(
                len(audio) / 16000,
                3,
            ),
            "num_segments": len(segments),
            "segments": result,
        }

    except Exception as e:
        logger.exception("VAD test failed")
        return {
            "error": str(e)
        }

    finally:
        safe_remove(audio_path)