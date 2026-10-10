from __future__ import annotations

import asyncio
import logging

from sqlmodel import Session

from meetasr.backend.db.connection import engine
from meetasr.backend.db.models_phase2 import (
    Job,
    Source,
    JobStatus,
    JobStage,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


class FinalTranscriptWorker:

    def __init__(
        self,
        queue,
        asr_service,
    ):
        self.queue = queue
        self.asr_service = asr_service


    async def run(self):

        while True:

            final_job = await self.queue.get()
            job_id = final_job.job_id
            audio = final_job.audio

            job = None

            try:

                with Session(engine) as db:

                    job = db.get(
                        Job,
                        job_id,
                    )

                    if job is None:
                        logger.warning(
                            "Job %s not found",
                            job_id,
                        )
                        continue


                    # -----------------------------
                    # Update trạng thái processing
                    # -----------------------------

                    job.status = JobStatus.PROCESSING
                    job.stage = JobStage.TRANSCRIBING

                    db.add(job)
                    db.commit()


                # -----------------------------
                # Chạy ASR ngoài DB session
                # tránh block connection
                # -----------------------------

                result = await self.asr_service.transcribe(
                    audio,
                )


                # -----------------------------
                # Lưu transcript segments
                # -----------------------------

                with Session(engine) as db:

                    for seg in result.segments:

                        segment = TranscriptSegment(
                            job_id=job_id,
                            start_ms=seg.start_ms,
                            end_ms=seg.end_ms,
                            speaker=seg.speaker,
                            text=seg.text,
                        )

                        db.add(segment)


                    job = db.get(
                        Job,
                        job_id,
                    )

                    if job is not None:

                        job.status = JobStatus.DONE
                        job.stage = JobStage.TRANSCRIBING
                        job.progress = 1.0

                        db.add(job)

                    db.commit()


                    # -----------------------------------------
                    # Tạo Document (live) qua DocumentService
                    # để frontend truy vấn /v1/documents/{id}
                    # -----------------------------------------

                    from meetasr.backend.services.document_service import DocumentService

                    doc_service = DocumentService(db)
                    source_id = db.get(Job, job_id).source_id
                    transcript_result = doc_service.build_transcript(source_id)
                    markdown = DocumentService.full_text_markdown(transcript_result)
                    doc_service.save_live_document(source_id, markdown)

                    logger.info(
                        "Live document saved for source=%s",
                        source_id,
                    )


                logger.info(
                    "Final transcript completed job=%s segments=%d",
                    job_id,
                    len(transcript.sentence_info),
                )


            except Exception as e:

                logger.exception(
                    "Final transcript failed job=%s",
                    job_id,
                )


                if job is not None:

                    with Session(engine) as db:

                        failed_job = db.get(
                            Job,
                            job_id,
                        )

                        if failed_job is not None:

                            failed_job.status = JobStatus.FAILED
                            failed_job.error = str(e)

                            db.add(failed_job)
                            db.commit()


            finally:

                self.queue.task_done()