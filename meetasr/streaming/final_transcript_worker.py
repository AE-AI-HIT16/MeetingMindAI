from __future__ import annotations

import asyncio
import logging

from sqlmodel import Session

from meetasr.db.connection import engine
from meetasr.db.models_phase2 import (
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
        pipeline,
    ):
        self.queue = queue
        self.pipeline = pipeline


    async def run(self):

        while True:

            job_id = await self.queue.get()

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


                    source = db.get(
                        Source,
                        job.source_id,
                    )

                    if source is None:
                        raise RuntimeError(
                            f"Source {job.source_id} not found"
                        )


                    # -----------------------------
                    # Update trạng thái processing
                    # -----------------------------

                    job.status = JobStatus.PROCESSING
                    job.stage = JobStage.TRANSCRIBING

                    db.add(job)
                    db.commit()


                    storage_path = source.storage_path


                # -----------------------------
                # Chạy ASR ngoài DB session
                # tránh block connection
                # -----------------------------

                transcript = await asyncio.to_thread(
                    self.pipeline.transcribe,
                    storage_path,
                )


                # -----------------------------
                # Lưu transcript segments
                # -----------------------------

                with Session(engine) as db:

                    for sentence in transcript.sentence_info:

                        segment = TranscriptSegment(
                            job_id=job_id,
                            start_ms=int(
                                sentence.start * 1000
                            ),
                            end_ms=int(
                                sentence.end * 1000
                            ),
                            speaker=sentence.speaker,
                            text=sentence.text,
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