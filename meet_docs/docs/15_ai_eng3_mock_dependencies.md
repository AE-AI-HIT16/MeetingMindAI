# AI Engineer 3 — Mock Dependencies

> **Mục đích tài liệu này:** Mô tả chính xác tất cả các phần đang được **mock** trong code của AI Engineer 3  
> (các file `meetasr/realtime/events.py`, `meetasr/realtime/job_queue.py`, `meetasr/api/routes/realtime.py`),  
> cùng cấu trúc dữ liệu mà mỗi mock cần tuân theo khi được thay bằng implementation thật.

---

## Bản đồ mock nhanh

| ID | Mock | Dùng ở đâu | Thay bằng gì | Ai làm |
|---|---|---|---|---|
| MOCK-1 | In-memory DB dict | `job_queue.py`, `realtime.py` | SQLModel `realtime_models.py` + `repository` | **Data Engineer** |
| MOCK-2 | `_process_job_mock()` | `job_queue.py` | `worker.process_job_real()` | **AI Engineer 1** |
| MOCK-3 | `_finalize_mock()` | `job_queue.py` | `DocumentPlanner.finalize()` | **AI Engineer 2** |
| MOCK-4 | `_stream_save_to_disk()` | `realtime.py` | `StorageBackend.save_stream()` | **Data Engineer** |

---

## MOCK-1 — DB Models (Data Engineer)

### Vị trí mock trong code AI Eng 3

```
meetasr/realtime/job_queue.py  — _mock_sources, _mock_jobs, _mock_segments, _mock_documents
meetasr/api/routes/realtime.py — doc/ghi các dict tren + import create_source_and_job()
```

### Cấu trúc cần implement (SQLModel)

File mới: `meetasr/db/realtime_models.py`

```python
from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship

class Source(SQLModel, table=True):
    """Media da upload."""
    __tablename__ = "sources"

    id:           str              = Field(primary_key=True)          # UUID
    filename:     str                                                  # ten file goc
    media_type:   str                                                  # "video/mp4" | "audio/mpeg"
    duration_s:   float            = Field(default=0.0)               # giay; cap nhat sau ffmpeg
    storage_path: str                                                  # duong dan local hoac S3 key
    created_at:   datetime         = Field(default_factory=datetime.utcnow)

    jobs:         List["Job"]      = Relationship(back_populates="source")
    documents:    List["Document"] = Relationship(back_populates="source")


class Job(SQLModel, table=True):
    """Trang thai xu ly pipeline cho 1 source."""
    __tablename__ = "jobs"

    id:         str               = Field(primary_key=True)           # UUID
    source_id:  str               = Field(foreign_key="sources.id")
    status:     str               = Field(default="queued")
    # Cac gia tri status hop le: "queued" | "processing" | "done" | "failed"
    progress:   float             = Field(default=0.0)                # 0.0-1.0
    error:      Optional[str]     = Field(default=None)               # message loi neu failed
    created_at: datetime          = Field(default_factory=datetime.utcnow)
    updated_at: datetime          = Field(default_factory=datetime.utcnow)

    source:     Optional[Source]          = Relationship(back_populates="jobs")
    segments:   List["TranscriptSegment"] = Relationship(back_populates="job")


class TranscriptSegment(SQLModel, table=True):
    """Mot cau/doan trong transcript."""
    __tablename__ = "transcript_segments"

    id:       Optional[int] = Field(default=None, primary_key=True)   # auto-increment
    job_id:   str           = Field(foreign_key="jobs.id")
    start_ms: int                                                       # milliseconds
    end_ms:   int                                                       # milliseconds
    speaker:  int           = Field(default=0)                         # speaker index (0-based)
    text:     str                                                       # noi dung cau

    job: Optional[Job] = Relationship(back_populates="segments")


class Document(SQLModel, table=True):
    """Tai lieu duoc sinh ra tu transcript."""
    __tablename__ = "documents"

    id:         str      = Field(primary_key=True)                    # UUID
    source_id:  str      = Field(foreign_key="sources.id")
    mode:       str      = Field(default="live")
    # Cac gia tri mode hop le: "live" | "summary" | "full_text"
    markdown:   str      = Field(default="")                          # noi dung markdown
    created_at: datetime = Field(default_factory=datetime.utcnow)

    source: Optional[Source] = Relationship(back_populates="documents")
```

### Repository functions can them

File: `meetasr/db/realtime_repository.py` (moi)

```python
def create_source(db, id, filename, media_type, storage_path) -> Source: ...
def get_source(db, source_id) -> Optional[Source]: ...
def get_sources(db, skip=0, limit=20) -> list[Source]: ...
def update_source_duration(db, source_id, duration_s) -> None: ...

def create_job(db, source_id) -> Job: ...
def get_job(db, job_id) -> Optional[Job]: ...
def update_job_status(db, job_id, status, progress=None, error=None) -> None: ...

def add_segment(db, job_id, start_ms, end_ms, speaker, text) -> TranscriptSegment: ...
def get_segments_by_job(db, job_id) -> list[TranscriptSegment]: ...

def create_document(db, source_id, mode, markdown="") -> Document: ...
def get_document(db, doc_id) -> Optional[Document]: ...
def update_document(db, doc_id, mode, markdown) -> None: ...
```

### Them vao init_db()

```python
# meetasr/db/connection.py — trong ham init_db()
from meetasr.db import realtime_models  # noqa: F401 — de SQLModel tao bang Phase 2
```

### Cach thay mock trong AI Eng 3's code

Tim cac comment `# MOCK-1` / `# TODO (Data Engineer)` trong `job_queue.py`:

```python
# Truoc (mock):
source_id, job_id = create_source_and_job(filename, media_type, dest_path)

# Sau (that):
from meetasr.db.realtime_repository import create_source, create_job
source = create_source(db, id=str(uuid4()), filename=filename,
                       media_type=media_type, storage_path=dest_path)
job = create_job(db, source.id)
source_id, job_id = source.id, job.id
```

---

## MOCK-2 — Job Processor / ASR Worker (AI Engineer 1)

### Vi tri mock trong code AI Eng 3

```
meetasr/realtime/job_queue.py  — ham _process_job_mock()
```

### Interface can implement

File: `meetasr/realtime/worker.py` (AI Engineer 1 tao)

```python
async def process_job_real(
    job_id: str,
    source_path: str,           # duong dan file media tren disk
    event_bus: EventBus,        # de phat su kien transcript_delta / status
    pipeline: MeetPipeline,     # pipeline ASR Phase 1 (VAD + ASR + punc + speaker)
) -> dict:
    """
    Xu ly file media: ffmpeg -> VAD -> chunk -> ASR -> diarization.
    Phat su kien qua event_bus trong qua trinh xu ly.

    Returns:
        {
            "num_segments": int,
            "duration_ms":  int,
            "segments": [
                {
                    "start_ms": int,
                    "end_ms":   int,
                    "speaker":  int,    # speaker index sau OnlineSpeakerMatcher
                    "text":     str,
                }
            ]
        }
    """
```

### Cac su kien worker can phat qua EventBus

```python
# Trang thai tong
await event_bus.publish(job_id, {"type": "status", "stage": "extracting_audio", "progress": 0.05})
await event_bus.publish(job_id, {"type": "status", "stage": "transcribing",     "progress": 0.40})
await event_bus.publish(job_id, {"type": "status", "stage": "generating_doc",   "progress": 0.85})

# Moi cau transcript xong
await event_bus.publish(job_id, {
    "type": "transcript_delta",
    "segment": {"start_ms": 0, "end_ms": 3200, "speaker": 0, "text": "Xin chao."},
})

# Doc delta (khi AI Eng 2 tich hop DocumentPlanner)
await event_bus.publish(job_id, {
    "type": "doc_delta",
    "section_id": "s1",
    "markdown": "## Muc tieu\n...",
})

# Khi re-cluster speaker xong (sau khi done) — AI Engineer 1 them
await event_bus.publish(job_id, {
    "type": "speaker_update",
    "mapping": {"0": "Nguoi noi A", "1": "Nguoi noi B"},
})
```

### Cach thay mock trong AI Eng 3's code

Trong `job_queue.py`, tim block `# MOCK-2`:

```python
# Truoc (mock):
result = await _process_job_mock(job_id, source_path, event_bus)

# Sau (that):
from meetasr.realtime.worker import process_job_real
from meetasr.api.dependencies import get_pipeline
result = await process_job_real(
    job_id, source_path, event_bus, pipeline=get_pipeline()
)
```

---

## MOCK-3 — Document Finalizer / DocumentPlanner (AI Engineer 2)

### Vi tri mock trong code AI Eng 3

```
meetasr/realtime/job_queue.py  — ham _finalize_mock()
```

### Interface can implement

File: `meetasr/llm/planner.py` (AI Engineer 2 tao)

```python
class DocumentPlanner:

    def __init__(self, llm) -> None:
        """
        Args:
            llm: LLM client (openai_client, ollama_client, ...) tu meetasr/llm/
        """

    async def finalize(
        self,
        segments: list[dict],   # list {start_ms, end_ms, speaker, text}
        mode: str,              # "summary" | "full_text"
    ) -> str:
        """
        Sinh markdown cuoi cung tu danh sach segments.

        - mode="summary":   PLAN (1 lan goi LLM sinh outline) + WRITE (map-reduce tung section)
        - mode="full_text": ghep toan van + 1 luot LLM lam muot chuyen doan

        Returns:
            str: Markdown document hoan chinh.
        """

    async def plan_only(self, sample_text: str) -> dict:
        """
        Buoc PLAN: LLM tu de xuat outline tu doan mau transcript.

        Returns:
            {
                "content_kind": "cuoc hop cong viec | bai giang | podcast ...",
                "outline": [
                    {"id": "s1", "heading": "...", "kind": "summary"},
                    {"id": "s2", "heading": "...", "kind": "key_points"},
                    ...
                ]
            }
        """

    async def write_one_section(
        self,
        section: dict,          # {"id": "s1", "heading": "...", "kind": "..."}
        transcript_chunk: str,  # doan transcript lien quan
        existing_outline: list, # danh sach section hien tai (context)
    ) -> str:
        """
        Buoc WRITE cho 1 section — dung trong doc realtime (doc_delta).
        Returns: markdown content cua section.
        """
```

### Cach thay mock trong AI Eng 3's code

Trong `job_queue.py`, tim block `# MOCK-3`:

```python
# Truoc (mock):
markdown = await _finalize_mock(segments, mode)

# Sau (that):
from meetasr.llm.planner import DocumentPlanner
from meetasr.api.dependencies import get_pipeline
planner = DocumentPlanner(llm=get_pipeline().llm)
markdown = await planner.finalize(segments, mode)
```

---

## MOCK-4 — Storage Backend (Data Engineer)

### Vi tri mock trong code AI Eng 3

```
meetasr/api/routes/realtime.py — ham _stream_save_to_disk()
meetasr/api/routes/realtime.py — endpoint GET /v1/sources/{id}/media (TODO comment)
```

### Interface can implement

File: `meetasr/storage/backend.py` (Data Engineer tao)

```python
from abc import ABC, abstractmethod

class StorageBackend(ABC):

    @abstractmethod
    async def save_stream(
        self,
        file_obj,           # file-like object (UploadFile.file)
        dest_path: str,     # duong dan / S3 key dich
        max_bytes: int,     # gioi han dung luong; raise FileTooLargeError neu vuot
    ) -> int:
        """
        Ghi file tu stream vao storage THEO TUNG CHUNK.
        Kiem tra dung luong TRONG LUC ghi — khong load vao RAM truoc.

        Returns:
            int: Tong so bytes da ghi.

        Raises:
            FileTooLargeError: Neu vuot max_bytes.
        """

    @abstractmethod
    async def get_presigned_url(
        self,
        path: str,
        expires: int = 3600,
    ) -> str:
        """
        Tra presigned URL de trinh duyet truy cap truc tiep (dung cho MinIO/S3).
        Voi LocalStorage: tra duong dan endpoint local.
        """

    @abstractmethod
    async def delete(self, path: str) -> None:
        """Xoa file/object. Dung khi xoa source cascade."""


class LocalStorage(StorageBackend):
    """Luu file local filesystem."""
    def __init__(self, base_dir: str = "data") -> None: ...
    async def save_stream(self, file_obj, dest_path, max_bytes) -> int: ...
    async def get_presigned_url(self, path, expires=3600) -> str: ...
    async def delete(self, path) -> None: ...


class S3Storage(StorageBackend):
    """Dung boto3 + MinIO endpoint."""
    def __init__(self, endpoint_url, bucket, access_key, secret_key) -> None: ...
    async def save_stream(self, file_obj, dest_path, max_bytes) -> int: ...
    async def get_presigned_url(self, path, expires=3600) -> str: ...
    async def delete(self, path) -> None: ...


def get_storage() -> StorageBackend:
    """Singleton — doc tu config de chon LocalStorage hay S3Storage."""
    ...
```

### Cach thay mock trong AI Eng 3's code

Trong `realtime.py`, tim ham `_stream_save_to_disk()`:

```python
# Truoc (mock — da co stream, chi can swap):
bytes_written = await _stream_save_to_disk(file, dest_path, MAX_UPLOAD_BYTES)

# Sau (that):
from meetasr.storage.backend import get_storage
bytes_written = await get_storage().save_stream(file.file, dest_path, MAX_UPLOAD_BYTES)
```

Trong endpoint `/v1/sources/{source_id}/media` — tim comment `# TODO (Data Engineer — MinIO/S3)`:

```python
# Sau (that voi S3 — them vao truoc phan return StreamingResponse hien tai):
from meetasr.storage.backend import get_storage
from fastapi.responses import RedirectResponse
url = await get_storage().get_presigned_url(source["storage_path"], expires=3600)
return RedirectResponse(url, status_code=307)
# (xoa phan StreamingResponse hien tai)
```

---

## Thu tu replace khi cac engineer hoan thien

```
Tuan 2:
  Data Engineer   → them realtime_models.py + repository → replace MOCK-1
  Data Engineer   → them storage/backend.py (LocalStorage) → replace MOCK-4

Tuan 3:
  AI Engineer 1   → hoan thien worker.py → replace MOCK-2
  AI Engineer 2   → hoan thien planner.py → replace MOCK-3
  Data Engineer   → them S3Storage → update MOCK-4 (/media endpoint)
```

> [!IMPORTANT]
> Khi replace MOCK-1, nho them `from meetasr.db import realtime_models  # noqa`  
> vao `init_db()` trong `connection.py` de SQLModel tao bang Phase 2.

> [!NOTE]
> Frontend hooks (`useJobEvents.ts`, `api.ts`) da noi vao cac route cua AI Engineer 3.  
> Khi AI Engineer 1 them su kien `speaker_update`, can update **ca 2 phia cung luc**:  
> backend phat event dung format + frontend xu ly case `"speaker_update"` trong `useJobEvents.ts`.
