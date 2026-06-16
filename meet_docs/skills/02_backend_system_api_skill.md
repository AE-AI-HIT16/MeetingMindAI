# Backend System API Design Skill — MeetASR

This document outlines the guidelines and technical patterns for designing the REST API backend layer in MeetASR using **FastAPI**. It focuses on asynchronous execution, database transaction management, and robust error handling.

---

## 1. Principles of REST API Design in AI Systems

ASR and LLM models are computationally intensive and require significant execution time (seconds to minutes). The API layer must be designed to avoid blocking client HTTP connections.

### 1.1 Asynchronous Background Tasks
- **Standard:** The API must not process long-running transcription and summarization jobs synchronously, as it will trigger HTTP timeouts.
- **Workflow:**
  1. The API accepts the file upload and saves it to a temporary directory.
  2. Create a meeting record in the Database with status `'pending'` or `'processing'`.
  3. Respond immediately to the client with HTTP status `202 (Accepted)` returning the `meeting_id` and initial state.
  4. Delegate the actual processing pipeline (VAD + ASR + LLM) to a background worker (`fastapi.BackgroundTasks` or Celery).
  5. The client polls the status endpoint or subscribes to SSE/Websockets to receive progress updates.

### 1.2 Database Connections & Transaction Lifecycle
- Inject database sessions using FastAPI's dependency injection (`Depends`).
- Ensure sessions are closed immediately after request termination using a generator context.
- Group complex multi-table writes (such as saving transcripts along with associated action items) into a single database transaction (`session.commit()`). If any operation fails, rollback the transaction (`session.rollback()`).

---

## 2. Input Validation & Standardized Error Schema

### 2.1 Input Schema Validation
- All incoming requests with JSON bodies must be validated using Pydantic/SQLModel schemas.
- File uploads must be checked for allowed extensions (`.wav`, `.mp3`, `.m4a`).

### 2.2 Global Exception Handler
Any unhandled exceptions must be caught by a global FastAPI exception handler to prevent leakages of system internals (e.g., raw SQL queries, stack traces) and to return unified JSON error messages.

```json
{
  "detail": "Descriptive error message",
  "error_code": "RESOURCE_NOT_FOUND",
  "status_code": 404
}
```

---

## 3. Reference Implementation & Test Cases

### 3.1 Async Endpoint Example (`meetasr/api/routes.py`)

```python
import uuid
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends
from sqlmodel import Session
from meetasr.schemas import MeetingStatusResponse
from meetasr.db.connection import get_db
from meetasr.db.repository import create_meeting, update_meeting_status
from meetasr.pipeline import MeetPipeline

router = APIRouter(prefix="/v1/meeting", tags=["Meeting"])

def run_transcription_pipeline(meeting_id: str, audio_path: str, db: Session):
    """Worker task executing pipeline ASR + LLM in the background."""
    try:
        pipeline = MeetPipeline.from_default_config()
        report = pipeline.process(audio_path)
        
        # Save results to DB
        from meetasr.db.repository import save_meeting_result
        save_meeting_result(db, meeting_id, report)
        update_meeting_status(db, meeting_id, "completed")
    except Exception as e:
        import logging
        logging.error(f"Failed to process meeting {meeting_id}: {str(e)}")
        update_meeting_status(db, meeting_id, "failed")

@router.post("/process", response_model=MeetingStatusResponse, status_code=202)
async def process_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Validate file extension
    if not file.filename.endswith((".wav", ".mp3", ".m4a")):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .wav, .mp3, or .m4a")

    # 2. Assign unique ID and save file temporarily
    meeting_id = str(uuid.uuid4())
    temp_path = f"/tmp/{meeting_id}_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # 3. Create initial database record
    create_meeting(db, id=meeting_id, title=file.filename, audio_path=temp_path)

    # 4. Dispatch background task
    background_tasks.add_task(run_transcription_pipeline, meeting_id, temp_path, db)

    # 5. Return immediate response
    return MeetingStatusResponse(meeting_id=meeting_id, status="processing")
```

### 3.2 Mock-based API Integration Test
AI Agents must test endpoint routing behavior using FastAPI's `TestClient`:

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from meetasr.api.app import app

client = TestClient(app)

def test_upload_invalid_file_format():
    # Send a text file instead of audio
    response = client.post(
        "/v1/meeting/process",
        files={"file": ("test.txt", b"plain text data", "text/plain")}
    )
    # Assert HTTP 400 Bad Request
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]
```

---

## 4. AI Verification Checklist

When assessing or developing the API Backend layer, the Agent must verify:
- [ ] Do endpoints dispatch long-running model executions to asynchronous background tasks?
- [ ] Are database sessions injected dynamically, ensuring isolation across request lifecycles?
- [ ] Does the global handler intercept server exceptions and format them into structured JSON?
- [ ] Are raw file uploads validated for allowed mime types or extensions before processing?
- [ ] Do unit tests mock model inference and cover HTTP status codes (200, 202, 400, 500)?
