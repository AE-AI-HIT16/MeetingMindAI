# Phase 2 — MeetingMind "NotebookLM cho Audio/Video"

> **Tầm nhìn Phase 2:** Người dùng upload **video/audio** (thay vì tài liệu như NotebookLM),
> hệ thống chuyển thành **văn bản có cấu trúc (docs)** và xuất **PDF/DOCX**.
> Điểm nhấn: **tạo docs realtime** — media được xử lý đến đâu, nội dung docs hiện ra đến đó.
> Kết thúc, người dùng chọn: **tóm tắt** văn bản hoặc **xuất toàn bộ** văn bản đã tổng hợp.
> Audio/video/docs đều được **lưu trữ trên web** để xem lại.
>
> Doc này thay thế kế hoạch realtime meeting cũ ([09_realtime_phase2.md](09_realtime_phase2.md)).
> Kế hoạch frontend Next.js: [12_frontend_nextjs_plan.md](12_frontend_nextjs_plan.md).

---

## 1. Tính năng người dùng (User Features)

| # | Tính năng | Mô tả |
|---|---|---|
| F1 | **Upload media** | Upload video (mp4, mkv, webm…) hoặc audio (wav, mp3, m4a…). Video được tách audio tự động. |
| F2 | **Realtime doc generation** | Ngay khi upload, media được xử lý theo chunk; transcript + nội dung docs **hiện dần trên màn hình** (không chờ xử lý xong toàn bộ). |
| F3 | **Chọn đầu ra cuối phiên** | Xử lý xong, user chọn: (a) **Tóm tắt** — LLM sinh bản tóm tắt có cấu trúc (chủ đề, quyết định, action items), hoặc (b) **Toàn văn** — xuất nguyên văn bản đã tổng hợp (đã chấm câu, chia đoạn, gán speaker). |
| F4 | **Xuất file** | Tải docs dưới dạng **PDF** hoặc **DOCX** (và Markdown). |
| F5 | **Thư viện (Library)** | Mọi media đã upload + docs đã sinh được lưu trên server; user xem lại, phát lại media, mở lại docs bất cứ lúc nào. |

## 2. So với hiện trạng (Phase 1 đã có gì)

**Đã có (tái dùng):** pipeline ASR offline hoàn chỉnh (VAD → ASR → chấm câu → diarization),
lớp LLM summarizer (chunking, map-reduce, prompt tiếng Việt), FastAPI + SQLModel CRUD, DB SQLite.

**Phải xây mới:**
1. Tách audio từ video (ffmpeg).
2. Xử lý **incremental**: pipeline chạy theo chunk và phát sự kiện dần thay vì trả 1 lần.
3. Kênh **WebSocket/SSE** đẩy nội dung realtime về trình duyệt.
4. Lớp **sinh docs**: chuyển transcript thô → văn bản có cấu trúc (heading theo chủ đề, đoạn văn) + tóm tắt theo lựa chọn.
5. **Export PDF/DOCX**.
6. **Lưu trữ file** (media + docs) có tổ chức trên server.
7. **Frontend mới** bằng Next.js + TypeScript (doc 12).

## 3. Kiến trúc

```
┌────────────────────────────── Next.js Frontend ──────────────────────────────┐
│  Upload page → Processing page (docs hiện dần) → Choice → Document viewer     │
└──────────┬──────────────────────────▲──────────────────────────▲─────────────┘
           │ POST /sources (upload)   │ WS/SSE: sự kiện realtime │ REST: docs, export
┌──────────▼──────────────────────────┴──────────────────────────┴─────────────┐
│                              FastAPI Backend                                  │
│                                                                               │
│  [Upload handler] → lưu file → tạo Job → đẩy vào Job Queue                    │
│                                                                               │
│  [Processing Worker]  (chạy nền, GPU)                                         │
│   ffmpeg tách audio → cắt chunk (~30s theo ranh giới VAD)                     │
│   → mỗi chunk: ASR + punc + diarization  → sự kiện `transcript_delta`         │
│   → mỗi N chunk: LLM sinh/ cập nhật section docs → sự kiện `doc_delta`        │
│   → xong: sự kiện `done` → user chọn → LLM summarize HOẶC format toàn văn     │
│                                                                               │
│  [Storage]  data/media/{id}.*  ·  data/docs/{id}.md  ·  export PDF/DOCX       │
│  [DB — SQLModel]  sources · jobs · documents · segments                       │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Điểm thiết kế quan trọng — "realtime" ở đây là *streaming processing*, không phải live mic:**
file đã upload đầy đủ, nhưng thay vì bắt user chờ 5–10 phút, backend xử lý tuần tự theo chunk
và **đẩy kết quả từng phần** về UI. Đơn giản hơn nhiều so với live-mic streaming (không cần
LocalAgreement, không cần online diarization) mà vẫn cho trải nghiệm "nói đến đâu chữ hiện đến đó".
Kiến trúc này cũng mở đường cho live-mic sau này (chỉ thay nguồn chunk).

### 3.1 Luồng sự kiện WebSocket (protocol)

`WS /v1/jobs/{job_id}/events` — Server → Client:

```json
{"type": "status",           "stage": "extracting_audio" | "transcribing" | "generating_doc", "progress": 0.42}
{"type": "transcript_delta", "segment": {"start_ms": 61000, "end_ms": 92000, "speaker": 1, "text": "..."}}
{"type": "doc_delta",        "section_id": "s3", "markdown": "## Ngân sách Q3\n..."}   // thêm/ghi đè 1 section
{"type": "done",             "duration_ms": 3541000, "num_segments": 87}
{"type": "error",            "code": "...", "message": "..."}
```

Client → Server (sau `done`): qua REST — `POST /v1/documents/{id}/finalize` body
`{"mode": "summary" | "full_text"}`.

### 3.2 Lớp sinh docs realtime (điểm khó nhất về LLM)

- Buffer transcript; mỗi khi đủ ~1500–2500 ký tự final (hoặc phát hiện chuyển chủ đề),
  gọi LLM: *"đây là phần transcript tiếp theo + outline hiện tại → viết/cập nhật section"*.
- Docs được quản lý dạng **danh sách section** (id, heading, markdown) — LLM chỉ được thêm
  section mới hoặc sửa section cuối, không viết lại toàn bộ → UI cập nhật mượt, chi phí token thấp.
- Kết thúc: **(a) Tóm tắt** → chạy `MeetingSummarizer` (Phase 1) trên toàn transcript, sinh docs
  tóm tắt có cấu trúc; **(b) Toàn văn** → ghép các section + hiệu đính lần cuối (1 lượt LLM
  làm mượt chuyển đoạn, không đổi nội dung).

### 3.3 Lưu trữ

- File media & docs: filesystem server `data/media/`, `data/docs/` (đường dẫn lưu trong DB).
  Thiết kế qua 1 interface `StorageBackend` để sau này đổi sang S3/MinIO không sửa logic.
- DB (mở rộng SQLModel Phase 1):
  - `sources` (id, filename, media_type, duration, storage_path, created_at)
  - `jobs` (id, source_id, status: queued|processing|done|failed, progress, error)
  - `documents` (id, source_id, mode: live|summary|full_text, markdown, created_at)
  - `segments` (job_id, start_ms, end_ms, speaker, text) — transcript chi tiết
- Serve media về frontend: endpoint `GET /v1/sources/{id}/media` hỗ trợ **HTTP Range**
  (để `<video>/<audio>` tua được).

### 3.4 Export PDF/DOCX

- Markdown là format gốc của docs. Export phía backend:
  - **PDF**: markdown → HTML (template có style) → PDF bằng `weasyprint`.
  - **DOCX**: `pandoc` (nếu cài được) hoặc `python-docx`.
- Endpoint: `GET /v1/documents/{id}/export?format=pdf|docx|md`.

## 4. Milestones

| MS | Nội dung | Tiêu chí verify |
|---|---|---|
| M0 | **Skeleton**: upload endpoint (lưu file + tạo job), ffmpeg tách audio từ video, bảng DB mới | Upload mp4 → file audio 16kHz xuất hiện trong `data/media/`, job ở trạng thái `queued` |
| M1 | **Worker xử lý theo chunk**: cắt chunk theo VAD, chạy pipeline ASR Phase 1 trên từng chunk, ghi `segments` dần vào DB | File 10 phút → segments xuất hiện dần trong DB trong lúc job đang `processing` |
| M2 | **Kênh realtime**: WS `/v1/jobs/{id}/events` phát `status` + `transcript_delta` | Script client nhận transcript hiện dần đúng thứ tự thời gian |
| M3 | **Sinh docs realtime**: lớp section-based doc generation, sự kiện `doc_delta` | Docs markdown hình thành dần, section có heading hợp lý theo chủ đề |
| M4 | **Finalize**: chọn summary / full_text, lưu `documents` | Cả 2 mode trả docs hợp lệ; full_text giữ đủ nội dung, summary có cấu trúc |
| M5 | **Export + Library API**: PDF/DOCX/MD, list/get sources & documents, serve media có Range | Mở PDF đọc tốt tiếng Việt (font!); `<video>` trên browser tua được |
| M6 | **Tích hợp frontend Next.js** (doc 12) end-to-end | Demo: upload video → xem docs hiện dần → chọn tóm tắt → tải PDF → thấy trong Library |
| M7 | **Polish**: job đồng thời, resume job lỗi, xóa source, dung lượng lưu trữ | 2 job song song không tranh GPU (queue tuần tự); xóa source xóa cả file + docs |

## 5. Phân công nhiệm vụ (3 AI Engineer + 1 Data Engineer)

> Frontend Next.js do **trưởng nhóm (duykhanh)** đảm nhiệm chính theo doc 12; các thành viên
> hỗ trợ tích hợp API ở M6.

### Sơ đồ phụ thuộc

```
[Data Eng: Upload + Storage + DB (M0)] ──► [AI Eng 1: Chunk worker (M1)] ──► [AI Eng 3: WS events (M2)]
                                                                                      │
[AI Eng 2: Doc generation + Finalize (M3, M4)] ◄──────────────────────────────────────┘
                     │
                     ▼
[AI Eng 2 + Data Eng: Export + Library (M5)] ──► [Cả nhóm: tích hợp frontend (M6), polish (M7)]
```

### Bảng nhiệm vụ chi tiết

| Vai trò | Milestone | File phụ trách | Nhiệm vụ cụ thể | Verify |
|---|---|---|---|---|
| **AI Engineer 1**<br>*Media & Chunk Processing* | M1, hỗ trợ M7 | `meetasr/processing/chunker.py`, `meetasr/processing/worker.py` (mới) | 1. Cắt audio thành chunk ~30s **theo ranh giới VAD** (không cắt giữa câu).<br>2. Worker chạy nền: với mỗi chunk gọi pipeline Phase 1 (ASR + punc + diarization), phát callback `on_segment`.<br>3. Giữ **nhất quán speaker giữa các chunk**: embedding CAM++ của segment mới so với centroid các speaker đã thấy (cosine > 0.7 → cùng người).<br>4. Đo và tối ưu: chunk đầu tiên phải ra kết quả < 15s sau khi job bắt đầu. | Test file 10 phút: transcript đầy đủ khớp bản chạy offline 1 lần; speaker nhất quán xuyên chunk; segments ghi DB dần |
| **AI Engineer 2**<br>*LLM Doc Generation & Export* | M3, M4, M5 (export) | `meetasr/llm/doc_generator.py`, `meetasr/llm/prompts/doc_*.txt`, `meetasr/export/` (mới) | 1. Thiết kế prompt tiếng Việt: transcript delta + outline hiện tại → section markdown (JSON: `{section_id, heading, markdown}`).<br>2. Quản lý danh sách section, phát hiện chuyển chủ đề, chỉ sửa section cuối/thêm mới.<br>3. Finalize 2 mode: `summary` (tái dùng `MeetingSummarizer`) và `full_text` (ghép + hiệu đính).<br>4. Export markdown → PDF (weasyprint, nhúng font tiếng Việt) và DOCX. | Docs sinh từ transcript mẫu 30 phút: heading đúng chủ đề, không lặp/mất nội dung; PDF hiển thị đúng dấu tiếng Việt |
| **AI Engineer 3**<br>*Backend API & Realtime Channel* | M2, M6, M7 | `meetasr/api/routes/jobs.py`, `routes/documents.py`, `meetasr/processing/queue.py` (mới), sửa `api/app.py` | 1. Job queue: 1 GPU worker tuần tự, nhiều job xếp hàng, cập nhật `status/progress`.<br>2. WS `/v1/jobs/{id}/events`: nhận callback từ worker (AI Eng 1) + doc generator (AI Eng 2), đẩy sự kiện đúng protocol mục 3.1; client vào giữa chừng nhận lại đủ sự kiện đã qua (replay từ DB).<br>3. REST: finalize, list/get documents.<br>4. M6: hỗ trợ frontend tích hợp (CORS, OpenAPI schema chuẩn để codegen TypeScript). | 2 job đồng thời: 1 chạy 1 chờ, event loop không block; client reconnect vẫn nhận đủ transcript |
| **Data Engineer**<br>*Storage, DB & Library* | M0, M5 (library), M7 | `meetasr/storage/backend.py` (mới), `meetasr/db/models.py`, `db/repository.py` (mở rộng), `meetasr/api/routes/sources.py` | 1. Upload endpoint (multipart, giới hạn dung lượng, validate định dạng), lưu file có tổ chức, gọi ffmpeg tách audio 16kHz mono.<br>2. `StorageBackend` interface + implementation local filesystem.<br>3. 4 bảng DB mới (mục 3.3) + repository CRUD; migration từ schema Phase 1.<br>4. Serve media với HTTP Range; API Library (list sources kèm document status).<br>5. M7: xóa source cascade (file + DB), thống kê dung lượng. | Upload mp4 1GB không tràn RAM (stream to disk); `<video>` tua được; xóa source không để rác file |

### Lịch gợi ý (6 tuần)

| Tuần | AI Eng 1 | AI Eng 2 | AI Eng 3 | Data Eng | Trưởng nhóm (FE) |
|---|---|---|---|---|---|
| 1 | Thiết kế chunker, PoC cắt theo VAD | Thiết kế prompt, PoC sinh section từ transcript mẫu | Chốt protocol WS + OpenAPI spec | M0: upload + ffmpeg + DB | Setup Next.js, design system (doc 12 — F0, F1) |
| 2 | M1: worker + speaker consistency | M3: doc generator hoàn chỉnh | M2: WS events + replay | Storage backend + serve media | Trang Upload + Library (F2) |
| 3 | Tối ưu latency chunk đầu | M4: finalize 2 mode | Job queue + REST documents | Library API | Trang Processing realtime (F3) |
| 4 | Hỗ trợ tích hợp, test E2E backend | M5: export PDF/DOCX | Tích hợp callback 2 module | M5: library hoàn chỉnh | Document viewer + export (F4) |
| 5 | — M6: tích hợp frontend ↔ backend, demo end-to-end — | | | | |
| 6 | — M7: polish (đồng thời, xóa, lỗi), fix bug, chuẩn bị demo — | | | | |

## 6. Rủi ro chính

| Rủi ro | Giảm thiểu |
|---|---|
| LLM sinh docs lặp/lan man khi gọi nhiều lần liên tiếp | Section-based (chỉ thêm/sửa section cuối) + đưa outline hiện tại vào prompt; test sớm với transcript dài tuần 1 |
| PDF tiếng Việt lỗi font | Chốt weasyprint + font Noto Sans nhúng ngay ở PoC export |
| Upload file lớn (video 1–2GB) làm nghẽn server | Stream multipart to disk, không đọc vào RAM; giới hạn dung lượng cấu hình được |
| GPU chỉ có 1, nhiều job | Queue tuần tự ngay từ M0 — không hứa xử lý song song |
| Speaker không nhất quán giữa các chunk | Centroid matching (AI Eng 1, task 3); fallback: re-cluster toàn bộ khi job xong |
