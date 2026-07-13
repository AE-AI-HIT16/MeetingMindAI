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
- Kết thúc: **(a) Tóm tắt** → chạy `DocumentPlanner` (xem
  [14_document_planner_design.md](14_document_planner_design.md)) trên toàn transcript — LLM tự
  đề xuất outline phù hợp với chính nội dung (không khoá cứng theo "cuộc họp"), rồi map-reduce
  viết từng mục; **(b) Toàn văn** → ghép các section + hiệu đính lần cuối (1 lượt LLM làm mượt
  chuyển đoạn, không đổi nội dung).

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

> **Cập nhật trạng thái:** một bản v1 chạy được **end-to-end** đã tồn tại — backend (upload →
> chunk transcribe → speaker matching → finalize → export markdown) VÀ frontend Next.js đã nối
> thật vào nhau (không còn mock), xem [13_phase2_limitations_and_fixes.md](13_phase2_limitations_and_fixes.md)
> để biết chính xác phần nào đã xong / còn thiếu. **Việc của 4 người dưới đây là hoàn thiện và làm
> cứng cáp phần đã có** theo các đề xuất trong doc 13 + 14 — không phải xây từ đầu. Cột **"Đã có sẵn"**
> chỉ đúng file hiện tại trong repo; cột **"Frontend liên quan"** để mỗi người biết UI của phần
> mình nằm ở đâu, không cần hỏi lại "frontend đâu rồi".

### Sơ đồ phụ thuộc

```
[Data Eng: Storage/Media/DB — đã có khung] ──► [AI Eng 1: Worker + Speaker — đã có v1] ──► [AI Eng 3: WS/Route — đã có v1]
                                                                                                    │
[AI Eng 2: DocumentPlanner (đã có v1) + Export (chưa có)] ◄────────────────────────────────────────┘
                     │
                     ▼
        [Cả nhóm: hoàn thiện theo doc 13, polish M7, review frontend cùng nhau]
```

### Bảng nhiệm vụ chi tiết

| Vai trò | Đã có sẵn (v1, review & hoàn thiện) | Việc cần làm tiếp (theo doc 13/14) | Frontend liên quan | Verify |
|---|---|---|---|---|
| **AI Engineer 1**<br>*Media & Speaker* | `meetasr/media.py` (ffmpeg extract), `meetasr/realtime/worker.py` (chunk 30s tuần tự), `meetasr/realtime/speaker.py` (`OnlineSpeakerMatcher` — centroid cosine + EMA, đã khắc phục speaker không nhất quán) | 1. Đổi cắt chunk cố định 30s → cắt theo **ranh giới VAD** (doc 13 §4).<br>2. Tinh chỉnh `THRESHOLD` trong `speaker.py` bằng dữ liệu thật.<br>3. **Re-cluster offline cuối phiên** (doc 13 §5): dùng `CommonClustering` sẵn có, phát sự kiện `speaker_update` mới. | `useJobEvents.ts` cần thêm case xử lý `speaker_update` (đổi nhãn speaker đã hiện) — hook đã có chỗ `switch(e.type)` để thêm case này. | Test file 2 người nói ≥ 2 phút: nhãn ổn định xuyên suốt; sau re-cluster khớp bản offline Phase 1 |
| **AI Engineer 2**<br>*LLM Document Planner & Export* | `meetasr/llm/planner.py` (`DocumentPlanner` — Plan→Write, outline tự sinh theo nội dung, thay `MeetingSummarizer` cho Phase 2), `meetasr/llm/prompts/plan_vi.txt` + `write_section_vi.txt`, `meetasr/schemas_doc.py` (`DocSection`/`DocumentReport`), nối sẵn vào `doc_gen.py` + route finalize | 1. **Doc realtime bằng LLM thật** (doc 13 §3 + doc 14 mục 7.2): hiện `doc_delta` trong `worker.py` chỉ ghép text thô — gọi `planner.plan_only()`/`write_one_section()` theo ngưỡng ký tự tích luỹ.<br>2. **Export PDF/DOCX** (doc 13 §6): thư mục `meetasr/export/` mới, weasyprint + font Noto Sans, endpoint export.<br>3. Chạy test plan ở doc 14 mục 10 (outline đa dạng theo domain, không bịa mục rỗng, đếm số lần gọi LLM cho transcript dài). | `DocumentView.tsx` đã có nút "Xuất" chỗ (`ExportMenu` — hiện chỉ tải media) — nối vào endpoint export khi xong; `sec.kind` đã có chỗ trong `types.ts`/`useJobEvents.ts` (`DocSection.kind?`) chờ backend gửi để chọn icon theo loại mục. | Transcript 30 phút → outline khác nhau theo loại nội dung (không luôn là summary/topics/actions/decisions); PDF đọc đúng dấu tiếng Việt |
| **AI Engineer 3**<br>*Backend API & Realtime Channel* | `meetasr/api/routes/realtime.py` (toàn bộ REST + WS `/v1/jobs/{id}/events` với **replay từ DB** khi reconnect), `meetasr/realtime/events.py` (EventBus), `meetasr/realtime/worker.py` (hàng đợi tuần tự 1-consumer — đã khắc phục tranh GPU) | 1. **Backpressure** (doc 13 §8): `asyncio.Queue` trong `events.py` hiện không giới hạn — thêm `maxsize` + chiến lược rớt bớt sự kiện cũ.<br>2. Giới hạn dung lượng khi **stream ghi đĩa** (`storage.save_stream`), không chỉ đọc hết vào RAM rồi mới check.<br>3. Endpoint xem vị trí hàng đợi (nice-to-have) khi nhiều job xếp hàng. | `frontend-next/src/lib/api.ts` + `hooks/useJobEvents.ts` đã nối thật vào các route này (không còn mock) — khi đổi protocol (vd thêm `speaker_update`), sửa **cả 2 phía cùng lúc**, dùng `types.ts` làm hợp đồng chung. | 3 upload liên tiếp → xếp hàng đúng thứ tự, `/v1/health` vẫn phản hồi nhanh khi đang xử lý; client ngắt kết nối giữa chừng rồi nối lại không mất dữ liệu |
| **Data Engineer**<br>*Storage, DB & Library* | `meetasr/storage/backend.py` (local filesystem, đã tách interface), `meetasr/db/realtime_models.py` (`Source`/`TranscriptSegment`/`Document`), route `GET /v1/sources`, `GET /v1/sources/{id}/media` (hỗ trợ Range) | 1. **Chuyển sang MinIO/S3** (doc 13 §7): thêm `S3Storage` cùng interface (`docker-compose.yml` đã có MinIO sẵn), đổi `/media` sang trả presigned URL.<br>2. Endpoint xoá source cascade + thống kê dung lượng (M7).<br>3. Data model cho `speaker_update` mapping nếu AI Eng 1 cần lưu lịch sử re-cluster. | Trang **Thư viện** (`frontend-next/src/app/page.tsx`) đã fetch thật `GET /v1/sources` + poll 5s; **Document viewer** (`DocumentView.tsx`) đã phát `<video>/<audio>` qua `api.mediaUrl()` — khi đổi sang presigned URL S3, chỉ cần `api.mediaUrl()` trả URL khác, component không cần sửa. | Upload mp4 1GB không tràn RAM; sau khi chuyển S3, `<video>` vẫn tua được qua presigned URL; xoá source không để rác file lẫn record MinIO |

### Lịch gợi ý (6 tuần)

> **Lưu ý:** lịch dưới đây viết cho lúc bắt đầu từ số 0. Vì v1 đã chạy end-to-end (backend + frontend
> đã nối thật), nhóm bắt đầu từ **Tuần 1 = đọc code v1 + doc 13/14**, rồi làm các mục "còn lại" ở
> bảng trên — có thể rút ngắn hơn 6 tuần tuỳ năng lực đã có sẵn.

| Tuần | AI Eng 1 | AI Eng 2 | AI Eng 3 | Data Eng | Frontend (cả nhóm hỗ trợ) |
|---|---|---|---|---|---|
| 1 | Đọc `worker.py`/`speaker.py`, đo tỉ lệ đúng speaker trên data thật | Đọc `planner.py` + doc 14, chạy test plan mục 10 doc 14 | Đọc `realtime.py`/`events.py`, đo hành vi reconnect thật | Đọc `storage/backend.py`, lên kế hoạch S3Storage | Đã có sẵn (v1) — rà soát UI thật với backend thật, ghi lại bug |
| 2 | Cắt chunk theo VAD (doc 13 §4) | Nối `DocumentPlanner` vào doc realtime (`worker.py` doc_delta) | Thêm backpressure cho `EventBus` (doc 13 §8) | Viết `S3Storage`, đổi `/media` sang presigned URL | Thêm xử lý `speaker_update` trong `useJobEvents.ts` |
| 3 | Re-cluster offline cuối phiên (doc 13 §5) + phát `speaker_update` | Export PDF/DOCX (doc 13 §6) | Giới hạn dung lượng khi stream ghi đĩa | Xoá cascade + thống kê dung lượng (M7) | Nối nút Export trong `DocumentView.tsx` vào endpoint mới |
| 4 | — Cả nhóm: test tích hợp lại toàn bộ luồng sau khi mỗi người xong phần mình — | | | | |
| 5 | — Polish chung: tinh chỉnh ngưỡng speaker, đo chi phí LLM/latency, sửa bug tích hợp — | | | | |
| 6 | — Buffer, demo thử với data thật, chuẩn bị demo cuối — | | | | |

## 6. Rủi ro chính

| Rủi ro | Giảm thiểu |
|---|---|
| LLM sinh docs lặp/lan man khi gọi nhiều lần liên tiếp | Section-based (chỉ thêm/sửa section cuối) + đưa outline hiện tại vào prompt; test sớm với transcript dài tuần 1 |
| PDF tiếng Việt lỗi font | Chốt weasyprint + font Noto Sans nhúng ngay ở PoC export |
| Upload file lớn (video 1–2GB) làm nghẽn server | Stream multipart to disk, không đọc vào RAM; giới hạn dung lượng cấu hình được |
| GPU chỉ có 1, nhiều job | Queue tuần tự ngay từ M0 — không hứa xử lý song song |
| Speaker không nhất quán giữa các chunk | Centroid matching (AI Eng 1, task 3); fallback: re-cluster toàn bộ khi job xong |
