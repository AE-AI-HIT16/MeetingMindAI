# Hướng dẫn cho team nghiên cứu source (MeetingMindAI / meetasr)

> Tài liệu onboarding: đọc file này đầu tiên để hiểu **dự án làm gì**, **chạy thế nào**,
> **web đang có chức năng gì**, và **cái gì chưa triển khai / còn dở**. Mọi mô tả dưới đây
> đối chiếu trực tiếp với code thật (đường dẫn ghi kèm), không phải mô tả trên giấy.
>
> - Muốn **lệnh chạy chi tiết** (docker, backend, frontend, CLI, biến môi trường): xem
>   [16_run_commands.md](16_run_commands.md).
> - Muốn hiểu **kiến trúc tổng**: [02_architecture.md](02_architecture.md).
> - Lộ trình & phase: [05_roadmap.md](05_roadmap.md), [11_phase2_notebooklm_plan.md](11_phase2_notebooklm_plan.md),
>   [15_phase3_plan.md](15_phase3_plan.md), [17_phase4_realtime_desktop.md](17_phase4_realtime_desktop.md).

---

## 1. Dự án là gì (1 phút)

MeetASR là bản viết lại FunASR, đơn giản hoá kiến trúc, cộng thêm LLM để **tóm tắt cuộc họp**.
Sản phẩm hiện tại là một web kiểu "**NotebookLM cho audio/video tiếng Việt**":

1. Người dùng **tải file** (hoặc **ghi trực tiếp** từ micro / tab) →
2. Hệ thống **phiên âm + tách người nói (diarization)** theo thời gian thực →
3. Người dùng chọn **Toàn văn** hoặc **Tóm tắt (LLM)** → **xuất PDF / DOCX / MD**.

Toàn bộ media + tài liệu lưu trong thư viện web.

**Pipeline lõi** ([meetasr/pipeline.py](../../meetasr/pipeline.py)): VAD → ASR → Punc → Speaker → LLM.
- **VAD** = silero-vad (lọc nhạc/lặng — tốt hơn fsmn cho audio Việt)
- **ASR** = zipformer-vi (sherpa-onnx, tiếng Việt)
- **Punc** = vibert-capu (thêm dấu câu + viết hoa cho text thô của zipformer)
- **Speaker** = cam++ (embedding + spectral clustering)
- **LLM** = OpenAI/Ollama (tóm tắt + sinh tài liệu có cấu trúc)

Cấu hình tất cả ở [meeting_config.yaml](../../meeting_config.yaml). File này giải thích rõ vì sao
chọn từng model (đọc comment trong đó — rất hữu ích).

---

## 2. Bố cục source code

```
meetasr/                       # Backend (Python package)
├── pipeline.py                # Lõi: VAD→ASR→Punc→SPK→LLM (MeetPipeline)
├── auto/                      # AutoPipeline / AutoModel — build pipeline từ config
├── models/                    # Wrapper các model: asr/ vad/ punc/ spk/
├── llm/                       # summarizer.py (Phase 1) + planner.py (Phase 2 doc planner)
├── api/
│   ├── app.py                 # FastAPI app, đăng ký router, load pipeline lúc startup
│   ├── dependencies.py        # get_pipeline(), save_upload()
│   └── routes/                # health, transcribe, summarize, db_routes, realtime, live
├── realtime/                  # Phase 2/3: worker.py, live_session.py, speaker.py, events.py...
├── db/                        # SQLModel: realtime_models.py (Source/Segment/Document), connection.py
├── export/                    # Xuất PDF/DOCX/MD
├── storage/                   # Lưu file media (local / MinIO)
└── bin/cli.py                 # CLI: meetasr transcribe | summarize | server

frontend-next/                 # Frontend (Next.js — LƯU Ý: bản có breaking changes, xem AGENTS.md)
└── src/
    ├── app/
    │   ├── page.tsx            # Trang thư viện (danh sách source)
    │   ├── upload/page.tsx     # Trang tải lên
    │   ├── live/page.tsx       # Trang ghi trực tiếp (mic/tab)
    │   └── sources/[id]/page.tsx  # Trang chi tiết: đang xử lý → ProcessingView; xong → DocumentView
    ├── components/             # ProcessingView, DocumentView, FinalizeDialog, Sidebar, ui.tsx...
    ├── hooks/                  # useJobEvents (WS job events), useLiveMic (thu mic → WS /v1/live)
    └── lib/                    # api.ts (client REST/WS), types.ts, format.ts

meet_docs/                     # Toàn bộ tài liệu thiết kế + lộ trình (đọc theo số thứ tự)
```

**Điểm vào đọc code theo thứ tự đề xuất:**
1. [meeting_config.yaml](../../meeting_config.yaml) — hiểu chọn model gì, vì sao.
2. [meetasr/pipeline.py](../../meetasr/pipeline.py) — luồng offline lõi.
3. [meetasr/realtime/worker.py](../../meetasr/realtime/worker.py) — luồng xử lý upload theo thời gian thực (Phase 2).
4. [meetasr/api/routes/realtime.py](../../meetasr/api/routes/realtime.py) — API upload/finalize/export/events.
5. [meetasr/api/routes/live.py](../../meetasr/api/routes/live.py) + [live_session.py](../../meetasr/realtime/live_session.py) — ghi mic trực tiếp (Phase 3).
6. Frontend: [api.ts](../../frontend-next/src/lib/api.ts) → [upload/page.tsx](../../frontend-next/src/app/upload/page.tsx) → [sources/[id]/page.tsx](../../frontend-next/src/app/sources/[id]/page.tsx) → [live/page.tsx](../../frontend-next/src/app/live/page.tsx).

---

## 3. Chạy nhanh (tóm tắt — chi tiết ở 16_run_commands.md)

Cần **3 tiến trình** bật cùng lúc để dùng web:

```powershell
# Terminal 1 — hạ tầng (Postgres 5432 + MinIO 9000/9001)
cd E:\MeetingMindAI; docker compose up -d

# Terminal 2 — backend FastAPI (nạp pipeline theo meeting_config.yaml)
cd E:\MeetingMindAI; meetasr server --config meeting_config.yaml --port 8000 --reload

# Terminal 3 — frontend Next.js
cd E:\MeetingMindAI\frontend-next; pnpm dev
```

Mở **http://localhost:3000**. Swagger API: **http://localhost:8000/docs**.

Lưu ý nhanh (đầy đủ ở [16_run_commands.md](16_run_commands.md)):
- Model ASR/punc/speaker **tự tải** lần chạy đầu; nhưng **silero_vad.onnx** phải có sẵn ở
  `models/silero-vad/silero_vad.onnx` (không tự tải).
- torch trong repo là bản **CPU-only** → VAD/punc/speaker chạy CPU. Muốn GPU: cài lại torch CUDA
  (hướng dẫn cuối [meeting_config.yaml](../../meeting_config.yaml)).
- WebSocket nối **thẳng** backend (`ws://127.0.0.1:8000`), REST đi qua proxy `/v1/*` của Next.
- Không có backend, frontend vẫn mở được nhưng danh sách sẽ báo lỗi "kiểm tra backend cổng 8000".

Chạy **không cần web** (CLI): `meetasr transcribe file.mp4` / `meetasr summarize file.mp4 --config meeting_config.yaml`.

---

## 4. Chức năng web ĐANG CÓ (đối chiếu endpoint + trang thật)

### 4.1. Thư viện — [app/page.tsx](../../frontend-next/src/app/page.tsx)
- Liệt kê mọi Source (`GET /v1/sources`), poll lại mỗi 5s để trạng thái `processing → done` tự cập nhật.
- Xoá source (`DELETE /v1/sources/{id}`, cascade sang segments + documents + file).
- Hai lối vào: **Tải lên media** (`/upload`) và **Ghi trực tiếp** (`/live`).

### 4.2. Tải lên & xử lý realtime — [app/upload/page.tsx](../../frontend-next/src/app/upload/page.tsx) + [worker.py](../../meetasr/realtime/worker.py)
- Upload file (`POST /v1/sources`, form: `file`, `language`, `num_speakers` tuỳ chọn).
- Backend đẩy vào **hàng đợi 1 consumer** (không tranh GPU): trích audio → VAD toàn file →
  chia **cửa sổ ~30s theo ranh giới VAD** (không cắt giữa câu) → phiên âm từng cửa sổ →
  stream `transcript_delta` qua WebSocket `GET /v1/jobs/{id}/events`.
- **Diarization 2 lớp:** khi đang stream, `OnlineSpeakerMatcher` gán nhãn tạm; xử lý xong toàn file
  thì **re-cluster lại toàn bộ** (`_diarize_full_file`) làm nguồn chân lý, rồi gộp các đoạn liền
  cùng người nói và bắn `transcript_rebuilt`.
- Frontend ([ProcessingView.tsx](../../frontend-next/src/components/ProcessingView.tsx)) hiển thị transcript
  chảy dần + tiến độ, dùng hook [useJobEvents.ts](../../frontend-next/src/hooks/useJobEvents.ts).

### 4.3. Ghi trực tiếp (mic / tab) — [app/live/page.tsx](../../frontend-next/src/app/live/page.tsx) + [live.py](../../meetasr/api/routes/live.py)
- Nguồn: **micro** (`getUserMedia`) hoặc **âm thanh tab/màn hình** (`getDisplayMedia`, chạy được cả khi đeo tai nghe).
- [useLiveMic.ts](../../frontend-next/src/hooks/useLiveMic.ts): downsample về 16kHz → PCM16 frame ~250ms →
  gửi binary qua WS `/v1/live`.
- Backend chạy **2 pass** ([live_session.py](../../meetasr/realtime/live_session.py)): pass 1 partial (xám, xem trước),
  pass 2 final (đen, chốt câu + speaker). Khi **lag > 2s** thì tạm bỏ partial để đuổi kịp realtime.
- Bấm Dừng → session **được lưu thành Source** (re-cluster speaker toàn bản ghi) → tự chuyển sang trang tài liệu.
- Ô "**Số người nói**" (oracle) giúp gán nhãn chính xác hơn; để trống = tự đoán.
- Giới hạn phiên đồng thời: `MEETASR_MAX_LIVE` (mặc định 2).

### 4.4. Tài liệu: finalize + xuất — [sources/[id]/page.tsx](../../frontend-next/src/app/sources/[id]/page.tsx) + [realtime.py](../../meetasr/api/routes/realtime.py)
- Đang xử lý → [ProcessingView](../../frontend-next/src/components/ProcessingView.tsx); xong →
  [DocumentView](../../frontend-next/src/components/DocumentView.tsx).
- **Finalize** (`POST /v1/documents/{id}/finalize`, mode `summary` | `full_text`):
  - `full_text`: ghép transcript.
  - `summary`: gọi **DocumentPlanner** (LLM) sinh tài liệu có cấu trúc (content-agnostic — xem
    [14_document_planner_design.md](14_document_planner_design.md)).
- **Xuất** (`GET /v1/documents/{id}/export/{docId}?format=pdf|docx|md`).

### 4.5. API khác (có sẵn trên Swagger)
| Nhóm | Endpoint | File |
|---|---|---|
| System | `GET /v1/health` | [health.py](../../meetasr/api/routes/health.py) |
| ASR | `POST /v1/audio/transcriptions` | [transcribe.py](../../meetasr/api/routes/transcribe.py) |
| Meeting (Phase 1) | `POST /v1/meeting/summarize`, `.../status`, `.../summarize-text` | [summarize.py](../../meetasr/api/routes/summarize.py) |
| Database (Phase 1) | `GET /v1/meetings`, `.../report`, `DELETE .../{id}` | [db_routes.py](../../meetasr/api/routes/db_routes.py) |
| Realtime (Phase 2) | `/v1/sources`, `/v1/documents/...`, `/v1/metrics`, WS `/v1/jobs/{id}/events` | [realtime.py](../../meetasr/api/routes/realtime.py) |
| Live (Phase 3) | WS `/v1/live` | [live.py](../../meetasr/api/routes/live.py) |
| Metrics | `GET /v1/metrics` (số job, RTF trung bình, độ dài hàng đợi) | [realtime.py](../../meetasr/api/routes/realtime.py) |

---

## 5. CHƯA triển khai / còn dở — team cần biết

> Đây là phần quan trọng nhất cho người mới: đừng tưởng có sẵn.

### 5.1. Diarization (tách người nói) — **hạn chế đã biết, đang mở**
- **Giọng nam + nữ vẫn bị gộp** thành 1 người trong hội thoại nhanh (qua lại liên tục). Nguyên nhân gốc:
  chunk 1.5s trộn 2 giọng + clustering gộp 2 giọng gần nhau. **Không sửa được bằng vá ở tầng hiện tại.**
- Đã có [diagnose_diar.py](../../diagnose_diar.py) (ở gốc repo) để chẩn đoán: dump nhãn cluster theo chunk,
  khoảng cách cosine giữa các centroid, timeline speaker, và vote speaker theo câu.
  Chạy: `conda run -n meet python diagnose_diar.py <audio.wav> [num_speakers]`.
- **Hướng chưa quyết:** giảm kích thước chunk / cho sửa nhãn thủ công trên UI / đổi model mạnh hơn (pyannote 3.1).
- **Cách né hiện tại:** người dùng nhập **số người nói** (oracle) ở form upload và trang live — cải thiện đáng kể.

### 5.2. Ghi trực tiếp (Phase 3) — **chỉ mic/tab trình duyệt**
- **CHƯA có system-audio cấp OS** (thu toàn bộ tiếng máy, kể cả app khác) — cần app desktop Tauri, là **Phase 4**
  (xem [17_phase4_realtime_desktop.md](17_phase4_realtime_desktop.md)). Hiện chỉ có mic + audio tab qua trình duyệt.
- Live session **KHÔNG lưu audio thô** (chỉ giữ buffer RAM để re-cluster lúc kết thúc). Nghĩa là:
  clip đã ghi trực tiếp **không tái phân tích lại được** — muốn chẩn đoán phải ghi ra WAV riêng.
- **KHÔNG sinh notes realtime** khi đang ghi; tóm tắt chỉ sinh sau khi Dừng (đi qua finalize Phase 2).

### 5.3. Tài liệu realtime bằng LLM khi đang xử lý upload — **tắt mặc định (opt-in)**
- `doc_delta` chảy nội dung LLM theo thời gian thực chỉ bật khi `MEETASR_REALTIME_DOC=1` **và** có DocumentPlanner.
  Mặc định tắt → khi xử lý chỉ ghép text thô theo cửa sổ; tóm tắt LLM sinh ở bước finalize.

### 5.4. LLM tóm tắt — **cần API key thật**
- [meeting_config.yaml](../../meeting_config.yaml) để `api_key: "fake"` chỉ để server khởi động được.
  Muốn "Tóm tắt" chạy: đặt key OpenAI thật, hoặc chuyển `provider: ollama` chạy local.
  Nếu key giả → finalize `summary` sẽ lỗi (`no_planner`/`llm_error`); `full_text` không cần LLM.

### 5.5. GPU
- torch = **CPU-only** trong repo. VAD/punc/speaker/LLM-embedding chạy CPU (chậm với file dài).
  Muốn GPU phải cài lại torch CUDA (hướng dẫn cuối [meeting_config.yaml](../../meeting_config.yaml)) rồi đổi `device: cuda:0`.

### 5.6. Bảo mật / production
- CORS đang `allow_origins=["*"]` ([app.py](../../meetasr/api/app.py) — `TODO` siết lại khi deploy).
- **WebSocket `/v1/live` và `/v1/jobs/{id}/events` không có xác thực** — bất kỳ ai vào được cổng đều nối được.
  Chưa có auth/authz ở bất kỳ endpoint nào. Cần bổ sung trước khi mở ra ngoài localhost.
- Upload giới hạn 500MB ([dependencies.py](../../meetasr/api/dependencies.py)); định dạng: wav/mp3/m4a/mp4/flac/ogg/webm.

### 5.7. Frontend Next.js — **bản có breaking changes**
- Đọc [frontend-next/AGENTS.md](../../frontend-next/AGENTS.md): "This is NOT the Next.js you know". Trước khi viết code
  frontend, đọc guide trong `node_modules/next/dist/docs/`. API/quy ước có thể khác training data.
- [lib/mock.ts](../../frontend-next/src/lib/mock.ts) còn tồn tại (dữ liệu mẫu) — kiểm tra trước khi tưởng là dữ liệu thật.

---

## 6. Trạng thái theo Phase (nhìn nhanh)

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 1 | Pipeline offline: upload → transcript + summary; API `/v1/meeting/*` | ✅ Xong |
| 2 | "NotebookLM": upload → doc realtime → finalize summary/full_text → export; frontend Next.js | ✅ Cốt lõi xong; realtime-doc LLM opt-in |
| 3 | Port 4 điểm mạnh FunASR + **ghi mic 2-pass tiếng Việt** (`/v1/live`) | ✅ Chạy được; diarization còn hạn chế (5.1), lag đã xử lý |
| 4 | App desktop Tauri + **system audio** cấp OS | ⏳ Kế hoạch — [17_phase4_realtime_desktop.md](17_phase4_realtime_desktop.md) |

Chi tiết hạn chế Phase 2: [13_phase2_limitations_and_fixes.md](13_phase2_limitations_and_fixes.md).
Chi tiết chuẩn hoá audio mic (vì sao Silero, xử lý nhạc): [18_chuan_hoa_audio_mic.md](18_chuan_hoa_audio_mic.md).
