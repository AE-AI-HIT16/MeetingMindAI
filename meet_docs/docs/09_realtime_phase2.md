# [ĐÃ THAY THẾ] Phase 2 — Realtime Meeting (Live Transcript + Online Diarization)

> ⚠️ **Doc này đã được thay thế.** Định hướng Phase 2 mới (NotebookLM-style: upload video/audio
> → docs realtime → PDF) xem tại [11_phase2_notebooklm_plan.md](11_phase2_notebooklm_plan.md)
> và [12_frontend_nextjs_plan.md](12_frontend_nextjs_plan.md). Giữ lại doc này để tham khảo
> hướng live-mic streaming trong tương lai.

> Mục tiêu Phase 2: biến MeetASR từ hệ thống **offline/batch** (upload file → chờ → nhận JSON)
> thành hệ thống **realtime**: người nói → chữ hiện ngay trên màn hình (~1–3s trễ), kèm nhãn
> speaker cập nhật trực tiếp. Chạy trên **GPU**, giao tiếp qua **WebSocket**.

---

## 0. Hiện trạng (Phase 1) và khoảng cách cần lấp

| Thành phần | Phase 1 (hiện tại) | Phase 2 (cần) |
|---|---|---|
| Luồng dữ liệu | Load toàn bộ audio 1 lần | Nhận audio theo chunk liên tục |
| VAD | Chạy trên cả file | VAD online (streaming, có endpointing) |
| ASR | SenseVoice/Whisper batch | Streaming ASR (chunk + LocalAgreement) |
| Diarization | Offline clustering (cần cả file) | Online clustering (centroid tăng dần) |
| API | HTTP POST đồng bộ, trả 1 lần | WebSocket, đẩy partial/final liên tục |
| Frontend | HTML tĩnh, upload form | Live page: mic → WS → render realtime |
| Inference | Gọi trực tiếp trong request | Worker bất đồng bộ + queue (không block event loop) |

**Kết luận:** không thể "sửa nhẹ" pipeline hiện tại. Cần một *đường xử lý streaming song song*
với pipeline offline (giữ nguyên offline để dùng cho file upload). Điểm mạnh: 2 model chính đã có sẵn
và chạy GPU tốt — `faster-whisper` (ASR đa ngôn ngữ + word timestamps) và `CAM++` + `CommonClustering`.

---

## 1. Kiến trúc tổng thể

```
┌─────────────┐   PCM16 16kHz     ┌──────────────────────────────────────────┐
│  Browser    │ ───(WS binary)──► │  FastAPI WebSocket  /v1/realtime/stream    │
│  mic +      │                   │  ┌──────────────────────────────────────┐ │
│  AudioWorklet│ ◄──(WS json)───── │  │ Session state (per connection)       │ │
└─────────────┘  partial/final/    │  │  • ring buffer audio                 │ │
                 speaker/summary    │  │  • streaming VAD state               │ │
                                    │  │  • ASR LocalAgreement buffer         │ │
                                    │  │  • speaker centroids (online)        │ │
                                    │  └──────────────────────────────────────┘ │
                                    │            │ enqueue audio                  │
                                    │            ▼                               │
                                    │  ┌──────────────────────────────────────┐ │
                                    │  │ Inference worker (asyncio + executor) │ │
                                    │  │  VAD → ASR(stream) → SPK(online)      │ │
                                    │  │  GPU model dùng chung, chạy nối tiếp   │ │
                                    │  └──────────────────────────────────────┘ │
                                    └──────────────────────────────────────────┘
                                                 │ (khi kết thúc phiên)
                                                 ▼
                                    Final offline re-cluster + LLM summary → DB
```

**Nguyên tắc thiết kế:**
- **Không phá offline.** Streaming là module mới (`meetasr/streaming/`), pipeline offline giữ nguyên.
- **1 WebSocket = 1 phiên họp = 1 session state** trong RAM. Dọn sạch khi disconnect.
- **Model GPU load 1 lần, dùng chung** mọi session (VRAM có hạn — không load theo từng session).
  Inference đẩy qua queue chạy nối tiếp (hoặc micro-batch) để không tranh chấp GPU.
- **ML inference chạy trong executor/worker**, tuyệt đối không block asyncio event loop.

---

## 2. Các khối kỹ thuật

### 2.1 Streaming ASR (khó vừa — nền tảng đã có)
- **Vấn đề:** faster-whisper là model *offline* (dịch trọn 1 đoạn). Không tự "stream".
- **Giải pháp: LocalAgreement-2** (kỹ thuật của `whisper_streaming`/WhisperLiveKit):
  1. Gom audio vào buffer, cứ mỗi ~0.5–1s lại chạy lại ASR trên phần buffer chưa "chốt".
  2. So khớp kết quả 2 lần chạy liên tiếp; các token **trùng khớp ở đầu** → coi là ổn định → phát **final**.
  3. Phần đuôi chưa ổn định → phát **partial** (chữ xám, có thể thay đổi).
  4. Khi VAD báo hết câu (endpoint) hoặc buffer quá dài → "commit" cứng, cắt buffer.
- **Vì sao khả thi:** faster-whisper đã hỗ trợ tiếng Việt + word timestamps + chạy `float16` trên GPU.
  Với `large-v3` trên GPU RTF thường < 0.3 → thừa sức realtime; nếu trễ, hạ xuống `medium`/`small`.
- **Tham khảo:** `whisper_streaming` (LocalAgreement gốc), WhisperLiveKit, SimulStreaming.

### 2.2 Streaming VAD + endpointing (dễ)
- Cần VAD online để: (a) không chạy ASR trên khoảng lặng, (b) xác định thời điểm "hết câu" để commit.
- Lựa chọn: **silero-vad** (streaming, nhẹ, phổ biến) hoặc bản streaming của FSMN-VAD.
- Output: sự kiện `speech_start` / `speech_end` theo thời gian → dùng để cắt segment cho diarization.

### 2.3 Online Speaker Diarization (khó nhất — tái dùng CAM++)
- **Vấn đề:** clustering offline hiện tại (`SpectralCluster`/`AHC`) cần *toàn bộ* embedding → không realtime.
- **Giải pháp: online clustering theo centroid** (dựng mới, tái dùng `campplus.py`):
  1. Mỗi segment vừa được VAD chốt → trích embedding CAM++ [192-d] (đã có `spk.embed`).
  2. Tính cosine similarity với danh sách **centroid** các speaker đã biết.
  3. Nếu `max_sim > threshold` (vd 0.7) → gán vào speaker đó + cập nhật centroid bằng EMA;
     ngược lại → tạo speaker mới.
  4. Gán nhãn speaker vào transcript ngay khi phát final.
- **Chống trôi (drift):** định kỳ (vd mỗi 30s) hoặc khi kết thúc phiên, chạy lại **offline
  re-cluster** bằng `CommonClustering` sẵn có trên toàn bộ embedding → "dọn" nhãn cho bản ghi cuối.
- **Tradeoff:** DER online cao hơn offline. Chấp nhận trong lúc live, sửa ở pass cuối.
- **Tham khảo:** `diart` (online diarization), pyannote streaming — dùng làm tham chiếu ý tưởng, không cần phụ thuộc.

### 2.4 (Tùy chọn) Running LLM summary
- Không nằm trong scope bắt buộc bạn chọn, nhưng dễ ghép: mỗi N segment final hoặc mỗi T phút,
  đẩy phần transcript mới vào LLM để cập nhật "tóm tắt cuộn" + action items tạm thời (stream token về client).

### 2.5 Backend concurrency (khó vừa — quyết định độ ổn định)
- FastAPI WebSocket là `async`; inference là blocking/GPU-bound → **phải** đẩy sang
  `run_in_executor` (ThreadPoolExecutor) hoặc process worker riêng.
- **Backpressure:** dùng queue có giới hạn; nếu xử lý chậm hơn audio đến → gộp/bỏ bớt chunk cũ.
- **GPU sharing:** 1 instance model, 1 lock/queue để inference chạy nối tiếp giữa các session; nâng cao thì micro-batch.
- **Session lifecycle:** khởi tạo state khi connect, cleanup (giải phóng buffer) khi disconnect/timeout.

### 2.6 Persistence
- Bảng mới: `live_session` (id, started_at, status), `transcript_segment` (session_id, start, end, speaker, text, is_final).
- Ghi dần các segment *final*. Kết thúc phiên → chạy re-cluster + LLM → lưu `MeetingReport` (tái dùng schema Phase 1).

### 2.7 Frontend — Live page
- `getUserMedia` → `AudioWorklet` để lấy PCM float32 → **downsample về 16kHz** → chuyển PCM16 → gửi WS binary (chunk ~200–500ms).
- Render: transcript với 2 vùng — **final** (đen) + **partial** (xám, ghi đè liên tục); màu theo speaker; đồng hồ phiên; trạng thái kết nối.
- Xử lý reconnect + thông báo lỗi mic/permission.

---

## 3. Kế hoạch triển khai theo milestone (goal-driven)

> Mỗi milestone có tiêu chí **verify** rõ ràng để loop độc lập (theo claude.md §4).

| # | Milestone | Nội dung | Verify |
|---|---|---|---|
| **M0** | Spike WS + mic | Endpoint WS echo; browser bắt mic, downsample 16kHz, gửi PCM; server nhận đúng | Log server nhận đúng samplerate/độ dài; nghe lại audio loopback không méo |
| **M1** | Streaming ASR PoC | Phát lại 1 file như luồng stream → LocalAgreement trên faster-whisper → in partial/final ra console | RTF < 1 trên GPU; tiếng Việt ra chữ; final không "nhảy" lung tung |
| **M2** | VAD online + endpoint | Ghép silero-vad; sinh sự kiện speech_start/end; commit buffer đúng chỗ | Khoảng lặng không sinh ASR; câu kết thúc → commit final đúng |
| **M3** | Backend session + worker | Route WS thật, session state, inference qua executor + queue có backpressure | 2–3 client đồng thời không block; đo latency ổn định |
| **M4** | Online diarization | CAM++ + online centroid clustering; gán speaker lúc phát final | Audio 2 người: nhãn speaker đúng ≥ ~80%; không tạo thừa speaker |
| **M5** | Frontend live page | Trang live hoàn chỉnh: mic → WS → render partial/final + màu speaker | Demo end-to-end nói → thấy chữ ~1–3s; nhãn speaker hiển thị |
| **M6** | Persistence + final pass | Lưu segment dần; kết thúc → offline re-cluster + LLM summary → DB | Reload thấy transcript đầy đủ + summary; nhãn speaker "sạch" hơn lúc live |
| **M7** | Polish | Reconnect, backpressure tuning, metrics (first-token/commit latency), load test | Chạy 30 phút không rò RAM/VRAM; báo cáo latency & DER/WER |

**Đường "demo tối thiểu"** nếu gấp: M0→M1→M3→M5 (transcript realtime, chưa có speaker) rồi thêm M4.

---

## 4. Rủi ro & cách giảm thiểu

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| LocalAgreement chỉnh chưa khéo → chữ nhấp nháy/nhảy | Cao | Dành riêng M1 để tinh chỉnh; bắt đầu từ tham số của whisper_streaming |
| Online diarization DER kém khi nhiều người/chồng tiếng | Cao | Chấp nhận lúc live + offline re-cluster ở M6; đặt threshold thận trọng |
| GPU nghẽn khi nhiều session | Trung bình | Queue nối tiếp + giới hạn số session; micro-batch nếu cần |
| Block event loop do inference | Trung bình | Bắt buộc `run_in_executor`/worker ngay từ M3 |
| Audio browser sai samplerate/format | Trung bình | Chốt PCM16 mono 16kHz ở M0, test loopback trước |

---

## 5. Kiến thức & kỹ năng cần trang bị

### 5.1 Streaming ASR
- Khái niệm: chunk/window, **LocalAgreement**, endpointing, RTF (real-time factor), latency vs accuracy.
- Đọc code: `whisper_streaming`, WhisperLiveKit; hiểu `faster-whisper` (CTranslate2, beam_size, word_timestamps).

### 5.2 Async Python & WebSocket (quan trọng nhất về kỹ thuật hệ thống)
- `asyncio`: event loop, `await`, `create_task`, `Queue`, `run_in_executor`.
- FastAPI WebSocket: `websocket.accept/receive_bytes/send_json`, vòng đời kết nối.
- Backpressure, bounded queue, xử lý disconnect/timeout, quản lý state theo session.

### 5.3 Audio trong trình duyệt
- `getUserMedia`, Web Audio API, **AudioWorklet**; resampling về 16kHz; encode PCM16; gửi WebSocket binary.

### 5.4 Speaker embedding & online clustering
- Cosine similarity, centroid, cập nhật EMA, ngưỡng tạo speaker mới.
- Ý tưởng online diarization: `diart`, pyannote-audio (tham chiếu).

### 5.5 Phục vụ mô hình trên GPU
- Chia sẻ model giữa request, micro-batching, VRAM, `float16`/`int8`, đo throughput.

### 5.6 Observability & kiểm thử realtime
- Đo first-token latency, commit latency, RTF; load test WebSocket (locust/websocket client giả lập).
- Metric chất lượng: WER (so bản offline), DER cho diarization.

### 5.7 (Nếu làm running summary) LLM streaming
- Token streaming, incremental prompting, cắt/gộp context khi transcript dài.

### Thư viện nên khảo sát
`faster-whisper` (đã có) · `whisper_streaming`/WhisperLiveKit · `silero-vad` · `diart`/`pyannote-audio` ·
`torchaudio` · FastAPI WebSocket · AudioWorklet (browser).

---

## 6. Cấu trúc thư mục đề xuất (mới, không đụng offline)

```
meetasr/streaming/
├── __init__.py
├── session.py          ← StreamSession: state per WebSocket (buffer, vad, asr, spk)
├── streaming_asr.py    ← LocalAgreement wrapper quanh faster-whisper
├── streaming_vad.py    ← silero/fsmn online VAD + endpointing
├── online_diarizer.py  ← CAM++ + online centroid clustering (tái dùng spk model)
└── worker.py           ← inference worker + queue (run_in_executor / GPU lock)

meetasr/api/routes/realtime.py   ← WebSocket /v1/realtime/stream
frontend/4_live.html             ← trang họp trực tiếp (mic → WS → render)
```
