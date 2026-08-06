# [ĐÃ THAY THẾ] Team Task Breakdown — Phase 2 Realtime (MeetASR)

> ⚠️ **Doc này đã được thay thế.** Phân công Phase 2 mới nằm trong
> [11_phase2_notebooklm_plan.md](11_phase2_notebooklm_plan.md) (mục 5).

> Phân công công việc Phase 2 (Realtime) cho nhóm 4 người: 3 AI Engineer + 1 Data Engineer.
> Đọc [09_realtime_phase2.md](09_realtime_phase2.md) trước để hiểu kiến trúc tổng thể và các milestone M0–M7.
> Kiến thức nền cần học cho từng vai trò: xem [skills/06_realtime_streaming_skill.md](../skills/06_realtime_streaming_skill.md).

---

## 1. Sơ đồ phụ thuộc

```
[AI Engineer 3: WebSocket + Session backend (M0, M3)] ◄─── nền tảng, làm TRƯỚC
        ▲                    ▲                    ▲
        │                    │                    │
[AI Eng 1: Streaming   [AI Eng 2: Online     [Data Eng: Frontend live
 ASR + VAD (M1, M2)]    Diarizer (M4)]        + Persistence (M0, M5, M6)]
        │                    │                    │
        └────────────────────┴────────────────────┘
                             ▼
              [AI Engineer 3: Tích hợp end-to-end (M5, M7)]
```

- **AI Engineer 1** và **AI Engineer 2** làm việc được **độc lập ngay từ đầu** (PoC chạy trên file
  phát lại như stream giả lập — không cần WebSocket).
- **AI Engineer 3** dựng khung WebSocket + session trước (M0, M3), sau đó lắp module của AI Eng 1/2 vào.
- **Data Engineer** làm phía client (mic capture) song song với M0 của AI Eng 3, và persistence sau khi
  có segment final đầu tiên chạy được.

---

## 2. Phân công theo vai trò

### 2.1 AI Engineer 1 — Streaming ASR + Streaming VAD (M1, M2)

**Vai trò:** Xây lớp streaming quanh faster-whisper và VAD online. Đây là "trái tim" của realtime.

#### File phụ trách (tạo mới):
- `meetasr/streaming/streaming_asr.py` — LocalAgreement wrapper quanh faster-whisper
- `meetasr/streaming/streaming_vad.py` — silero-vad online + endpointing
- `tests/test_streaming_asr.py`, `tests/test_streaming_vad.py`

#### Nhiệm vụ cụ thể:
1. **PoC LocalAgreement-2 (M1):**
   - Viết script phát lại 1 file wav như luồng stream (mỗi lần đẩy 0.5s audio).
   - Buffer audio; mỗi lần nhận chunk mới → chạy lại faster-whisper trên phần buffer chưa "chốt".
   - So khớp token giữa 2 lần chạy liên tiếp: prefix trùng nhau → phát **final**; phần đuôi → **partial**.
   - Nghiên cứu kỹ code của [whisper_streaming](https://github.com/ufal/whisper_streaming)
     (file `whisper_online.py`) — thuật toán LocalAgreement đã có sẵn ở đó, hiểu và port lại tối giản.
2. **Streaming VAD + endpointing (M2):**
   - Tích hợp [silero-vad](https://github.com/snakers4/silero-vad) chế độ streaming (chunk 30ms).
   - Sinh sự kiện `speech_start` / `speech_end`; dùng `speech_end` (im lặng > ~0.8s) để commit cứng
     buffer ASR và cắt segment (segment này sẽ chuyển cho AI Engineer 2 làm diarization).
3. **Giao diện chuẩn với các module khác** (thống nhất với AI Engineer 3 trước khi code):
   ```python
   class StreamingASR:
       def feed(self, pcm: np.ndarray) -> list[AsrEvent]: ...
       # AsrEvent: {"type": "partial"|"final", "text": str, "start_ms": int, "end_ms": int}
   ```

#### Verify:
- Chạy PoC trên `tests/data/test1.wav`: RTF < 1 trên GPU; text final khớp ≥ 90% so với bản offline;
  partial không "nhảy" nội dung đã final.
- Khoảng im lặng không sinh lệnh gọi ASR (đo bằng log).

---

### 2.2 AI Engineer 2 — Online Speaker Diarization + Final Pass (M4, M6-LLM)

**Vai trò:** Nhận diện "ai đang nói" theo thời gian thực, tái dùng CAM++ và clustering sẵn có.

#### File phụ trách:
- `meetasr/streaming/online_diarizer.py` (mới)
- Tái dùng: `meetasr/models/spk/campplus.py`, `meetasr/models/spk/cluster.py`
- `tests/test_online_diarizer.py`

#### Nhiệm vụ cụ thể:
1. **Online centroid clustering (M4):**
   - Mỗi segment do VAD chốt → trích embedding CAM++ [192-d] (dùng `spk.embed` sẵn có).
   - So cosine similarity với danh sách centroid speaker hiện có:
     `max_sim > threshold` (khởi điểm 0.7, tinh chỉnh sau) → gán vào speaker đó, cập nhật centroid
     bằng EMA (`centroid = 0.9*centroid + 0.1*emb`); ngược lại → tạo speaker mới.
   - Segment quá ngắn (< 0.5s) → gán theo speaker của segment liền trước, không tạo mới.
2. **Final re-cluster (M6):**
   - Lưu lại toàn bộ embedding trong phiên; khi kết thúc → chạy `CommonClustering` (offline, sẵn có)
     trên toàn bộ → sinh mapping nhãn "sạch" → cập nhật lại transcript trong DB.
3. **(Nếu còn thời gian) Running summary:** mỗi N segment final, đẩy phần transcript mới vào LLM
   (tái dùng `summarizer.py`) để cập nhật tóm tắt cuộn.

#### Giao diện chuẩn:
```python
class OnlineDiarizer:
    def assign(self, pcm_segment: np.ndarray, start_ms: int, end_ms: int) -> int:  # speaker id
    def finalize(self) -> dict[int, int]:  # mapping: online id → clean id sau re-cluster
```

#### Verify:
- Audio 2 người nói (dùng data test diarization sẵn có): nhãn online đúng ≥ ~80%,
  không tạo quá `số người thật + 1` speaker.
- Sau `finalize()`, nhãn khớp với kết quả pipeline offline Phase 1.

---

### 2.3 AI Engineer 3 — WebSocket Backend + Session + Tích hợp (M0, M3, M5, M7)

**Vai trò:** Người tích hợp hệ thống (như Phase 1). Dựng hạ tầng realtime TRƯỚC để 2 bạn kia có chỗ lắp module.

#### File phụ trách (tạo mới):
- `meetasr/api/routes/realtime.py` — WebSocket `/v1/realtime/stream`
- `meetasr/streaming/session.py` — StreamSession: state theo từng kết nối
- `meetasr/streaming/worker.py` — inference worker + queue
- Cập nhật `meetasr/api/app.py` (đăng ký router)

#### Nhiệm vụ cụ thể:
1. **Spike WebSocket (M0):** endpoint WS nhận binary PCM16 16kHz, echo lại metadata
   (số byte, duration) — để Data Engineer test phía mic. Chốt **protocol** (xem mục 3).
2. **Session + Worker (M3):**
   - `StreamSession` giữ: ring buffer audio, state VAD/ASR/diarizer, transcript tạm.
   - Inference **không được block event loop**: đẩy qua `asyncio.Queue` + `run_in_executor`.
   - Queue có giới hạn (backpressure): nếu xử lý chậm hơn audio đến → gộp chunk, log cảnh báo.
   - Model GPU load 1 lần lúc startup (thêm vào `lifespan` của app), dùng chung mọi session, có lock.
   - Cleanup toàn bộ state khi client disconnect/timeout.
3. **Tích hợp (M5):** lắp `StreamingASR` (AI Eng 1) + `OnlineDiarizer` (AI Eng 2) vào session;
   đẩy sự kiện partial/final/speaker về client theo protocol.
4. **Polish (M7):** xử lý reconnect, đo metric (first-token latency, commit latency), load test 2–3 client.

#### Verify:
- M0: client giả lập (script Python + `websockets`) gửi file wav → server log đúng duration.
- M3: 3 client đồng thời, event loop không block (endpoint `/health` vẫn trả lời < 100ms).
- M7: chạy phiên 30 phút không rò RAM/VRAM.

---

### 2.4 Data Engineer — Frontend Live Page + Persistence (M0-client, M5, M6)

**Vai trò:** Phía trình duyệt (bắt mic, gửi audio, render transcript) và lưu trữ phiên realtime.

#### File phụ trách:
- `frontend/4_live.html` (mới) — trang họp trực tiếp
- `meetasr/db/models.py`, `meetasr/db/repository.py` (mở rộng — đã làm ở Phase 1)
- `tests/test_live_session_db.py`

#### Nhiệm vụ cụ thể:
1. **Mic capture (song song với M0):**
   - `getUserMedia` → `AudioWorklet` lấy PCM float32 → downsample về **16kHz mono** →
     convert PCM16 → gửi WebSocket binary theo chunk ~250ms.
   - Xử lý lỗi: từ chối quyền mic, mất kết nối (auto-reconnect + thông báo).
2. **Render transcript live (M5):**
   - Vùng **final**: text đen, kèm màu/badge theo speaker, timestamp.
   - Vùng **partial**: text xám ở cuối, bị ghi đè liên tục theo sự kiện mới.
   - Đồng hồ phiên + trạng thái kết nối. Style thống nhất với `1_dashboard.html`.
3. **Persistence (M6):**
   - Bảng mới: `live_sessions` (id, started_at, ended_at, status),
     `transcript_segments` (session_id, start_ms, end_ms, speaker, text).
   - Ghi dần segment final trong phiên; khi kết thúc → nhận mapping từ `finalize()` của diarizer,
     cập nhật nhãn speaker, rồi lưu `MeetingReport` (tái dùng schema Phase 1).

#### Verify:
- Loopback test: thu 10s giọng nói → server ghi ra wav → nghe lại không méo, đúng 16kHz.
- Reload trang sau phiên → transcript đầy đủ hiện từ DB.

---

## 3. WebSocket Protocol (chốt ở M0 — mọi người code theo chuẩn này)

**Client → Server:**
- Binary frame: PCM16 little-endian, mono, 16kHz (chunk ~250ms = 8000 bytes)
- Text frame (JSON): `{"type": "start", "language": "vi"}` · `{"type": "stop"}`

**Server → Client (JSON):**
```json
{"type": "partial",  "text": "xin chào các b",  "start_ms": 1200}
{"type": "final",    "text": "Xin chào các bạn.", "start_ms": 1200, "end_ms": 2800, "speaker": 0}
{"type": "speaker_update", "mapping": {"2": 0}}          // re-cluster đổi nhãn
{"type": "session_end", "session_id": 42, "report_url": "/v1/meetings/42"}
{"type": "error", "code": "...", "message": "..."}
```

---

## 4. Lịch trình gợi ý (5 tuần)

| Tuần | AI Eng 1 | AI Eng 2 | AI Eng 3 | Data Eng |
|---|---|---|---|---|
| 1 | Đọc whisper_streaming, PoC LocalAgreement (M1) | Đọc diart/pyannote, thiết kế OnlineDiarizer | Spike WS + chốt protocol (M0) | Mic capture + loopback test |
| 2 | Hoàn thiện M1, bắt đầu VAD (M2) | Implement centroid clustering (M4) | Session + worker + queue (M3) | Render live page cơ bản |
| 3 | Hoàn thiện M2, tinh chỉnh latency | Test với audio nhiều người, tinh chỉnh threshold | Tích hợp ASR vào session | Bảng DB mới + ghi segment |
| 4 | Hỗ trợ tích hợp, đo WER | Final re-cluster (M6) | Tích hợp diarizer, demo E2E (M5) | Persistence hoàn chỉnh (M6) |
| 5 | — Buffer / polish chung (M7): reconnect, load test, đo metric, sửa bug tích hợp — | | | |

---

## 5. Branching & quy ước

- Nhánh theo người: `khanh/streaming-asr`, `xxx/online-diarizer`, `xxx/realtime-ws`, `xxx/live-frontend`
  — đều rebase từ `develop`.
- **Không sửa pipeline offline** (`meetasr/pipeline.py` và các model wrapper Phase 1) trừ khi bắt buộc —
  code streaming nằm trọn trong `meetasr/streaming/` + 1 route mới.
- Chốt interface (mục 2.1/2.2/2.3 + protocol mục 3) **trước khi code** — họp 30 phút đầu tuần 1.
- Tuân thủ [coding_style.md](../constraints/coding_style.md): type hints, docstrings, unit tests.
