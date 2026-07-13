# Skill 06 — Realtime Streaming (Phase 2)

> Tài liệu học tập cho Phase 2 Realtime. Mỗi thành viên đọc phần **chung** + phần của **vai trò mình**
> (phân công trong [10_realtime_team_tasks.md](../docs/10_realtime_team_tasks.md)).
> Mục tiêu: hiểu đủ để bắt đầu code milestone đầu tiên trong tuần 1.

---

## A. Kiến thức chung (CẢ NHÓM phải nắm)

### A.1 Realtime ASR khác offline ASR thế nào?

| | Offline (Phase 1) | Realtime (Phase 2) |
|---|---|---|
| Input | Cả file audio | Chunk audio liên tục (~250ms/lần) |
| Output | 1 kết quả cuối | Chuỗi sự kiện `partial` (tạm) + `final` (chốt) |
| Metric chính | WER (độ chính xác) | WER **+ latency** (độ trễ) — luôn phải đánh đổi |
| Trạng thái | Stateless | Stateful — mỗi phiên có buffer, state riêng |

Khái niệm cần biết:
- **RTF (Real-Time Factor)** = thời_gian_xử_lý / thời_lượng_audio. RTF < 1 mới realtime được.
  GPU + faster-whisper large-v3 thường RTF ≈ 0.1–0.3.
- **Partial vs Final**: partial là kết quả tạm (có thể đổi), final là đã chốt (không đổi nữa).
  UI hiện partial màu xám, final màu đen — giống caption của Google Meet.
- **Endpointing**: phát hiện "người nói đã dứt câu" (thường = im lặng > 0.5–1s) để chốt kết quả.

### A.2 Đọc gì trước (theo thứ tự)
1. [09_realtime_phase2.md](../docs/09_realtime_phase2.md) — kiến trúc Phase 2 của chính dự án.
2. README của [whisper_streaming](https://github.com/ufal/whisper_streaming) — hiểu ý tưởng LocalAgreement (10 phút).
3. FastAPI WebSocket docs: https://fastapi.tiangolo.com/advanced/websockets/ (15 phút).

---

## B. Theo vai trò

### B.1 AI Engineer 1 — Streaming ASR + VAD

**Kiến thức cần:**
- **Thuật toán LocalAgreement-2**: chạy lại ASR trên buffer mỗi khi có audio mới; phần prefix
  giống nhau giữa 2 lần chạy liên tiếp = ổn định → phát final. Đây là cách phổ biến nhất để
  "streaming hóa" một model offline như Whisper.
- **faster-whisper API**: `WhisperModel.transcribe()`, `word_timestamps=True`, `beam_size`,
  `compute_type="float16"`. Đã có wrapper trong `meetasr/models/asr/faster_whisper_asr.py` — đọc trước.
- **silero-vad streaming**: model nhận chunk 30ms, trả xác suất speech; có sẵn utils
  `VADIterator` cho chế độ streaming.

**Tài nguyên:**
- Code chính cần đọc: [whisper_streaming/whisper_online.py](https://github.com/ufal/whisper_streaming/blob/main/whisper_online.py)
  — class `OnlineASRProcessor` + `HypothesisBuffer` (~300 dòng, đọc kỹ).
- Paper (đọc lướt phần 3): "Turning Whisper into Real-Time Transcription System" (Macháček et al., 2023).
- [silero-vad examples](https://github.com/snakers4/silero-vad/tree/master/examples) — file streaming example.
- Tham khảo thêm: [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) — bản production-ready của cùng ý tưởng.

**Bài tập khởi động (ngày 1–2):**
Viết script: đọc `tests/data/test1.wav`, cắt thành chunk 0.5s, feed lần lượt vào faster-whisper
với buffer tăng dần, in ra transcript của mỗi lần chạy. Quan sát: phần đầu output có ổn định
dần không? → đó chính là trực giác của LocalAgreement.

---

### B.2 AI Engineer 2 — Online Diarization

**Kiến thức cần:**
- **Speaker embedding**: vector đặc trưng giọng nói (CAM++ → 192-d); 2 đoạn cùng người →
  cosine similarity cao (~0.7+), khác người → thấp. Đã có `meetasr/models/spk/campplus.py`.
- **Online vs offline clustering**: offline (Phase 1) thấy hết dữ liệu rồi mới chia cụm;
  online phải quyết định NGAY khi mỗi segment đến → dùng **centroid + threshold**:
  giống ai nhất và đủ giống → gán vào, không → người mới.
- **EMA (Exponential Moving Average)** để cập nhật centroid: `c = α*c + (1-α)*emb` — giúp
  centroid "học" thêm về giọng từng người theo thời gian.
- **Vấn đề kinh điển**: segment ngắn cho embedding kém tin cậy; 2 người giọng giống nhau;
  threshold quá thấp → gộp người, quá cao → đẻ ra người ảo. Vì vậy Phase 2 có bước
  **re-cluster offline cuối phiên** để sửa lại.

**Tài nguyên:**
- [diart](https://github.com/juanmc2005/diart) — thư viện online diarization; đọc README + kiến trúc
  để lấy ý tưởng (không cần dùng làm dependency).
- Code sẵn có của dự án: `meetasr/models/spk/cluster.py` (đọc `CommonClustering` — sẽ tái dùng cho final pass)
  và `meetasr/pipeline.py:_run_spk` (hiểu flow offline hiện tại).
- Khái niệm DER (Diarization Error Rate): https://github.com/nryant/dscore

**Bài tập khởi động (ngày 1–2):**
Lấy 1 file audio 2 người nói, cắt thành các đoạn 2s, trích embedding CAM++ từng đoạn,
tính ma trận cosine similarity, vẽ heatmap (matplotlib). Nhìn heatmap sẽ thấy ngay
2 "khối" speaker — và hiểu vì sao threshold ~0.7 hợp lý.

---

### B.3 AI Engineer 3 — Async Backend + WebSocket

**Kiến thức cần (quan trọng nhất về mặt hệ thống):**
- **asyncio cơ bản**: `async/await`, event loop, `asyncio.create_task`, `asyncio.Queue`.
  Quy tắc vàng: *code blocking (inference GPU, I/O nặng) KHÔNG được chạy trực tiếp trong
  async function* → dùng `loop.run_in_executor(None, blocking_fn, args)`.
- **FastAPI WebSocket**: `websocket.accept()`, `receive_bytes()`, `send_json()`,
  `WebSocketDisconnect`. Mỗi connection là 1 coroutine sống suốt phiên.
- **Backpressure**: audio đến đều đặn 1s/1s; nếu inference chậm hơn → queue phình →
  latency tăng vô hạn. Giải pháp: bounded queue + gộp chunk khi đầy.
- **Chia sẻ model GPU**: load 1 lần lúc startup (trong `lifespan` — xem `meetasr/api/app.py`
  hiện tại), mọi session dùng chung qua lock/queue. KHÔNG load model theo request.

**Tài nguyên:**
- FastAPI WebSocket: https://fastapi.tiangolo.com/advanced/websockets/
- asyncio: https://docs.python.org/3/library/asyncio-task.html (phần Coroutines & Tasks)
- Bài đọc hay: "Common asyncio mistakes" — tìm hiểu vì sao `time.sleep()` / inference blocking
  trong async function làm đứng cả server.
- Client test: thư viện [`websockets`](https://websockets.readthedocs.io/) để viết script giả lập client.

**Bài tập khởi động (ngày 1–2):**
Viết endpoint WS echo: nhận binary, trả `{"bytes": n}`. Viết client Python gửi file wav theo
chunk 250ms có `asyncio.sleep(0.25)` giữa các lần gửi (giả lập realtime). Sau đó thử bỏ một
`time.sleep(2)` (blocking) vào handler và quan sát `/health` bị đứng → hiểu tận gốc event loop.

---

### B.4 Data Engineer — Browser Audio + Persistence

**Kiến thức cần:**
- **Web Audio API**: `navigator.mediaDevices.getUserMedia({audio: true})` → `AudioContext` →
  **AudioWorklet** (worklet chạy trên thread riêng, nhận PCM float32 theo block 128 samples).
  Lưu ý: `ScriptProcessorNode` đã deprecated — dùng AudioWorklet.
- **Resample & encode**: mic thường ở 44.1/48kHz → phải downsample về 16kHz mono;
  float32 [-1,1] → PCM16 (`sample * 32767`, clamp). Gửi qua `websocket.send(arrayBuffer)`.
- **WebSocket JS**: `new WebSocket(url)`, `ws.binaryType = "arraybuffer"`, xử lý
  `onclose` → auto-reconnect với backoff.
- **SQLModel** (đã dùng Phase 1): thêm bảng mới + quan hệ; ghi incremental (mỗi segment final
  1 insert) thay vì ghi 1 lần cuối phiên.

**Tài nguyên:**
- MDN AudioWorklet: https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet
- MDN getUserMedia: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- Ví dụ thực tế đáng đọc: source của [WhisperLiveKit web client](https://github.com/QuentinFuxa/WhisperLiveKit)
  (phần HTML/JS bắt mic gửi WS).
- Code sẵn có: `meetasr/db/models.py`, `meetasr/db/repository.py` (của bạn từ Phase 1).

**Bài tập khởi động (ngày 1–2):**
Trang HTML 1 file: bấm nút → xin quyền mic → AudioWorklet thu 5 giây → downsample 16kHz →
PCM16 → tải xuống file `.wav` (tự viết header WAV 44 byte). Mở file bằng Audacity kiểm tra:
đúng 16kHz mono, nghe rõ. Xong bài này là xong 70% phần khó nhất của mic capture.

---

## C. Thuật ngữ nhanh (cheat sheet)

| Thuật ngữ | Nghĩa |
|---|---|
| RTF | Real-Time Factor = thời gian xử lý / thời lượng audio (cần < 1) |
| Partial / Final | Kết quả tạm (đổi được) / đã chốt (không đổi) |
| LocalAgreement | Chốt phần transcript giống nhau giữa 2 lần decode liên tiếp |
| Endpointing | Phát hiện điểm kết thúc câu nói (dựa vào im lặng) |
| Backpressure | Cơ chế kìm khi producer nhanh hơn consumer (queue giới hạn) |
| Centroid | Vector trung bình đại diện 1 speaker trong clustering online |
| EMA | Trung bình trượt mũ — cách cập nhật centroid dần theo dữ liệu mới |
| DER | Diarization Error Rate — % thời gian gán sai speaker |
| PCM16 | Audio thô 16-bit integer — format gửi qua WebSocket |
| AudioWorklet | API browser xử lý audio realtime trên thread riêng |
