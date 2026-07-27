# Phase 4 — Ghi âm & ghi chú cuộc họp REALTIME (Desktop hybrid)

> **Mục tiêu Phase 4:** cho phép ghi âm + ghi chú cuộc họp **đang diễn ra** (không chỉ file đã ghi
> sẵn như Phase 1-3). Bắt được **cả mic lẫn system audio** (tiếng người khác trong Google
> Meet/Zoom) — điều web/trình duyệt không làm tốt.
>
> Tài liệu này dựa trên: khảo sát code thật của **meetily** (`_reference/meetily`) và **hiện trạng
> backend realtime** của dự án (Phase 3-C đã hoàn thành). Đọc kỹ mục "Sự thật nền tảng" trước khi
> tranh luận bất kỳ task nào.

---

## 0. Sự thật nền tảng (đọc trước)

Ba phát hiện quyết định hình dạng toàn bộ plan:

1. **Backend realtime + bản mic WEB ĐÃ XONG & verified.** Phase 3-C1/C2/C3 là sản phẩm đang chạy:
   `WS /v1/live` (`meetasr/api/routes/live.py`) có protocol 2 chiều đầy đủ; `RealtimeSession`
   2-pass VAD-segment (`meetasr/realtime/live_session.py`); `DynamicStreamingVAD` +
   `SileroStreamingVAD` (`streaming_vad.py`, `models/vad/silero_vad.py` — **VAD mặc định giờ là
   Silero**); `OnlineSpeakerMatcher` (`speaker.py`); event bus; GPU semaphore. **Frontend mic web
   đã xong**: `useLiveMic.ts` + `/live` page, verified end-to-end (partial+final+session_end+Source).
   Cuối phiên tự lưu thành `Source` + `TranscriptSegment` → finalize/export Phase 2 chạy miễn phí.
   → **Không cần viết lại transcription/diarization/LLM.** Phase 4 = thêm lớp desktop để bắt
   **system audio** (thứ web không làm được).

2. **Meetily là desktop app Tauri (Rust) và điểm mạnh thật sự của họ là AUDIO CAPTURE.** Họ bắt
   mic + system audio bằng CPAL + WASAPI loopback, mix chuyên nghiệp (RMS ducking), resample.
   Nhưng họ **transcribe bằng whisper-rs/Parakeet không tối ưu tiếng Việt** và **không có
   diarization** (để dành bản PRO trả phí). → **Chỉ lấy phần audio capture của họ, không lấy
   phần ASR/summary.**

3. **Trình duyệt không bắt được system audio ổn định.** `getUserMedia` chỉ lấy mic; muốn bắt
   tiếng loa (người khác trong call online) cần quyền hệ thống — chỉ desktop app làm được đáng
   tin cậy. → Đây là lý do **bắt buộc** phải có lớp desktop, không thể thuần web.

**Kết luận kiến trúc — Desktop HYBRID:**
- **Tauri/Rust CHỈ lo capture** mic + system audio → stream PCM16 16kHz qua WebSocket.
- **Backend Python tái dùng nguyên vẹn** (transcribe tiếng Việt + diarization + LLM planner).
- **Notes sinh SAU khi họp kết thúc** (như meetily) — tái dùng finalize Phase 2.
- Tauri **bọc chính `frontend-next` hiện có**, không viết lại UI.

---

## 1. Kiến trúc mục tiêu

```
┌───────────────────────────────────────────────────────────────┐
│  Tauri Desktop App (Windows trước, macOS sau)                  │
│  ┌─────────────────────────┐   ┌───────────────────────────┐  │
│  │ Rust Audio Engine (MỚI) │   │ Next.js UI (frontend-next │  │
│  │ • CPAL mic capture      │   │  nhúng trong Tauri)       │  │
│  │ • WASAPI loopback (sys) │   │ • /live page              │  │
│  │ • mix + resample 16kHz  │──▶│ • transcript partial/final│  │
│  │ • emit PCM16 frames     │IPC│ • nút Start/Stop          │  │
│  └─────────────────────────┘   └───────────┬───────────────┘  │
└──────────────────────────────────────────────┼────────────────┘
                                                │ WS /v1/live (PCM16 16kHz ~250ms/frame)
                                    ┌───────────▼────────────────┐
                                    │ Backend Python (ĐÃ CÓ)     │
                                    │ RealtimeSession 2-pass:    │
                                    │  VAD→partial→final+speaker │
                                    │ → lưu Source → finalize LLM│
                                    └────────────────────────────┘
```

**Nguyên tắc:** audio đi Rust → WebSocket → Python. KHÔNG transcribe trong Rust. Rust chỉ là
"micro thông minh" bắt được cả system audio.

**Protocol WS `/v1/live`** (đã có sẵn, không đổi):
- Client→Server: `{"type":"start","language":"vi"}` → binary PCM16 LE mono 16kHz → `{"type":"stop"}`.
- Server→Client: `{"type":"ready"}`, `{"type":"partial","text":...}`, `{"type":"final",
  "segment":{startMs,endMs,speaker,text}}`, `{"type":"session_end","source_id":...}`.

---
## 2. Phân rã công việc & phân công (5 người)

Mỗi task có **mục tiêu**, **file tham khảo**, **tiêu chí nghiệm thu (verify)** riêng để chạy độc
lập. Thứ tự vai trò theo doc Phase 3 (`15_phase3_plan.md`).

### P4-A — Rust audio capture: MIC  ·  *AI/Rust Engineer 1*
**Mục tiêu:** dựng khung Tauri + bắt được mic, đẩy luồng PCM16 16kHz ổn định lên JS.
- Dựng Tauri shell bọc `frontend-next` (target Windows trước).
- CPAL capture mic → resample 16kHz mono (`rubato`) → PCM16.
- Tauri command `start_capture` / `stop_capture`; emit frame PCM16 ~250ms qua event tới JS.

**Tham khảo (meetily):** `frontend/src-tauri/src/audio/capture/microphone.rs`, `audio/devices/`,
`audio/pipeline.rs` (ring buffer + resample).
**Verify:** lưu 10s PCM ra WAV → nghe lại đúng tiếng nói, đúng 16kHz. Frame đến JS đều đặn ~250ms.
**Ước lượng:** 1 tuần (phần lớn là dựng Tauri lần đầu).

### P4-B — Rust system-audio loopback (phần KHÓ NHẤT)  ·  *AI/Rust Engineer 2*
**Mục tiêu:** bắt được tiếng loa (người khác trong Meet/Zoom) và mix với mic thành 1 luồng.
- WASAPI loopback trên Windows.
- Mix mic + system (RMS ducking chống lấn tiếng) → 16kHz PCM16.

**Tham khảo (meetily):** `audio/capture/system.rs`, `audio/devices/platform/windows.rs`,
`audio/pipeline.rs` (professional mixing). Port **có chọn lọc**, không bê nguyên.
**Verify:** thu 30s vừa nói (mic) vừa phát video có tiếng (loa) → WAV nghe rõ **cả 2 nguồn**,
không méo/lệch tiếng.
**Phụ thuộc:** cần khung capture của P4-A. **Ước lượng:** 1.5 tuần.

### P4-C — Cầu nối Tauri ↔ WebSocket /v1/live  ·  *Frontend Engineer*
**Mục tiêu:** nối luồng PCM từ Rust ra backend qua WebSocket, tái dùng shape event Phase 2.
- Viết hook `useLiveMic.ts` (tên đã được tham chiếu trong comment `public/pcm-worklet.js`):
  nhận PCM16 frame từ Tauri event → mở WS `/v1/live` → gửi `start` → stream binary → `stop`.
- Thêm `api.liveUrl()` vào `frontend-next/src/lib/api.ts` (đã có sẵn `WS_BASE`).
- Xử lý event server: `partial` (text xám), `final` (text đen + speaker), `session_end` → điều
  hướng sang trang Source vừa tạo.

**Tham khảo (nội bộ):** `frontend-next/src/hooks/useJobEvents.ts` (pattern WebSocket + reconnect +
chống trùng segment), `public/pcm-worklet.js`.
**Verify:** mở /live, nói → partial hiện <1s, final chốt khi ngừng câu.
**Phụ thuộc:** cần P4-A emit frame. **Ước lượng:** 1 tuần.

### P4-D — UI trang /live  ·  *Frontend Engineer (nối tiếp C)*
**Mục tiêu:** màn hình họp trực tiếp hoàn chỉnh.
- Trang `/live`: chọn nguồn (mic / mic+system), nút Start/Stop, transcript trực tiếp (partial xám
  → final đen + nhãn speaker), đồng hồ thời lượng, cảnh báo giới hạn phiên đồng thời.
- Cuối phiên: link "Xem tài liệu" → trang Source (finalize Phase 2 tạo PDF/DOCX).

**Tham khảo (nội bộ):** tái dùng ~80% `frontend-next/src/components/ProcessingView.tsx`.
**Verify:** phiên 5 phút mic+system → transcript đúng → Stop → ra Source finalize được.
**Phụ thuộc:** cần P4-C. **Ước lượng:** 1 tuần.

### P4-E — Kiểm chứng & tinh chỉnh backend realtime (audio Việt thật)  ·  *AI Engineer 3*
**Mục tiêu:** đảm bảo backend realtime (đã có) chạy đạt chất lượng với tiếng Việt + system audio.
- **VAD đã đổi sang Silero** (mặc định hiện tại — lọc non-speech tốt cho tiếng Việt; FSMN cũ tag
  100% speech). Đường mic web đã verified end-to-end (xem [15_phase3_plan.md] C4). Việc còn lại
  của P4-E: test với **system audio** (từ Tauri) và phiên **dài** trên thiết bị đích.
- Đo: latency partial (<1s), latency final (≤1.5s sau ngừng câu), phiên 30 phút không phình RAM,
  text final khớp ≥95% so với upload cùng audio.
- **Đã biết (đo phiên web):** lần decode partial ĐẦU tiên chậm (RTF ~1.2, model warmup), các lần
  sau ~0.2. Đoạn nói dài không nghỉ khiến partial-window 15s decode tốn tới ~1.9s/lần — theo dõi
  khi họp dài trên CPU; giảm `partial_window_sec` / `stream_max_speech_duration` nếu dồn ứ.
- Tinh chỉnh `MEETASR_MAX_LIVE`, semaphore GPU nếu chạy song song batch upload.

**Verify:** bảng số liệu latency + RTF + RAM cho phiên 30 phút; fallback silero-vad nếu VAD tệ.
**Phụ thuộc:** chạy song song ngay từ tuần 1 (backend đã có). **Ước lượng:** xuyên suốt.

### P4-F — Tài liệu hoá & vận hành  ·  *Data Engineer*
**Mục tiêu:** người khác dựng lại được toàn bộ.
- Docs: build Tauri trên Windows, cài đặt, xin quyền mic/loopback, chạy backend cùng app.
- Cập nhật `meet_docs/`; mở rộng docker-compose nếu backend chạy tách.

**Verify:** máy sạch làm theo docs → chạy được end-to-end.
**Ước lượng:** 1 tuần (cuối Phase).

---

## 3. Lịch trình & phụ thuộc (4 tuần)

```
Tuần 1:  P4-A (Rust mic capture)          +  P4-E bắt đầu test backend với audio Việt
Tuần 2:  P4-B (system loopback, cần A)    +  P4-C (hook, cần A emit frame)
Tuần 3:  P4-D (UI /live, cần C)           +  P4-E đo latency end-to-end
Tuần 4:  Tích hợp toàn bộ + P4-F docs + demo end-to-end; sửa bug
```
**Phụ thuộc cứng:** B cần A; C cần A; D cần C. E độc lập, chạy song song từ đầu.

---

## 4. Files chính sẽ tạo/sửa (khi triển khai)

| Loại | Đường dẫn | Ghi chú |
|---|---|---|
| Mới (Rust) | `frontend-next/src-tauri/` | Tauri shell + `audio/` (mic, system, mix, resample). Port chọn lọc từ `_reference/meetily/frontend/src-tauri/src/audio/` |
| Mới (FE) | `frontend-next/src/hooks/useLiveMic.ts` | Hook nối Tauri event → WS /v1/live |
| Mới (FE) | `frontend-next/src/app/live/page.tsx` | Trang họp trực tiếp |
| Sửa nhỏ (FE) | `frontend-next/src/lib/api.ts` | thêm `liveUrl()` |
| Tái dùng | `public/pcm-worklet.js`, `components/ProcessingView.tsx` | không sửa hoặc sửa rất ít |
| Backend | (hầu như không đổi) | chỉ tinh chỉnh config nếu P4-E cần |

---

## 5. Rủi ro (trung thực)

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| WASAPI loopback (P4-B) là Rust khó, team mới | Cao | Port trực tiếp từ meetily (đã chạy production Windows); cắt B thì A+C+D vẫn cho sản phẩm chỉ-mic |
| ~~FSMN-VAD checkpoint tiếng Trung — cắt câu tiếng Việt chưa kiểm chứng~~ **[ĐÃ XỬ LÝ: đổi Silero]** | — | Đã đo & chuyển mặc định sang Silero VAD (lọc non-speech đúng); FSMN vẫn giữ để fallback |
| Partial decode đoạn nói dài dồn ứ trên CPU (đo được ~1.9s/lần) | Trung bình | Giảm `partial_window_sec`/`stream_max_speech_duration`; warmup model lúc khởi động |
| Dựng Tauri lần đầu tốn thời gian (toolchain Windows) | Trung bình | P4-A dành trọn tuần 1 cho khung + mic |
| Độ trễ Rust→WS→Python cao hơn desktop thuần | Thấp | WS localhost, PCM16 16kHz nhẹ (~32KB/s); đo ở P4-E |
| Đồng bộ clock mic vs system lệch | Trung bình | Ring buffer + resample như meetily (`pipeline.rs`) |

---

## 6. Cố tình KHÔNG làm (giữ phạm vi)

- KHÔNG transcribe/summary trong Rust — tái dùng backend Python (giữ chất lượng tiếng Việt +
  diarization + LLM planner đã đầu tư).
- KHÔNG sinh notes realtime trong lúc họp — summary sinh **sau** khi kết thúc (tái dùng finalize
  Phase 2). Có thể nâng cấp về sau.
- KHÔNG làm macOS/Linux ở Phase 4 (Windows trước; ghi TODO).
- KHÔNG bỏ web app hiện có (Tauri bọc `frontend-next`, dùng lại toàn bộ UI).

---

## 7. Nghiệm thu tổng Phase 4

1. Máy Windows: mở app → chọn mic+system → họp thử (phát video Việt + nói) 10 phút.
2. Transcript trực tiếp: partial <1s, final ≤1.5s sau ngừng câu, có nhãn speaker.
3. Bấm Stop → tạo Source → finalize ra tài liệu PDF/DOCX (luồng Phase 2).
4. Số liệu latency/RAM (P4-E) đạt ngưỡng; text final khớp ≥95% so với upload cùng audio.
5. Máy sạch làm theo docs P4-F chạy được end-to-end.

