# Phase 2 — Hạn chế & Cách khắc phục (cho team)

> Tài liệu này ghi lại các hạn chế của bản backend Phase 2 **v1**, phần nào **đã khắc phục**
> (kèm cách kiểm chứng và tinh chỉnh), và phần nào **còn lại** (kèm đề xuất triển khai chi tiết).
> Liên quan: kiến trúc ở [11_phase2_notebooklm_plan.md](11_phase2_notebooklm_plan.md),
> phân công ở mục 5 của doc đó.
>
> **Trạng thái kiểm chứng:** code đã pass `py_compile` và frontend đã build sạch, nhưng phần
> chạy thật (ASR/LLM/ffmpeg) **chưa chạy end-to-end trên GPU** — mỗi mục dưới có bước "Cách kiểm chứng"
> để team xác nhận trên máy có model.

---

## Bản đồ nhanh

| # | Hạn chế v1 | Trạng thái | File |
|---|---|---|---|
| 1 | Speaker không nhất quán giữa các chunk | ✅ Đã khắc phục (v1.1) | `meetasr/realtime/speaker.py`, `worker.py` |
| 2 | Nhiều upload tranh GPU (spawn thread vô hạn) | ✅ Đã khắc phục (hàng đợi tuần tự) | `meetasr/realtime/worker.py` |
| 3 | Doc realtime chỉ là text thô, chưa dùng LLM | ⏳ Còn lại — đề xuất §3 | `worker.py`, thêm `doc_stream.py` |
| 4 | Cắt chunk theo thời gian cố định (cắt giữa câu) | ⏳ Còn lại — đề xuất §4 | `worker.py` |
| 5 | Chưa re-cluster speaker offline cuối phiên | ⏳ Còn lại — đề xuất §5 | thêm bước finalize |
| 6 | Chưa export PDF/DOCX (mới có markdown) | ⏳ Còn lại — đề xuất §6 | thêm `export/` |
| 7 | Lưu file local, chưa dùng MinIO/S3 | ⏳ Còn lại — đề xuất §7 | `storage/backend.py` |
| 8 | Chưa có backpressure / giới hạn kích thước WS | ⏳ Còn lại — đề xuất §8 | `events.py`, route WS |
| 9 | Tóm tắt LLM khoá cứng theo template "cuộc họp" — không scale sang nội dung khác | ⏳ Còn lại — đề xuất §9 | `llm/summarizer.py`, `llm/prompts/`, `schemas.py` |

---

## 1. ✅ Speaker nhất quán giữa các chunk

**Vấn đề:** mỗi cửa sổ 30s được diarize độc lập → "Người nói 1" ở đoạn A không liên quan
"Người nói 1" ở đoạn B.

**Đã làm:** `OnlineSpeakerMatcher` (`meetasr/realtime/speaker.py`). Với mỗi câu chốt:
trích embedding CAM++ (192-d) → so cosine với centroid mọi speaker đã biết → giống nhất và
`≥ 0.70` thì gán vào speaker đó và cập nhật centroid bằng EMA; ngược lại tạo speaker mới.
Worker gọi `matcher.assign(span)` cho từng câu để **ghi đè** nhãn cục bộ của chunk.

**Cách kiểm chứng:**
- Upload file 2 người nói ≥ 2 phút → mở tab Lời thoại → nhãn "Người nói N" phải ổn định
  xuyên suốt, số speaker ≈ số người thật (±1).
- Chỉnh ngưỡng nếu lệch: sửa `THRESHOLD` trong `speaker.py` (giọng bị gộp → **tăng** ~0.75;
  một người bị tách làm nhiều → **giảm** ~0.65).

**Còn yếu:** quyết định online, chưa nhìn toàn cục → xem §5 (re-cluster cuối phiên).

---

## 2. ✅ Hàng đợi xử lý tuần tự (bounded)

**Vấn đề:** mỗi upload spawn một thread chạy pipeline → nhiều upload cùng lúc tranh GPU,
dễ hết VRAM.

**Đã làm:** một `queue.Queue` + **một** consumer thread duy nhất (`worker._consume`).
`start()` chỉ đẩy job vào hàng đợi; job chạy lần lượt. Nhiều người upload cùng lúc → xếp hàng,
không sập.

**Cách kiểm chứng:** upload 3 file liên tiếp → `GET /v1/sources` thấy 1 file `processing`,
các file còn lại `queued`, rồi chạy tuần tự. `/v1/health` vẫn phản hồi nhanh khi đang xử lý.

**Nâng cấp sau (tuỳ chọn):** thêm biến môi trường `MEETASR_MAX_QUEUE` để từ chối khi hàng đợi
quá dài, và endpoint xem vị trí hàng đợi.

---

## 3. ⏳ Sinh tài liệu realtime bằng LLM (thay vì text thô)

**Hiện tại:** sự kiện `doc_delta` chỉ ghép text thô của mỗi chunk. Chưa phải "tài liệu có cấu trúc"
như NotebookLM.

**Đề xuất (task AI Engineer 2 — M3):** thêm `meetasr/realtime/doc_stream.py`:

```python
class DocStreamer:
    """Buffer transcript, mỗi ~1500-2500 ký tự gọi LLM sinh/cập nhật 1 section."""
    def __init__(self, llm, min_chars=1800):
        self.llm = llm; self.buf = ""; self.outline = []  # [(section_id, heading)]
    def feed(self, text: str) -> list[dict]:
        self.buf += " " + text
        if len(self.buf) < self.min_chars:
            return []
        # prompt: đưa outline hiện tại + buf → trả JSON {section_id, heading, markdown}
        section = self.llm.chat(prompt_build(self.outline, self.buf))  # parse JSON
        self.buf = ""
        # thêm section mới hoặc cập nhật section cuối
        return [section]
```

Worker gọi `streamer.feed(text)` sau mỗi câu, phát `doc_delta` với `heading` + `markdown`.
Quy tắc quan trọng (chống LLM lan man): **chỉ cho phép thêm section mới hoặc sửa section cuối**,
luôn đưa outline hiện tại vào prompt.

**Cách kiểm chứng:** transcript 30 phút → docs có heading đúng theo chủ đề, không lặp/mất ý;
chi phí token/độ trễ chấp nhận được (đo số lần gọi LLM ≈ tổng_ký_tự / min_chars).

**Lưu ý chi phí:** để mặc định **tắt** (giữ text thô) và bật qua config `realtime_doc: true`,
vì gọi LLM liên tục tốn tiền/thời gian.

---

## 4. ⏳ Cắt chunk theo ranh giới VAD (không cắt giữa câu)

**Hiện tại:** cắt cứng mỗi 30s → có thể cắt ngang một câu, gây rớt/nhân đôi từ ở mép chunk.

**Đề xuất (task AI Engineer 1):** chạy VAD một lần trên toàn audio (đã có `pipeline._run_vad`),
gom các segment VAD thành nhóm ~30s **nhưng luôn cắt tại khoảng lặng**. Truyền từng nhóm
(gồm nhiều segment liền mạch) vào ASR. Ranh giới nhóm = ranh giới segment gần mốc 30s nhất.

**Cách kiểm chứng:** so text ở mép các chunk với bản chạy offline một lần — không được rớt/lặp từ.

---

## 5. ⏳ Re-cluster speaker offline cuối phiên

**Hiện tại:** speaker gán online (§1), chưa tối ưu toàn cục.

**Đề xuất (task AI Engineer 1 — M6):** khi job xong, lấy toàn bộ embedding đã lưu (nên lưu
embedding cùng segment, hoặc tính lại), chạy `CommonClustering`/`SpectralCluster` sẵn có
trên toàn bộ → sinh mapping nhãn sạch → cập nhật `transcript_segments.speaker` → phát sự kiện
`speaker_update {mapping}` để frontend đổi nhãn (hook `useJobEvents` cần xử lý case này).

**Cách kiểm chứng:** nhãn sau re-cluster khớp kết quả pipeline offline Phase 1 trên cùng file.

---

## 6. ⏳ Export PDF / DOCX

**Hiện tại:** tài liệu chỉ có markdown; frontend mới có nút "Tải media", chưa có nút xuất doc.

**Đề xuất (task AI Engineer 2 — M5):** thêm `meetasr/export/`:
- PDF: markdown → HTML (template có CSS) → PDF bằng `weasyprint`, **nhúng font Noto Sans**
  để không lỗi dấu tiếng Việt.
- DOCX: `python-docx` (hoặc `pandoc` nếu cài được).
- Endpoint: `GET /v1/documents/{id}/export?format=pdf|docx|md`.
- Frontend: thêm menu "Xuất" trong `DocumentView` (đã có sẵn chỗ, chỉ cần nối).

**Cách kiểm chứng:** mở PDF đọc được tiếng Việt có dấu; DOCX mở bằng Word không vỡ layout.

---

## 7. ⏳ Chuyển lưu trữ sang MinIO/S3

**Hiện tại:** `StorageBackend` lưu file dưới `data/` local. `docker-compose.yml` đã có sẵn MinIO.

**Đề xuất (task Data Engineer):** thêm implementation `S3Storage` cho cùng interface trong
`storage/backend.py` (dùng `boto3`, trỏ endpoint MinIO từ `.env`). Route không phải sửa vì đã
gọi qua interface. `GET /media` chuyển sang trả **presigned URL** thay vì stream file.

**Cách kiểm chứng:** upload → file xuất hiện trong MinIO console (cổng 9001); phát lại video tua được.

---

## 8. ⏳ Backpressure & giới hạn cho realtime

**Hiện tại:** `EventBus` dùng `asyncio.Queue` không giới hạn; nếu client chậm, queue phình.
Upload giới hạn 500MB (kế thừa Phase 1) nhưng chưa stream kiểm tra dung lượng khi ghi.

**Đề xuất:**
- `asyncio.Queue(maxsize=...)`; nếu đầy thì bỏ bớt sự kiện `status/transcript_delta` cũ
  (client vẫn replay được từ DB khi reconnect).
- Kiểm tra dung lượng **trong lúc** stream ghi đĩa (`save_stream`) và dừng sớm nếu vượt ngưỡng
  cấu hình, thay vì đọc hết vào RAM.

---

## 9. ⏳ Tóm tắt LLM khoá cứng theo template "cuộc họp" — cần thiết kế lại để scale sang mọi loại nội dung

> 📄 **Thiết kế triển khai đầy đủ** (schema, prompt, code từng hàm, migration, test plan):
> [14_document_planner_design.md](14_document_planner_design.md). Phần dưới đây là tóm tắt
> nhận xét + hướng giải quyết; đọc doc 14 khi bắt tay code.

### Nhận xét về cách tóm tắt hiện tại

Đã đọc kỹ `meetasr/llm/summarizer.py` + toàn bộ `meetasr/llm/prompts/`. Đánh giá: cách làm hiện tại
là **template cứng cho một domain duy nhất (cuộc họp công việc)**, và có bằng chứng cụ thể cho thấy
việc "thêm domain mới" đã từng được thử nhưng **không scale** — nên với hướng Phase 2 (tóm tắt bất kỳ
video/audio nào: bài giảng, podcast, phỏng vấn, video giải trí, hội thoại đời thường...), cách này
sẽ gãy hoặc phải vá bằng ngày càng nhiều "if domain == ...".

**Bằng chứng cụ thể:**

1. **Schema đầu ra cố định, mang từ vựng của cuộc họp.** `MeetingReport` (`schemas.py`) luôn có
   đúng 4 phần: `summary`, `topics`, `action_items` (task/**assignee**/**deadline**/priority),
   `decisions` (content/**made_by**). Các trường in đậm chỉ có nghĩa trong bối cảnh công việc.
   Một bài giảng không có "assignee" hay "deadline"; một podcast không có "decisions"; một cuộc
   trò chuyện phiếm không có phần nào trong 4 phần này cả. Không có chỗ để biểu diễn thứ mà những
   nội dung đó THỰC SỰ cần — ví dụ: khái niệm chính + câu hỏi ôn tập (bài giảng), trích dẫn đáng
   chú ý + tiểu sử khách mời (phỏng vấn), dòng thời gian sự kiện (tường thuật), nhân vật + bối
   cảnh (kể chuyện).

2. **Cơ chế "chọn domain" đã được xây nhưng không ai gọi tới — bằng chứng sống của việc không scale.**
   `load_prompts()` (`meetasr/llm/llm_utils/prompts.py`) nhận tham số `prompt_type` để chọn bộ prompt
   theo domain, và **đã có sẵn bộ prompt cho domain "consultation"** (bác sĩ – bệnh nhân) song song
   với "meeting" — ai đó trong nhóm đã cố mở rộng sang domain thứ 2. Nhưng `MeetingSummarizer.__init__`
   gọi `load_prompts(language=self.language)` — **không truyền `prompt_type`**, nên luôn dùng mặc định
   `"meeting"`. Toàn bộ 4 file prompt `*_consultation_vi.txt` là **dead code**, không endpoint nào có
   tham số để chọn tới chúng (`routes/summarize.py` chỉ nhận `language`, `include_topics`,
   `include_actions`, `include_decisions` — không có `domain`/`prompt_type`).
   → Cứ mỗi domain mới là **+4 file prompt** phải viết tay, **+wiring** tham số xuyên suốt API →
   summarizer → prompt loader, và lần trước wiring đó **đã bị quên**. Đây chính là chi phí không
   scale: công sức tăng tuyến tính theo số domain, và dễ vỡ ở khâu nối dây.

3. **`_truncate()` là no-op — "map-reduce" được quảng cáo trong docs cũ chưa thực sự tồn tại.**
   ```python
   def _truncate(self, text: str, max_chars: int = MAX_CHARS_DIRECT) -> str:
       return text   # không cắt gì cả
   ```
   `topics`/`action_items`/`decisions` gọi `self._truncate(text)` nhưng hàm trả nguyên văn — nghĩa là
   với transcript dài (video 1-2 tiếng), toàn bộ text được nhét thẳng vào 1 lần gọi LLM, không chia
   nhỏ. Vượt context window hoặc tốn phí không cần thiết. Tài liệu 08 (Phase 1 team tasks) có nhắc
   "Chunking Logic" + "Map-Reduce" nhưng phần thực thi cho 3/4 lời gọi LLM chưa làm.

4. **Áp lực buộc-điền-schema gây hallucination.** Vì 4 trường luôn bị "hỏi", kể cả khi nội dung
   không có gì đáng liệt kê, model dễ bịa ra action-item/decision yếu ớt để không trả mảng rỗng —
   nhất là ở nhiệt độ thấp (0.3) nhưng vẫn bị ép hình thức. Schema càng cứng, áp lực bịa càng lớn khi
   nội dung không khớp khuôn.

**Kết luận:** vấn đề gốc không phải "cần thêm nhiều template hơn", mà là **kiến trúc lấy template làm
đơn vị mở rộng** (mỗi domain = 4 file .txt + N chỗ code phải sửa) — vốn dĩ tuyến tính-không-scale, và
đã tự chứng minh điều đó bằng domain thứ 2 bị bỏ quên giữa chừng.

### Đề xuất kiến trúc — tách "cấu trúc tài liệu" ra khỏi "loại nội dung"

Ý tưởng cốt lõi: đừng hard-code schema theo domain. Để **LLM tự đề xuất bộ khung mục lục** phù hợp
với nội dung nó vừa đọc, rồi mới viết nội dung cho từng mục. Thêm domain mới = **0 dòng code, 0 file
prompt mới** — vì không còn khái niệm "domain" trong code nữa, chỉ có "outline được đề xuất".

```
Bước A — PLAN (1 lần gọi LLM, rẻ)
  Input:  một đoạn mẫu của transcript (đủ dài để nhận diện thể loại — vd 2-3 đoạn trích đầu/giữa/cuối,
          hoặc toàn văn nếu transcript ngắn)
  Output: JSON
    {
      "content_kind": "mô tả ngắn LLM tự nhận diện, vd 'cuộc họp công việc' | 'bài giảng' | 'podcast'…",
      "outline": [
        {"id": "s1", "heading": "...", "kind": "summary"},
        {"id": "s2", "heading": "...", "kind": "key_points"},
        {"id": "s3", "heading": "...", "kind": "action_items"},   // chỉ xuất hiện nếu THỰC SỰ có
        ...
      ]
    }

Bước B — WRITE (map-reduce theo từng mục, không theo domain)
  Với mỗi mục trong outline: 1 prompt tổng quát duy nhất
    "Viết nội dung cho mục '{heading}' (loại: {kind}) dựa trên transcript sau: {chunk}"
  → chạy map-reduce qua các chunk transcript như summarizer hiện tại làm cho `summarize`,
    áp dụng THẬT SỰ cho mọi mục (khắc phục luôn hạn chế #3 ở trên).
```

**`kind` là tập mở, có gợi ý nhưng không giới hạn cứng** — cho AI Engineer 2 một danh sách gợi ý để
giữ nhất quán khi hiển thị icon/màu ở frontend (`summary`, `key_points`, `action_items`, `decisions`,
`quotes`, `timeline`, `glossary`, `open_questions`, `people`), nhưng **prompt Bước A phải nói rõ**:
nếu nội dung cần một loại mục khác không có trong danh sách, cứ đặt `kind` tự do — không ép vào 4 slot
cố định như hiện tại. Mục không phù hợp thì **không xuất hiện trong outline**, thay vì bị ép trả `[]`.

**Vì sao khớp sẵn với Phase 2:** đây chính xác là hình dạng `DocSection { id, heading, markdown }` đã
thiết kế ở `frontend-next/src/lib/types.ts` và `meetasr/realtime/doc_gen.py` — nên việc này không phải
xây thêm một hệ thống song song, mà là **thay `MeetingSummarizer` bằng một `DocumentPlanner` dùng
chung khuôn `DocSection`** cho cả luồng finalize (`summary`/`full_text`) lẫn luồng realtime (`doc_delta`
ở mục 3 phía trên) của cùng doc này. AI Engineer 2 nên coi đây và mục 3 là **một việc**, không phải hai.

**Việc cần làm (AI Engineer 2):**
1. Thêm `meetasr/llm/planner.py`: hàm `plan_outline(sample_text, llm) -> {content_kind, outline}`.
2. Viết 1 prompt tổng quát cho Bước A (thay toàn bộ `prompts/topics_*`, `action_items_*`,
   `decisions_*`) + 1 prompt tổng quát cho Bước B (thay `prompts/summarize_*` khi cần > 1 mục).
   Không tạo thêm file theo domain nữa.
3. Thêm map-reduce **thật** cho Bước B (sửa hạn chế `_truncate` no-op): chunk transcript theo
   `MAX_CHARS_DIRECT`, viết nháp từng chunk cho mỗi mục, rồi 1 lượt LLM gộp/rút gọn nháp thành bản
   cuối — reduce thật, không phải truncate.
4. Giữ `MeetingSummarizer`/`MeetingReport` cho các route Phase 1 hiện có (không phá API cũ đang chạy),
   nhưng **route Phase 2 mới nên gọi thẳng `DocumentPlanner`**, không đi qua `MeetingSummarizer`.
5. Xoá (hoặc archive) 8 file prompt domain-cứng (`*_meeting_vi.txt`, `*_consultation_vi.txt`) sau khi
   planner thay thế được — tránh dead code lặp lại như đã xảy ra với "consultation".

**Cách kiểm chứng:**
- Cho planner 3 loại transcript khác hẳn nhau (họp công việc, bài giảng 45 phút, hội thoại phiếm
  2 người) → outline sinh ra phải **khác nhau về cấu trúc** (không phải lúc nào cũng ra đúng
  4 mục summary/topics/actions/decisions).
- Với hội thoại phiếm không có quyết định/việc cần làm nào → outline **không chứa** mục
  action_items/decisions (thay vì chứa mục đó với mảng rỗng hoặc bị bịa).
- Transcript dài 90 phút → xác nhận có nhiều lượt gọi LLM map-reduce (log số lần gọi), không phải
  1 lượt duy nhất nhét toàn văn.

---

## Ghi chú tích hợp Frontend (đã nối)

Frontend Phase 2 đã nối thật vào backend:
- `frontend-next/src/lib/api.ts` — REST client qua proxy `/v1`.
- `frontend-next/src/hooks/useJobEvents.ts` — WebSocket + **replay + reconnect** (backoff).
- Trang Thư viện / Upload / Processing / Document đều đọc dữ liệu thật.

**Việc cần làm khi thêm tính năng mới ở trên:**
- §3: `useJobEvents` đã xử lý `doc_delta` có `heading` — chỉ cần backend gửi heading thật.
- §5: bổ sung xử lý sự kiện `speaker_update` trong `useJobEvents` (đổi nhãn theo mapping).
- §6: nối menu "Xuất" trong `DocumentView` vào endpoint export.
- WebSocket nối **thẳng** tới `ws://127.0.0.1:8000` (biến `NEXT_PUBLIC_WS_BASE`), không qua
  proxy Next — vì proxy WS ở dev hay lỗi. Khi deploy, đặt lại biến này.
