# Phase 3 — Port 4 thế mạnh của FunASR & vượt lên cho tiếng Việt

> **Mục tiêu Phase 3:** đưa 4 thứ FunASR đang hơn meetasr vào meetasr — (A) batching/hiệu năng,
> (B) gán speaker theo ranh giới câu punc, (C) streaming/realtime thật (mic sống), (D) triển khai
> production — và **vượt FunASR ở đúng chỗ FunASR không làm được: tiếng Việt**.
>
> Toàn bộ plan này dựa trên khảo sát code thật của FunASR tại `E:\MeetingMindAI\FunASR`
> (trích dẫn file + dòng cụ thể trong từng mục), không dựa trên suy đoán.

---

## 0. Sự thật nền tảng (đọc trước khi tranh luận bất kỳ mục nào)

Ba phát hiện quyết định hình dạng của toàn bộ plan:

1. **FunASR streaming thật CHỈ có tiếng Trung/Anh.** Registry model của FunASR
   (`funasr/download/name_maps_from_hub.py`) chỉ có đúng **một** checkpoint streaming ASR:
   `paraformer-zh-streaming` (zh/en, vocab8404). Không có streaming tiếng Việt, không có
   streaming đa ngôn ngữ. → **Không thể "port model streaming của FunASR" cho tiếng Việt.**

2. **Chính FunASR cũng dùng kiến trúc "VAD-segment + decode lại" cho đa ngôn ngữ.** Server
   realtime mới nhất của họ (`examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py`)
   không dùng model streaming — nó dùng **streaming VAD** (`DynamicStreamingVAD`) chốt từng đoạn,
   decode nguyên đoạn bằng model offline, partial được tạo bằng cách decode lại đoạn đang mở
   (giới hạn cửa sổ 15s để O(L) thay vì O(L²)). → Đây chính là kiến trúc meetasr sẽ theo, và nó
   là **kiến trúc FunASR tự chọn** cho bài toán giống bài toán của chúng ta.

3. **Server production C++ của FunASR KHÔNG có speaker diarization.** `run_server_2pass.sh` chạy
   binary `funasr-wss-server-2pass` với ASR+VAD+PUNC+ITN+LM — không có CAM++. Diarization chỉ có
   trong đường PyTorch `AutoModel`. → meetasr đưa được **diarization vào server realtime** là một
   điểm vượt thật, không phải khẩu hiệu.

**Vượt FunASR nghĩa là gì (trung thực):** không phải vượt về độ rộng model zoo hay độ chín của
C++ runtime — điều đó không thể và không cần. Vượt ở 4 điểm cụ thể, đo được:

| # | Điểm vượt | FunASR | meetasr Phase 3 |
|---|---|---|---|
| 1 | Realtime **tiếng Việt** (mic sống, partial + final) | ❌ không có | ✅ 2-pass VAD-segment |
| 2 | Diarization **trong** server realtime | ❌ (chỉ offline AutoModel) | ✅ speaker tracker online + re-cluster |
| 3 | Đầu ra là **tài liệu** (LLM planner, PDF/DOCX) chứ không chỉ text | ❌ | ✅ đã có Phase 2 |
| 4 | Sản phẩm web hoàn chỉnh (UI, thư viện, storage) | ❌ | ✅ đã có Phase 2 |

---

## P3-A. Batching / hiệu năng xử lý hàng loạt

### Đã làm (Phase 2.5, xong rồi)
`MeetPipeline._run_asr()` đã port đúng thuật toán FunASR `inference_with_vad()`: sort segment
theo độ dài → đóng gói theo ngân sách thời gian (`asr_batch_size_s`) → khôi phục thứ tự → đo RTF.

### Việc còn lại

**A1. Batch GPU thật cho SenseVoice** — *phát hiện quan trọng:* wrapper hiện tại
(`meetasr/models/asr/sense_voice.py:83`) lặp Python từng chunk:
```python
for chunk in audio:
    res = self._model.inference(data_in=[chunk], ...)   # 1 chunk / 1 lời gọi GPU
```
nhưng `SenseVoiceSmall.inference` của FunASR nhận `data_in` là **list** — sửa thành 1 lời gọi cho
cả batch là có batch GPU thật, hưởng trọn lợi ích của sort+pack đã port:
```python
res = self._model.inference(data_in=audio, ...)         # cả batch / 1 lời gọi GPU
```
Việc nhỏ (~10 dòng) nhưng là mảnh ghép còn thiếu để chuỗi sort→pack→batch hoạt động trọn vẹn với
model mặc định của dự án. **Verify:** file 30 phút nhiều segment, đo RTF trước/sau; kết quả text
phải giống hệt bản chạy từng chunk.

**A2. Semaphore giới hạn đồng thời theo model** — port từ `funasr_wss_server.py:298-303`: mỗi
model (VAD/ASR/PUNC/SPK) có một `asyncio.Semaphore` riêng + `ThreadPoolExecutor` chung, chặn nhiều
request cùng chiếm GPU. Hiện meetasr có hàng đợi job tuần tự (1 consumer) — đủ cho batch upload
nhưng **không đủ khi có realtime** (P3-C cần VAD chạy song song với ASR của phiên khác).
Thay hàng đợi 1-consumer bằng: worker pool + semaphore per-model.

**A3. Metric hiệu năng chuẩn** — RTF đã log trong `_run_asr`; thêm endpoint `GET /v1/metrics`
(tổng job, RTF trung bình, độ dài hàng đợi, VRAM) để đo được "trước/sau" mọi tối ưu. Không đoán
hiệu năng — đo.

**Giao cho:** AI Engineer 1. **Ước lượng:** 1 tuần (A1 vài giờ; A2 là phần chính).

---

## P3-B. Gán speaker theo ranh giới câu punctuation (`punc_segment`)

### FunASR làm gì (bằng chứng)

Luồng trong `auto_model.py:685-884`:
1. Mỗi VAD segment → `sv_chunk()` cắt cửa sổ 1.5s/bước 0.75s → CAM++ embedding từng chunk.
2. Punc chạy trên text ASR → giữ lại `punc_res[0]["punc_array"]` (tensor id: **1=không,
   2=phẩy, 3=chấm, 4=hỏi** — `timestamp_tools.py:133-134`).
3. `ClusterBackend` phân cụm embedding: **<20 chunk → 1 speaker; <2048 → SpectralCluster
   (eigen-gap tự tìm số speaker, p-pruning pval=0.022); ≥2048 → UMAP+HDBSCAN**; sau đó
   `merge_by_cos` gộp cụm có cosine centroid ≥ **0.78** (`cluster_backend.py:214-268`).
4. `postprocess()` + `smooth()` (đoạn < **0.7s** bị gán lại cho hàng xóm) → speaker-turn list.
5. `timestamp_sentence()` cắt câu theo `punc_array` (mọi id>1 đều ngắt) + char-timestamps.
6. `distribute_spk()` gán speaker cho từng câu bằng **max-overlap voting**
   (`campplus/utils.py:263-276`).

**Điều kiện tiên quyết FunASR tự thừa nhận:** cần **timestamps từ ASR**. SenseVoice (CTC) không
có → FunASR **tự fallback** về `vad_segment` với warning (`auto_model.py:833-835`). Chỉ
paraformer-vad-punc / seaco_paraformer có timestamp thật.

### meetasr port thế nào

**B1. Port 4 hàm thuần Python** (không GPU, rẻ, an toàn): `timestamp_sentence()` (kèm bảng punc
id→glyph, viết lại glyph cho tiếng Việt: `,` `.` `?`), `distribute_spk()` (max-overlap voting),
`postprocess()`/`merge_seque()`/`smooth()` (mindur 0.7s). Đích: `meetasr/utils/diarization.py`
(đã có sẵn một phần — `compressed_seg`, `assign_speakers_by_overlap` là bản đơn giản của cùng ý
tưởng; thay bằng bản port đầy đủ).

**B2. Nâng ClusterBackend** — `meetasr/models/spk/cluster.py` đã có SpectralCluster/AHC port từ
3D-Speaker; bổ sung 2 thứ FunASR có mà bản port thiếu: **`merge_by_cos` (ngưỡng 0.78)** và
**nhánh UMAP+HDBSCAN cho ≥2048 chunk** (file rất dài mới chạm nhánh này — làm sau nếu chưa cần,
ghi TODO rõ).

**B3. Chiến lược timestamps cho tiếng Việt** — đây là quyết định kỹ thuật quan trọng nhất của B:
- `faster-whisper` wrapper **đã có** char-timestamps (expand từ word-level) → dùng được
  `punc_segment` ngay.
- `zipformer-vi` (sherpa-onnx) có token-level timestamps → dùng được.
- `SenseVoice` — kiểm chứng thực tế xem `output_timestamp=True` (wrapper đang truyền) có trả
  timestamp thật không; nếu không → **giữ đúng fallback `vad_segment` như FunASR làm**, log
  warning giống họ. Không cố bịa timestamp.

**Kết quả mong đợi:** transcript chia câu theo ngữ nghĩa (dấu câu) thay vì theo khoảng lặng VAD,
mỗi câu gán đúng người nói — chất lượng hiển thị + tài liệu LLM tốt hơn rõ rệt (LLM nhận câu
hoàn chỉnh thay vì mảnh vụn VAD).

**Giao cho:** AI Engineer 2. **Ước lượng:** 1.5 tuần. **Verify:** audio 2-3 người nói có sẵn
ground truth thô; so sánh câu-speaker với đường `vad_segment` cũ; DER giảm hoặc ít nhất ranh
giới câu tự nhiên hơn (đánh giá mù bởi 2 người trong team).

---

## P3-C. Streaming / realtime thật (mic sống) — kiến trúc 2-pass tiếng Việt

### Vì sao KHÔNG port `ParaformerStreaming`
Model chỉ có zh/en (mục 0.1). Port cả cơ chế SCAMA chunk-attention mà không có checkpoint tiếng
Việt là vô nghĩa. Thay vào đó port **kiến trúc server** FunASR dùng cho đa ngôn ngữ (mục 0.2).

### Kiến trúc: 2-pass VAD-segment (port từ `serve_realtime_ws.py` + `funasr_wss_server.py`)

```
Mic (browser AudioWorklet, PCM16 16kHz, chunk ~250ms)
  │  WebSocket binary
  ▼
RealtimeSession (port từ RealtimeASRSession, serve_realtime_ws.py:226-410)
  ├─ DynamicStreamingVAD  ← port từ funasr/models/fsmn_vad_streaming/dynamic_vad.py
  │    feed(chunk) → sự kiện [beg,-1] / [-1,end] / [beg,end]
  │    + dynamic silence schedule (nói càng lâu, cắt càng nhanh)
  ├─ PASS 1 — partial: decode lại đoạn đang mở, giới hạn partial_window_sec=15
  │    (chống O(L²)), nhịp decode ~0.5s/lần (decode-interval 0.48s như FunASR)
  │    → gửi {"type":"partial", ...} — text xám trên UI
  └─ PASS 2 — final: khi VAD đóng đoạn ([−1,end]) → decode NGUYÊN đoạn bằng
       SenseVoice/whisper offline + punc + speaker (OnlineSpeakerMatcher đã có
       ở Phase 2) → sentence "khóa" lại → {"type":"final", ...} — text đen
Kết thúc phiên: re-cluster speaker toàn cục (đã có `_recluster_speakers` Phase 2)
  + DocumentPlanner finalize (đã có Phase 2)
```

### Việc cụ thể

**C1. Port `DynamicStreamingVAD`** — file `dynamic_vad.py` của FunASR là wrapper tự chứa, sạch,
~160 dòng: `feed(audio_chunk, is_final) → list [start_ms, end_ms]`, tự thông dịch quy ước
`[beg,-1]/[-1,end]` của FSMN-VAD streaming và tự chỉnh silence-threshold theo schedule. meetasr
đã tải FSMN-VAD rồi (dùng offline) — cùng weight đó chạy được streaming (cùng module
`fsmn_vad_streaming`). Đích: `meetasr/realtime/streaming_vad.py`.

**C2. `RealtimeSession`** — port khung session của FunASR: `audio_buffer`, `locked_sentences`,
`last_partial_text`, `add_audio()`, `decode(is_final)`, nhịp decode theo `should_decode()`.
Kèm 2 chi tiết đáng port nguyên văn: **guard chống hallucination** (cắt n-gram lặp,
`serve_realtime_ws.py:29-60` — model decode đoạn đang mở dở rất hay lặp) và **giới hạn cửa sổ
partial 15s**. Đích: `meetasr/realtime/session.py`.

**C3. WebSocket route mic sống** — `WS /v1/live` (tách khỏi `/v1/jobs/{id}/events` hiện có).
Protocol lấy phần giao của 2 server FunASR, đơn giản hoá:
- Client→Server: JSON `{"type":"start","language":"vi"}` / `{"type":"stop"}`; binary = PCM16.
- Server→Client: `{"type":"partial","text":...}`, `{"type":"final","sentence":{text,start_ms,
  end_ms,speaker}}`, `{"type":"session_end","source_id":...}` — **tái dùng đúng shape sự kiện
  Phase 2** để frontend hook `useJobEvents` mở rộng được thay vì viết mới.
- Cuối phiên: lưu thành `Source` + `TranscriptSegment` như một upload bình thường → toàn bộ
  finalize/export/library của Phase 2 hoạt động miễn phí trên phiên mic sống.

**C4. Frontend mic sống** — trang `/live`: `getUserMedia` + AudioWorklet downsample 16kHz PCM16
(đã có thiết kế chi tiết trong doc 10 cũ — mục 2.4, giờ mới đến lúc dùng), partial xám/final đen
(UI ProcessingView tái dùng ~80%).

> **[ĐÃ XONG — bản web]** Đã triển khai: `frontend-next/src/hooks/useLiveMic.ts` +
> `frontend-next/src/app/live/page.tsx` + `api.liveUrl()`. Verified end-to-end (partial +
> final + session_end + Source lưu DB). Trong quá trình test phát hiện & sửa 2 bug backend:
> (1) gửi sau khi socket đóng → thêm `_safe_send` guard trong `live.py`; (2) uvicorn ws ping
> timeout cắt kết nối khi decode dài → `ws_ping_interval=None` trong `cli.py`. Bản **desktop
> (Tauri + system audio)** là Phase 4 — xem [17_phase4_realtime_desktop.md].

**Giao cho:** AI Engineer 3 (C1-C3 backend) + người phụ trách frontend (C4).
**Ước lượng:** 3 tuần — đây là mục lớn nhất của Phase 3.
**Verify:** nói vào mic 2 phút: partial xuất hiện < 1s sau khi nói; final chốt ≤ 1.5s sau khi
ngừng câu; phiên 30 phút không phình RAM; text final khớp ≥ 95% so với upload cùng audio.

---

## P3-D. Triển khai production

### FunASR làm gì (bằng chứng)
- Export ONNX: `AutoModel.export()` → `export_meta` per-model → `torch.onnx.export(opset 14)` →
  **quantize INT8 động** bằng onnxruntime (`QUInt8`, chỉ MatMul, per-channel).
  Có sẵn `export_meta` cho: **sense_voice ✅, fsmn_vad_streaming ✅, ct_transformer ✅** —
  **campplus ❌ (không export được qua đường này)**.
- Runtime Python ONNX: package `funasr_onnx` (`runtime/python/onnxruntime/`) — class
  `SenseVoiceSmall`, `Fsmn_vad(_online)`, `CT_Transformer(_VadRealtime)` chạy thuần
  onnxruntime, tự export nếu thiếu file.
- Server C++ + docker image dựng sẵn; concurrency = thread pool (`decoder-thread-num` = số core,
  `model-thread-num` intra-op).

### meetasr làm gì (chọn lọc, KHÔNG port C++)

**D1. Đường ONNX cho CPU-serving** — dùng thẳng `funasr_onnx` (đã là package cài được) làm
backend thay thế: thêm wrapper `sensevoice-small-onnx` / `fsmn-vad-onnx` / `ct-punc-onnx` đăng ký
vào registry meetasr, chọn qua config `asr: {model: sensevoice-small-onnx}`. INT8 quantize giúp
máy **không GPU** chạy được với tốc độ chấp nhận được — mở rộng được nơi deploy (VPS rẻ).
CAM++ giữ PyTorch (FunASR cũng vậy — không có export path).

**D2. Docker hoá toàn stack** — FunASR không có Dockerfile tự build server (image kéo sẵn),
nhưng có mẫu tốt: `examples/openai_api/Dockerfile` (python-slim + ffmpeg + libsndfile). Viết:
- `Dockerfile.backend` (python 3.10-slim, ffmpeg, cache model vào volume)
- `Dockerfile.frontend` (node build → serve tĩnh)
- mở rộng `docker-compose.yml` hiện có (đã có Postgres+MinIO): thêm backend + frontend +
  volume model-cache → **một lệnh `docker compose up` chạy cả stack**.

**D3. Concurrency + vận hành** — như P3-A2 (semaphore per-model); thêm graceful shutdown
(đợi job đang chạy xong), `GET /v1/health` trả trạng thái model đã load, log có cấu trúc.

**Giao cho:** Data Engineer (D2, D3) + AI Engineer 1 (D1, gộp với P3-A vì cùng chạm wrapper).
**Ước lượng:** 2 tuần.
**Verify:** D1 — cùng file audio, so text ONNX vs PyTorch (WER chênh < 1%), đo RTF trên máy
không GPU; D2 — máy sạch chỉ có docker: `docker compose up` → upload → tài liệu, không cần cài
gì thêm.

---

## Lịch trình & phụ thuộc (6 tuần)

```
Tuần 1-2:  P3-A (AI1)  +  P3-B1/B2 (AI2)  +  C1 port DynamicStreamingVAD (AI3)  +  D2 docker (DE)
Tuần 3-4:  P3-B3 + tích hợp punc_segment vào pipeline (AI2)  +  C2/C3 session+WS (AI3)
           + D1 ONNX wrappers (AI1)  +  D3 (DE)
Tuần 5:    C4 frontend mic sống (FE + AI3 hỗ trợ)  +  tích hợp B vào cả batch lẫn realtime
Tuần 6:    Đo đạc tổng (RTF, latency partial/final, DER), sửa bug, demo end-to-end
```
Phụ thuộc cứng: C2 cần C1; C4 cần C3; B3 nên xong trước khi C2 dùng punc trong pass-2.
A2 (semaphore) cần xong trước khi C chạy song song với batch upload.

## Rủi ro chính (trung thực)

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| ~~FSMN-VAD streaming là checkpoint zh — chưa kiểm chứng chất lượng trên tiếng Việt~~ **[ĐÃ XỬ LÝ]** | Cao | **Đã đo trên 3 clip Việt thật: FSMN-VAD tag 100% là speech (không lọc nhạc/lặng) → đã CHUYỂN mặc định sang Silero VAD (lọc đúng non-speech, chênh text ASR chỉ 0.8%). Xem [18_chuan_hoa_audio_mic.md] và wrapper `meetasr/models/vad/silero_vad.py`.** |
| Partial decode lặp lại mỗi 0.5s ăn GPU đáng kể khi nhiều phiên mic đồng thời | Cao | Giới hạn số phiên live đồng thời (config), semaphore P3-A2, partial window 15s |
| SenseVoice không có timestamp thật → punc_segment không chạy với model mặc định | Trung bình | Đã có kế hoạch fallback đúng kiểu FunASR; khuyến nghị dùng faster-whisper cho phiên cần diarization đẹp |
| `funasr_onnx` có thể kén version onnxruntime trên Windows | Trung bình | D1 là backend *thay thế* opt-in, không thay mặc định; PyTorch path vẫn nguyên |
| Ước lượng 6 tuần cho team beginner là lạc quan | Cao | Mỗi mục có verify riêng — cắt được C4 hoặc D1 mà không sập các mục khác |

## Những gì Phase 3 cố tình KHÔNG làm
- Không port SCAMA/ParaformerStreaming (vô nghĩa khi không có checkpoint Việt).
- Không viết server C++/gRPC/Triton (đội 4 người không bảo trì nổi; Python + docker đủ cho quy mô hiện tại).
- Không train/fine-tune model streaming tiếng Việt (ngoài năng lực và thời gian — ghi nhận là
  hướng dài hạn nếu dự án lớn lên).
