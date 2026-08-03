# Realtime Pipeline V2 — Implementation Plan

## 1. Mục tiêu

Nâng cấp luồng live microphone để transcript được chốt và lưu ngay trong phiên,
sau đó tái sử dụng transcript này cho diarization và tạo tài liệu. Full pipeline
offline vẫn được giữ làm fallback cho đến khi realtime chứng minh không mất dữ
liệu.

Luồng mục tiêu:

```text
PCM16 mono 16 kHz
  -> streaming VAD
  -> partial ASR (preview, không lưu DB)
  -> confirmed utterance ASR (lưu DB)
  -> post-session diarization
  -> speaker update
  -> live document
  -> done
```

## 2. Phạm vi và giả định đã chốt

- Làm việc trên nhánh `anhtu/feat/realtime_pipeline-v2`.
- Giữ Qwen3-ASR 1.7B làm ASR chính; không chuyển sang Faster-Whisper.
- Qwen3-ASR tiếng Việt hiện không có word/character timestamp.
- Trong lúc ghi chỉ hiển thị transcript, chưa gán speaker realtime.
- Speaker được gán trong final pass sau phiên bằng CAM++ và clustering offline.
- Không xóa `FinalTranscriptWorker` ngay; refactor nó thành finalizer có fallback.
- Không thay đổi pipeline upload/offline ngoài phần API dùng chung thật sự cần thiết.
- Không thêm running LLM summary trong phạm vi này.
- Không thay đổi thiết kế hình ảnh hiện tại; frontend chỉ bổ sung trạng thái cần
  thiết cho stop/finalization và lỗi.

## 3. Quyết định khi tài liệu có xung đột

`meet_docs/constraints/technical.md` nói model không được chia sẻ giữa request,
trong khi tài liệu realtime và code hiện tại yêu cầu một model GPU được load một
lần. Với giới hạn VRAM và kiến trúc hiện hành, Pipeline V2 dùng quyết định sau:

- Một model ASR GPU được dùng chung ở cấp application.
- Tất cả inference phải đi qua một coordinator/queue tuần tự.
- Không gọi đồng thời trực tiếp vào model từ `TempASRWorker`, `ASRWorker` và
  upload/fallback worker.
- Session state, VAD state và audio buffer vẫn tách riêng cho từng WebSocket.

Các tài liệu `09_realtime_phase2.md` và `10_realtime_team_tasks.md` đã được đánh
dấu là tài liệu cũ. Các ý tưởng endpointing, backpressure và model queue vẫn được
tái sử dụng; thiết kế LocalAgreement dành cho Faster-Whisper không được áp dụng
nguyên trạng cho Qwen.

## 4. Contract dữ liệu

### 4.1 Audio vào

- PCM16 little-endian.
- Mono, 16 kHz.
- Browser gửi binary frame khoảng 100 ms.
- Server từ chối frame sai kích thước hoặc sai contract bằng event lỗi có cấu
  trúc; không để inference crash.

### 4.2 Client -> realtime WebSocket

```json
{"type": "start", "language": "vi"}
```

Binary frame sau `start` chứa PCM16.

```json
{"type": "stop"}
```

`stop` là yêu cầu ngừng nhận audio mới và drain toàn bộ audio đã nhận. Mất kết
nối đột ngột phải đi qua cùng cleanup path nhưng không thể gửi acknowledgement
trực tiếp cho socket cũ.

### 4.3 Server -> client

Giữ contract Phase 2 hiện có và bổ sung lifecycle event tối thiểu:

```json
{"type": "session_init", "source_id": "...", "job_id": "..."}
{"type": "transcript_partial", "segment": {"id": null, "start_ms": 0, "end_ms": 1500, "speaker": null, "text": "..."}}
{"type": "transcript_delta", "segment": {"id": 101, "start_ms": 0, "end_ms": 4200, "speaker": null, "text": "..."}}
{"type": "stream_stopping", "job_id": "..."}
{"type": "stream_stopped", "job_id": "..."}
{"type": "error", "code": "...", "message": "..."}
```

Job event WebSocket tiếp tục dùng:

```text
status -> transcript_delta -> speaker_update -> doc_delta -> done
```

Mọi event replayable phải được persist trước khi publish.

## 5. Streaming state machine

### 5.1 VAD state

```text
IDLE
  -> đủ speech liên tiếp
  -> SPEAKING

SPEAKING
  -> đủ silence liên tiếp
  -> flush(reason=speech_end)
  -> IDLE

SPEAKING
  -> đạt max utterance
  -> flush(reason=max_window)
  -> tiếp tục SPEAKING nếu speech vẫn còn

IDLE hoặc SPEAKING
  -> stop/disconnect
  -> flush(reason=session_stop)
```

Thông số khởi đầu, phải config-driven và benchmark trước khi coi là tối ưu:

```yaml
realtime:
  vad:
    frame_ms: 32
    start_threshold: 0.50
    end_threshold: 0.35
    min_speech_ms: 250
    min_silence_ms: 600
    pre_roll_ms: 300
    post_roll_ms: 100
  asr:
    partial_interval_ms: 800
    partial_min_audio_ms: 700
    partial_max_audio_ms: 5000
    max_utterance_ms: 15000
```

Partial đầu tiên chỉ được yêu cầu khi VAD đã tích lũy đủ 700 ms audio của
utterance. Mỗi request sau đó thay request preview cũ đang chờ và chỉ mang tối
đa 5 giây audio gần nhất. Model được warm-up khi backend khởi động, trước khi
WebSocket nhận audio người dùng.

`SileroVAD.detect()` hiện là API full-audio và reset/mutate model state. Streaming
VAD phải dùng adapter stateful riêng; không gọi lại `detect()` trên toàn bộ buffer
mỗi 32-100 ms.

### 5.2 Partial và confirmed

`transcript_partial`:

- Là preview, được phép thay đổi hoặc drop khi inference backlog.
- Không persist.
- Chỉ giữ request partial mới nhất cho mỗi session.

`transcript_delta`:

- Sinh khi `speech_end`, `max_window` hoặc `session_stop`.
- Có absolute `start_ms`/`end_ms`, không suy ra bằng cách cộng duration window.
- Persist và commit trước khi publish.
- Có stable database ID.
- Không được thay đổi text âm thầm sau khi đã chốt; mọi replacement phải giữ
  contract ID rõ ràng.

## 6. Graceful stop và coverage

Thứ tự dừng bắt buộc:

```text
stop accepting audio
  -> flush AudioReceiver bytes
  -> finalize VAD state
  -> enqueue final utterance
  -> wait audio_queue.join()
  -> wait confirmed ASR queue.join()
  -> record coverage result
  -> enqueue post-session finalizer
  -> cancel remaining preview/session tasks
```

Không được cancel worker trước khi confirmed queue đã drain.

Coverage không chỉ là `TranscriptSegment` khác rỗng. Session phải theo dõi:

- Những speech range đã tạo.
- Những range đã ASR confirmed thành công.
- ASR failure count.
- Queue item bị drop (confirmed item không được drop).
- Final flush có hoàn tất hay không.

Policy ban đầu:

```text
coverage complete -> diarization-only finalization
coverage incomplete -> full offline pipeline fallback
```

Selective ASR cho riêng missing range là tối ưu tiếp theo, không phải điều kiện
để bỏ full fallback ở phiên bản đầu.

## 7. Persistence và replay

Mỗi confirmed utterance tạo một `TranscriptSegment` Phase 2:

```text
job_id, start_ms, end_ms, speaker=null, text
```

Trình tự transaction:

```text
insert/update segment
  -> commit
  -> refresh stable ID
  -> publish transcript_delta
```

Không persist partial. Retry cùng utterance không được tạo duplicate. Cách nhận
diện retry phải được chốt trong bước persistence sau khi kiểm tra migration hiện
có; ưu tiên giải pháp ít thay đổi schema nhất.

Quyết định Bước 5: confirmed realtime dùng khóa tự nhiên
`(job_id, start_ms, end_ms)`. Retry tuần tự trả lại record đã commit, giữ nguyên
ID và text đầu tiên đã chốt. Chưa thêm unique index vì mỗi Job hiện chỉ có một
confirmed ASR consumer; nếu Bước 7 cho phép nhiều consumer cùng ghi một Job thì
phải nâng invariant này thành unique index/migration trước.

Job snapshot phải replay được:

- Stage/progress gần nhất.
- Confirmed transcript đã commit.
- Live document đã commit.
- `done` hoặc `error` nếu đã terminal.

## 8. Post-session diarization

Khi coverage đầy đủ:

```text
full audio or durable audio spool
  -> speech ranges
  -> CAM++ embeddings
  -> global clustering
  -> speaker timeline
  -> align speaker timeline with TranscriptSegment
```

Quyết định Bước 6: worker dùng public API
`MeetPipeline.finalize_realtime_transcript()`; API này chỉ chạy diarization và
punctuation cần thiết trên các segment realtime đã persist, không gọi lại ASR.
Worker không gọi private diarization method trực tiếp.

Nếu coverage thiếu, full pipeline chạy đúng một lần. Các segment realtime đã
publish được giữ nguyên ID, text và mốc thời gian; kết quả offline chỉ cập nhật
speaker cho segment đã được che phủ ít nhất 80%, đồng thời thêm những câu thực
sự chưa được che phủ. Policy merge này ưu tiên không xóa dữ liệu đã hiển thị;
ranh giới sát nhau có thể được xử lý bảo thủ và sẽ được đo thêm ở E2E.

### 8.1 Phân loại segment

Do Qwen tiếng Việt không có character timestamp, mỗi transcript segment được
phân loại bằng overlap của speaker timeline:

```text
single:
  dominant speaker rõ ràng

mixed:
  secondary ratio đáng kể hoặc có secondary turn đủ dài

uncertain:
  speaker turn quá ngắn/rung nhãn hoặc diarization không đủ dữ liệu
```

Ngưỡng khởi đầu để test, không hardcode trong thuật toán:

```text
dominant ratio: 0.80
secondary ratio: 0.15
meaningful secondary turn: 400 ms
very short jitter: dưới 150 ms
```

Segment `single` giữ nguyên text và chỉ cập nhật speaker. Segment `mixed` không
được chia text theo tỷ lệ ký tự và hiện giữ `speaker=null`. Segment `uncertain`
cũng giữ `speaker=null`.

Quyết định Bước 6: log/đếm cả `single`, `mixed`, `uncertain`; selective re-ASR
vẫn tắt và chỉ được bật sau khi có test chứng minh không làm mất text.

### 8.2 Targeted re-ASR sau nghiệm thu thực tế

Ba bản ghi đối chiếu cho thấy CAM++ vẫn tìm được hai cụm giọng, nhưng các đoạn
ASR realtime dài có thể chứa nhiều speaker. Vì Qwen tiếng Việt không có timestamp
từng chữ, finalizer áp dụng policy mới:

```text
single -> giữ nguyên text, chỉ cập nhật speaker
mixed/uncertain có diarization coverage an toàn
       -> ASR lại từng speaker turn trong đúng segment đó
coverage không đủ hoặc một turn ASR lỗi
       -> giữ nguyên toàn bộ segment cũ với speaker=null
```

Replacement là atomic theo từng segment realtime: chỉ thay câu cũ khi tất cả
speaker turn của câu đó đều ASR thành công. Speaker ID cuối được chuẩn hóa theo
thứ tự xuất hiện. Full-file ASR vẫn chỉ là fallback khi coverage realtime của cả
phiên không đầy đủ.

Quyết định Bước 2:

- Diarization lập plan trong một `FINALIZE` request, không gọi ASR bên trong.
- Mỗi speaker turn cần decode được gửi thành một request `TARGETED` riêng để
  confirmed realtime của phiên khác có thể chen vào giữa các turn.
- Segment `single` giữ nguyên row/ID và chỉ cập nhật speaker.
- Segment được thay thế xóa row cũ và insert các speaker turn mới trong cùng một
  transaction. Validation hoặc insert lỗi sẽ rollback toàn bộ transaction.
- Live document chỉ được tạo sau khi transaction transcript đã commit.
- Sau transaction commit, finalizer phát `transcript_snapshot` chứa toàn bộ
  transcript chính thức. Frontend thay toàn bộ state hiện tại bằng snapshot để
  xóa row mixed cũ, partial cũ và nhận các speaker turn mới.
- `speaker_update` vẫn được giữ để tương thích với client cũ; snapshot là nguồn
  đồng bộ cuối cùng cho client đã cập nhật.

Quyết định và kết quả Bước 4:

- Log finalizer ghi cả `asr_audio_ms` và `asr_audio_ratio`; tỷ lệ này chỉ tính
  các speaker turn thực sự gửi lại vào ASR, không tính VAD/diarization.
- Integration test 3.000 ms chỉ ASR lại hai turn `1.800-2.300` và
  `2.300-3.000`: tổng 1.200 ms, tương đương `0.400` (40% audio). Full-file ASR
  không được gọi khi coverage realtime đầy đủ.
- Smoke run thực tế `realtime_2026-08-03T09:50:00.949101.wav` dài 141,4 giây
  hoàn tất với 18 segment: 15 speaker 0, 1 speaker 1 và 2 unknown. Một segment
  realtime đã được thay bằng ba targeted rows, xác nhận transaction và snapshot
  hoạt động trên runtime thật.
- Hai segment unknown được giữ có chủ đích khi diarization coverage không đủ
  hoặc targeted ASR không tạo được replacement an toàn. DB không lưu metric của
  lượt chạy cũ, nên không suy diễn tỷ lệ ASR chính xác từ duration của output;
  các lượt chạy sau đọc trực tiếp tỷ lệ từ log finalizer.

Nghiệm thu E2E tab audio sau khi restart backend bằng code mới:

- Input WAV dài 120,4 giây, PCM16 mono 16 kHz; media endpoint trả partial content
  và Job hoàn tất không lỗi.
- Realtime tạo 23 confirmed segment. Finalizer giữ 15 segment `single`, chọn 8
  segment để targeted re-ASR và tạo 20 speaker turn thay thế.
- Transcript cuối có 35 segment: speaker 0 có 15 segment/41.346 ms speech,
  speaker 1 có 20 segment/37.978 ms speech, không còn segment unknown.
- Targeted ASR xử lý 36.544 ms, bằng 30,4% thời lượng file. Coordinator ghi nhận
  20 request `targeted`, 0 full fallback và 0 inference failure.
- Transcript snapshot, live document và full-text document đều được tạo sau khi
  transaction transcript hoàn tất.
- Kết luận: luồng hybrid hoạt động đúng và không chạy lại ASR toàn file. Chất
  lượng ranh giới speaker vẫn chưa tuyệt đối; một số lượt ngắn làm câu bị vụn,
  nên cần đánh giá bằng ground truth nếu muốn tối ưu threshold tiếp theo.

## 9. Final document và terminal events

Thứ tự finalizer:

```text
status(transcribing/finalizing)
  -> update speakers
  -> publish speaker_update
  -> publish transcript_snapshot
  -> status(generating_doc)
  -> create/update live document
  -> commit document and source duration
  -> mark job done
  -> publish doc_delta
  -> publish done
```

Nếu lỗi:

```text
rollback active transaction
  -> mark job failed with sanitized message
  -> publish error
```

Không đánh dấu `done` trước khi live document đã commit.

## 10. Trình tự triển khai

### Bước 1 — Planning baseline

- Tạo `implementation_plan.md` và `task.md`.
- Chốt scope, contract, fallback và test matrix.

### Bước 2 — Job events correctness

- Viết test tái hiện Job đã xong trong DB nhưng client không nhận `done`.
- Publish status/transcript/document/done/error từ final worker.
- Sửa source duration và terminal stage.

### Bước 3 — Stop protocol và graceful drain

- Client gửi control message `stop` thay vì đóng ngay.
- Server nhận binary và JSON frames.
- Flush/drain queues rồi mới cancel task.
- Giữ disconnect fallback.

### Bước 4 — Streaming VAD và utterance window

- Thêm stateful streaming VAD adapter.
- Thay `split_fixed_segments(1s)`.
- Window trở thành giới hạn tối đa, không phải điều kiện chờ bình thường.
- Giữ absolute timeline qua silence.

### Bước 5 — Persist confirmed transcript

- Persist delta, commit rồi publish.
- Stable ID, dedup/retry và replay.
- Không persist partial.

### Bước 6 — Post-session finalizer

- Coverage check.
- Diarization-only happy path.
- Full pipeline fallback khi thiếu.
- Speaker/document/terminal events.
- Mixed/uncertain policy; selective re-ASR chỉ bật khi test đạt.

### Bước 7 — Inference scheduling và resource safety

- Một inference coordinator tuần tự cho model dùng chung.
- Partial latest-wins và được phép drop.
- Confirmed/fallback không được drop.
- Audio dài spool xuống file/storage thay vì giữ toàn bộ trong RAM.
- Xóa lần gọi VAD trùng trong realtime pipeline.

Quyết định Bước 7:

- App chỉ tạo một `InferenceCoordinator` cho model được full pipeline và
  realtime pipeline dùng chung.
- Priority không preemptive: `confirmed` → `fallback` → `upload` → `finalize`
  → `partial`. Một inference đã bắt đầu vẫn phải chạy xong; confirmed chỉ vượt
  các request còn đang chờ.
- Partial dùng generation theo Job/session; request cũ chưa chạy sẽ bị thay thế.
  Confirmed và fallback không có nhánh drop.
- Final queue giữ giới hạn nhưng `put()` chờ chỗ trống, không bỏ Job khi đầy.
- Audio PCM16 dưới 16 MiB giữ trong RAM; vượt ngưỡng chuyển sang file tạm và
  final worker chỉ materialize khi lấy Job ra xử lý. File được xóa ở terminal
  path hoặc khi queue bị clear lúc shutdown.
- Confirmed ASR tiếp tục dùng `skip_vad=True`; stateful streaming VAD là lần VAD
  duy nhất trước confirmed ASR.
- Metrics live được công bố tại `GET /v1/metrics/inference`: queue depth hiện
  tại/cực đại, average wait/run latency, ASR call count, fallback count,
  partial drop, failure và số hoàn tất theo loại.
- Baseline partial latency tách average wait/run theo từng inference kind. Mỗi
  preview phát ra log thêm worker wait, inference time, request-to-emit,
  first-audio-to-partial, utterance-to-partial và update interval. Bước đo này
  không thay đổi timing `700/800/5000 ms` hay priority hiện tại.

Hotfix nghiệm thu sau Bước 7:

- Client gửi JWT trong control frame đầu tiên của WebSocket; backend xác thực
  trước khi tạo Job và gắn owner vào Source để Library truy vấn được.
- Audio raw dùng cho inference đồng thời được đóng gói WAV PCM16 mono 16 kHz,
  lưu qua `StorageBackend` và commit `storage_path` trước khi enqueue finalizer.
- Post-session diarization chạy lại VAD toàn file để có speech boundary ổn định
  như nhánh offline; đây không phải lần VAD lặp trước confirmed ASR.
- Clustering exception giữ speaker unknown thay vì trả tất cả label 0.
- ProcessingView hiển thị audio/video player khi Job hoàn tất.

### Bước 8 — E2E nghiệm thu và cleanup

- Benchmark latency/RTF, memory và ASR call count.
- Chạy frontend lint/build và backend test suite liên quan.
- Chỉ xóa code cũ khi fallback và replay đã được chứng minh.
- Cập nhật `walkthrough.md`.

## 11. Test matrix bắt buộc

Unit/integration test không được tải model thật; dùng fake VAD/ASR/SPK và hoàn
thành dưới hai giây mỗi test logic.

| Trường hợp | Kết quả mong đợi |
|---|---|
| Audio rỗng | Không crash; error hoặc done rỗng theo contract |
| Chỉ silence | Không gọi ASR confirmed; kết thúc sạch |
| Noise ngắn | Không tự mở utterance nếu dưới điều kiện speech |
| Phiên dưới 1 giây | Stop flush an toàn, không deadlock |
| Một câu rồi im lặng | Delta sinh theo speech end |
| Hai câu có silence | Hai utterance đúng absolute timeline |
| Nói liên tục | Forced flush theo max window |
| Stop giữa câu | Câu cuối được flush |
| Disconnect bất ngờ | Finalizer/fallback vẫn chạy |
| ASR partial chậm | Partial cũ bị drop, delta không bị chậm vô hạn |
| Confirmed ASR lỗi | Coverage incomplete và full fallback |
| DB commit lỗi | Không publish event chưa persist |
| Reconnect | Snapshot replay transcript/document/done |
| Một speaker | Speaker gán bằng overlap |
| Secondary turn ngắn | Không bỏ chỉ vì tỷ lệ dưới 15% |
| Mixed speaker | Không chia text theo tỷ lệ ký tự |
| Diarization lỗi | Transcript vẫn được giữ; fallback/policy rõ ràng |
| Document lỗi | Job failed và client nhận error |

Live model test được tách riêng và chỉ chạy khi bật biến môi trường hiện có.

## 12. Definition of Done

- Transcript confirmed xuất hiện theo speech end hoặc max window, không đợi gần
  30 giây trong trường hợp thông thường.
- Stop/disconnect không làm mất tail audio.
- Confirmed transcript được lưu và replay từ DB.
- Happy path không chạy lại full-audio ASR sau phiên.
- Coverage không đầy đủ luôn đi qua full fallback.
- Speaker update không làm mất hoặc tự chia text thiếu căn cứ.
- Live document được commit trước `done`.
- Frontend luôn nhận `done` hoặc `error`, không chờ vô hạn.
- Model inference không chạy đồng thời ngoài coordinator.
- Test, lint và build liên quan đều pass.
- `task.md` và `walkthrough.md` phản ánh đúng trạng thái cuối.
