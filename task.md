# Realtime Pipeline V2 — Task Tracker

Quy ước:

- `[ ]` chưa bắt đầu
- `[/]` đang thực hiện
- `[x]` đã hoàn thành và verify

## Bước 1 — Planning baseline

- [x] Đọc `claude.md`.
- [x] Đọc toàn bộ `meet_docs/constraints/`.
- [x] Đọc toàn bộ `meet_docs/skills/` và tài liệu realtime được tham chiếu.
- [x] Đối chiếu tài liệu với code realtime/Qwen/VAD/DB hiện tại.
- [x] Ghi quyết định xử lý xung đột model sharing.
- [x] Tạo `implementation_plan.md`.
- [x] Tạo `task.md`.
- [x] Người dùng xác nhận chuyển sang Bước 2.

## Bước 2 — Job EventBus correctness

- [x] Viết test tái hiện final worker không phát terminal events.
- [x] Persist trước, publish sau.
- [x] Publish `status`, `transcript_delta`, `doc_delta`, `done`.
- [x] Publish `error` khi finalization thất bại.
- [x] Cập nhật `Source.duration`, Job stage/progress và timestamp.
- [x] Chạy backend tests liên quan (16 test pass; 1 vertical-slice test nền lỗi do thiếu `current_user`).
- [x] Người dùng xác nhận chuyển sang Bước 3.

## Bước 3 — Stop protocol và graceful drain

- [x] Viết test cho stop/disconnect và tail flush.
- [x] Frontend gửi JSON `stop` thay vì đóng socket ngay.
- [x] Backend nhận được binary audio và control JSON.
- [x] Flush receiver, VAD/window cuối và confirmed ASR queue.
- [x] Chỉ cancel task sau khi drain hoàn tất.
- [x] Bổ sung trạng thái UI “Đang hoàn tất”.
- [x] Chạy backend tests và frontend lint/build liên quan.
- [x] Người dùng xác nhận chuyển sang Bước 4.

## Bước 4 — Streaming VAD và max-window

- [x] Viết unit test cho VAD state machine bằng fake predictor.
- [x] Thêm stateful streaming VAD adapter cho session.
- [x] Thay `split_fixed_segments(1s)`.
- [x] Thêm pre-roll, endpointing và stop flush.
- [x] Dùng max utterance làm giới hạn, không chờ mặc định 30 giây.
- [x] Giữ absolute timeline qua các khoảng silence.
- [x] Chạy unit/integration tests liên quan.
- [x] Khóa lazy-load và inference Qwen dùng chung.
- [x] Bước 4.1: warm-up Qwen khi backend khởi động.
- [x] Bước 4.1: Partial đầu ở 700 ms, refresh 800 ms, rolling window 5 giây.
- [x] Bước 4.1: Partial dùng ngôn ngữ `vi` và pending request latest-wins.
- [/] Xác nhận runtime Qwen và đo first-partial latency.
- [x] Người dùng xác nhận chuyển sang Bước 5.

## Bước 5 — Persist confirmed realtime transcript

- [x] Viết test persist-before-publish.
- [x] Chỉ persist `transcript_delta`, không persist partial.
- [x] Trả stable segment ID.
- [x] Chống duplicate khi retry.
- [x] Publish cùng contract cho recording WS và Job EventBus.
- [x] Verify reconnect/replay từ DB.
- [x] Người dùng xác nhận chuyển sang Bước 6.

## Bước 6 — Post-session finalizer

- [x] Viết test coverage-complete và coverage-incomplete.
- [x] Diarization-only khi coverage đầy đủ.
- [x] Full pipeline fallback khi coverage thiếu hoặc realtime lỗi.
- [x] Phân loại `single`, `mixed`, `uncertain` bằng overlap.
- [x] Không chia text bằng tỷ lệ ký tự.
- [x] Giữ selective re-ASR tắt cho đến khi test chứng minh không mất text.
- [x] Publish `speaker_update`, `doc_delta`, `done/error`.
- [x] Tạo Document và commit trước `done`.
- [x] Chạy 54 backend regression tests; Ruff và `git diff --check` pass.
- [x] Người dùng xác nhận chuyển sang Bước 7.

## Bước 7 — Inference scheduling và resource safety

- [x] Viết test backpressure/priority không dùng model thật.
- [x] Một coordinator tuần tự cho model ASR dùng chung.
- [x] Partial latest-wins và được phép drop.
- [x] Confirmed/fallback không được drop.
- [x] Confirmed realtime dùng `skip_vad=True`, không chạy lại offline VAD.
- [x] Spool audio dài xuống file/storage.
- [x] Thêm metric latency, queue depth, ASR call count và fallback count.
- [x] Chạy 103 backend regression tests; Ruff, compile và diff check pass.
- [ ] Người dùng xác nhận chuyển sang Bước 8.

## Hotfix nghiệm thu realtime trước Bước 8

- [x] Xác thực JWT ở frame WebSocket đầu và gắn `Source.user_id`.
- [x] Lưu audio realtime thành WAV 16 kHz mono vào Storage.
- [x] Hiện trình phát media sau khi realtime hoàn tất.
- [x] Diarization-final refresh VAD toàn file thay vì chỉ dùng range streaming.
- [x] Không biến lỗi clustering thành Speaker 0 giả.
- [x] Chạy 108 backend tests, frontend lint/build và Ruff.

## Targeted re-ASR cho đoạn mixed/unknown

- [x] Bước 1: Xây dựng contract/thuật toán targeted re-ASR và unit test.
- [ ] Bước 2: Tích hợp final worker và cập nhật DB theo transaction.
- [ ] Bước 3: Đồng bộ transcript cuối về frontend.
- [ ] Bước 4: Kiểm thử tích hợp, đo lượng audio ASR lại và cập nhật tài liệu.

## Bước 8 — Nghiệm thu E2E

- [ ] Chạy backend test suite liên quan.
- [ ] Chạy frontend lint và build.
- [ ] Chạy live smoke test khi môi trường model cho phép.
- [ ] Đo first partial latency, confirmed latency, RTF và memory.
- [ ] Xác nhận phiên ngắn/dài/silence/noise/disconnect không treo.
- [ ] Xác nhận frontend luôn nhận `done` hoặc `error`.
- [ ] Cập nhật `walkthrough.md`.
- [ ] Review diff cuối và ghi các giới hạn còn lại.
