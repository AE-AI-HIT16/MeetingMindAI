# Lệnh chạy chương trình MeetASR

> Tổng hợp mọi lệnh để chạy dự án — backend, frontend, CLI, hạ tầng (Postgres/MinIO),
> và biến môi trường. Tất cả lệnh dưới đây đối chiếu với code thật
> (`meetasr/bin/cli.py`, `frontend-next/package.json`, `docker-compose.yml`), không phải trí nhớ.
>
> Yêu cầu: Python ≥ 3.10, Node ≥ 18 (cho frontend), Docker (cho Postgres/MinIO).

---

## 0. Cài đặt lần đầu

```powershell
# Backend — cài package meetasr ở chế độ editable
cd E:\MeetingMindAI
pip install -e ".[dev]"          # gồm cả công cụ dev; bỏ [dev] nếu chỉ chạy

# Frontend — cài dependency
cd E:\MeetingMindAI\frontend-next
pnpm install                     # hoặc: npm install
```

Model ASR / punctuation / speaker (theo `meeting_config.yaml`: zipformer-vi, vibert-capu, cam++)
**tự tải** từ HuggingFace/ModelScope ở lần chạy đầu tiên — không cần tải tay. Lần đầu mất vài phút.

> **VAD mặc định là Silero** (`meetasr/models/vad/silero_vad.py`) — file cục bộ
> `models/silero-vad/silero_vad.onnx` (~640KB), KHÔNG tự tải từ hub. Nếu thiếu, tải từ
> sherpa-onnx releases (`silero_vad.onnx`) đặt vào đúng đường dẫn đó. Lý do chọn Silero thay
> FSMN-VAD: đo trên audio Việt cho thấy FSMN tag 100% là speech (không lọc nhạc/lặng) — xem
> [18_chuan_hoa_audio_mic.md] và [15_phase3_plan.md] mục Rủi ro. Muốn quay lại FSMN: đổi
> `vad.model: fsmn-vad`, `hub: ms` trong config.

---

## 1. Hạ tầng — Postgres + MinIO (Docker)

Backend cần Postgres (lưu metadata) và MinIO (lưu file media). Bật bằng compose:

```powershell
cd E:\MeetingMindAI
docker compose up -d             # chạy nền postgres_db (cổng 5432) + minio_storage (9000/9001)
docker compose ps                # kiểm tra trạng thái
docker compose logs -f           # xem log
docker compose down              # tắt (giữ dữ liệu trong volume)
docker compose down -v           # tắt + XOÁ dữ liệu (cẩn thận)
```

- MinIO console: http://localhost:9001 (tài khoản xem trong `docker-compose.yml`).
- Dữ liệu nằm trong volume `postgres_data` / `minio_data`, không mất khi `down` (không có `-v`).

---

## 2. Backend — REST API server

Lệnh chính (đây là lệnh bạn đã dùng để test):

```powershell
cd E:\MeetingMindAI
meetasr server --config meeting_config.yaml --port 8000 --reload
```

Các cờ của `meetasr server`:

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--host` | `0.0.0.0` | Địa chỉ lắng nghe |
| `--port` | `8000` | Cổng |
| `--config` | `meeting_config.yaml` | Đường dẫn config (ASR/VAD/punc/LLM) |
| `--reload` | tắt | Tự khởi động lại khi sửa code (chỉ dùng khi phát triển) |

Sau khi chạy:
- API gốc: http://localhost:8000
- Tài liệu API (Swagger): http://localhost:8000/docs
- Kiểm tra hiệu năng: `GET http://localhost:8000/v1/metrics` (số job, RTF trung bình, độ dài hàng đợi — thêm ở Phase 3)
- WebSocket mic sống: `ws://localhost:8000/v1/live` (Phase 3)

> **Lưu ý:** server đọc config qua biến môi trường `MEETASR_CONFIG` nếu `--config` không truyền.
> Nếu tài liệu ra rỗng khi dùng tính năng "Tóm tắt", kiểm tra mục `llm.api_key` trong
> `meeting_config.yaml` — "Toàn văn" thì không cần LLM.

---

## 3. Frontend — giao diện web (Next.js)

```powershell
cd E:\MeetingMindAI\frontend-next
pnpm dev                         # chế độ phát triển — http://localhost:3000
```

Các script khác (trong `package.json`):

```powershell
pnpm build                       # build production
pnpm start                       # chạy bản đã build
pnpm lint                        # kiểm tra lint (eslint)
```

Frontend proxy mọi request `/v1/*` sang backend. Mặc định trỏ tới `http://127.0.0.1:8000`;
đổi bằng biến môi trường trước khi chạy nếu backend ở nơi khác:

```powershell
$env:MEETASR_API = "http://127.0.0.1:8000"; pnpm dev
```

**Chạy đủ bộ để dùng web:** cần 3 thứ bật cùng lúc — (1) `docker compose up -d`,
(2) `meetasr server ...`, (3) `pnpm dev` — rồi mở http://localhost:3000.

---

## 4. CLI — chạy trực tiếp không cần server

### 4.1. `transcribe` — chuyển audio thành văn bản

```powershell
meetasr transcribe audio.mp4
meetasr transcribe a.wav b.wav --language vi --output-format srt --output-dir ./out
meetasr transcribe hop.m4a --config meeting_config.yaml --device cuda:0
```

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `audio` (bắt buộc) | — | Một hoặc nhiều file audio/video |
| `--model` | `sensevoice-small` | Model ASR |
| `--device` | `cpu` | `cpu` \| `cuda:0` |
| `--hub` | `ms` | Nguồn model: `ms` (ModelScope) \| `hf` (HuggingFace) |
| `--language` | `auto` | `auto` \| `vi` \| `zh` \| `en` |
| `--config` | — | Đường dẫn `meeting_config.yaml` |
| `--no-punc` | tắt | Bỏ qua bước thêm dấu câu |
| `-f, --output-format` | `text` | `text` \| `json` \| `srt` |
| `-o, --output-dir` | — | Thư mục lưu file SRT/JSON |

### 4.2. `summarize` — phiên âm + tóm tắt bằng LLM

```powershell
meetasr summarize hop.mp4 --config meeting_config.yaml
meetasr summarize hop.mp4 --config meeting_config.yaml --language vi -f markdown -o ./reports
```

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `audio` (bắt buộc) | — | File audio/video |
| `--config` (bắt buộc) | — | Config có mục `llm` (cần API key thật) |
| `--language` | `vi` | Ngôn ngữ đầu ra: `vi` \| `en` |
| `-f, --output-format` | `json` | `json` \| `markdown` |
| `-o, --output-dir` | — | Thư mục lưu báo cáo |

### 4.3. Cờ chung

```powershell
meetasr --log-level INFO transcribe audio.mp4    # đặt mức log (mặc định WARNING)
meetasr --help                                    # xem trợ giúp
meetasr transcribe --help                         # trợ giúp cho lệnh con
```

---

## 5. Biến môi trường

| Biến | Mặc định | Tác dụng |
|---|---|---|
| `MEETASR_CONFIG` | `meeting_config.yaml` | Đường dẫn config khi không truyền `--config` |
| `MEETASR_API` | `http://127.0.0.1:8000` | (Frontend) URL backend để proxy |
| `MEETASR_MAX_LIVE` | `2` | Số phiên mic sống đồng thời tối đa (Phase 3) |
| `MEETASR_SEM_VAD` | `2` | Giới hạn đồng thời model VAD (Phase 3) |
| `MEETASR_SEM_ASR` | `1` | Giới hạn đồng thời model ASR trên GPU (Phase 3) |
| `MEETASR_SEM_PUNC` | `2` | Giới hạn đồng thời model dấu câu (Phase 3) |
| `MEETASR_SEM_SPK` | `2` | Giới hạn đồng thời model speaker (Phase 3) |

Đặt biến môi trường trong PowerShell:

```powershell
$env:MEETASR_CONFIG = "E:\MeetingMindAI\meeting_config.yaml"
$env:MEETASR_MAX_LIVE = "1"        # máy yếu: chỉ cho 1 phiên mic sống
meetasr server --port 8000
```

---

## 6. Quy trình nhanh (bật cả stack để dùng web)

Mở 3 cửa sổ terminal:

```powershell
# Terminal 1 — hạ tầng
cd E:\MeetingMindAI; docker compose up -d

# Terminal 2 — backend
cd E:\MeetingMindAI; meetasr server --config meeting_config.yaml --port 8000 --reload

# Terminal 3 — frontend
cd E:\MeetingMindAI\frontend-next; pnpm dev
```

Mở http://localhost:3000 → tải file lên → xem tài liệu hình thành → xuất PDF/DOCX.
