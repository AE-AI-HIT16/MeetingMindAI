# MeetASR / MeetingMindAI

Hệ thống nhận dạng tiếng nói cuộc họp, phân tách người nói và tạo tài liệu từ
audio/video. Cấu hình mặc định của nhánh integration sử dụng:

```text
Silero VAD → CAM++ → speaker turns → Zipformer tiếng Việt → ViBERT punctuation
```

Phần xử lý file tải lên là pipeline offline. WebSocket realtime có vòng đời VAD
riêng và không bị thay đổi bởi cấu hình này.

## Yêu cầu

- Linux, macOS hoặc Windows/WSL.
- Python 3.10 trở lên; Python 3.10 được khuyến nghị cho các thư viện ML.
- FFmpeg.
- Node.js 20 trở lên và pnpm 10 để chạy frontend.
- Kết nối mạng trong lần chạy đầu để tải model từ Hugging Face và ModelScope.

Cài FFmpeg trên Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Hoặc trong Conda:

```bash
conda install -c conda-forge ffmpeg
```

## 1. Cài backend

Từ thư mục gốc của repository:

```bash
conda create -n meetasr python=3.10 -y
conda activate meetasr
pip install -e ".[dev,export]"
```

Nếu không dùng Conda:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,export]"
```

`requirements.txt` là lệnh cài tiện lợi cho người cần cả PostgreSQL và S3/MinIO:

```bash
pip install -r requirements.txt
```

## 2. Tạo cấu hình local

```bash
cp .env.example .env
cp meeting_config.example.yaml meeting_config.yaml
```

Điền `GROQ_API_KEY` trong `.env` để tạo bản tóm tắt. Có thể đặt key trực tiếp
trong `meeting_config.yaml` trên máy cá nhân vì file này được Git ignore, nhưng
không được đặt key trong `meeting_config.example.yaml` hoặc `.env.example`.

Mặc định hệ thống dùng SQLite và lưu media tại `data/media`, vì vậy không cần
Docker để phát triển local. `HF_TOKEN` là tùy chọn, dùng để tránh giới hạn tải
ẩn danh của Hugging Face.

Lần khởi động đầu tiên sẽ tải Zipformer, CAM++ và ViBERT. Những lần sau model
được dùng từ cache.

## 3. Chạy backend

```bash
conda activate meetasr
meetasr server --config meeting_config.yaml --host 127.0.0.1 --port 8000
```

Kiểm tra backend ở terminal khác:

```bash
curl http://127.0.0.1:8000/v1/health
```

API docs: <http://127.0.0.1:8000/docs>

Chạy trực tiếp một file mà không cần frontend:

```bash
meetasr transcribe input.wav \
  --config meeting_config.yaml \
  --language vi \
  --output-format json \
  --output-dir tests/output
```

## 4. Chạy frontend

Mở terminal thứ hai:

```bash
cd frontend-next
corepack enable
pnpm install --frozen-lockfile
cp .env.example .env.local
pnpm dev
```

Mở <http://localhost:3000>. Có thể chọn chế độ khách mà không cần OAuth.

Muốn đăng nhập Google/GitHub, điền các biến OAuth trong
`frontend-next/.env.local`. Trong flow local hiện tại, đặt `NEXTAUTH_SECRET`
giống `JWT_SECRET` của file `.env` backend.

Frontend gửi upload trực tiếp tới FastAPI và dùng WebSocket để theo dõi job.
Các origin development `localhost` và `127.0.0.1` đều đã được cho phép.

## PostgreSQL và MinIO tùy chọn

Cài driver rồi khởi động dịch vụ:

```bash
pip install -e ".[postgres,s3]"
docker compose --env-file .env up -d postgres_db minio_storage
```

Sau đó sửa `.env`:

```dotenv
DATABASE_URL=postgresql://admin:password123@localhost:5432/meetasr_db
STORAGE_BACKEND=s3
MINIO_ENDPOINT=http://localhost:9000
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=password123
MINIO_BUCKET=meetasr-audio
```

SQLite và local storage vẫn là lựa chọn đơn giản nhất khi chỉ test end-to-end
trên một máy.

## Cấu hình Zipformer

Cấu hình mặc định nằm trong `meeting_config.example.yaml`; preset không có LLM
nằm trong `configs/models/zipformer_vi.yaml`.

```yaml
asr:
  model: zipformer-vi
  hub: hf
  device: cpu
  model_variant: fp32
  encoder: encoder-epoch-20-avg-10.onnx
  decoder: decoder-epoch-20-avg-10.onnx
  joiner: joiner-epoch-20-avg-10.onnx
```

- `fp32`: ưu tiên chất lượng và dùng để benchmark.
- `int8`: nhẹ và nhanh hơn trên CPU. Khi đổi sang `int8`, bỏ ba tên file FP32
  hoặc thay bằng đúng tên file `.int8.onnx` trong snapshot.

Chạy preset Zipformer không tạo tóm tắt:

```bash
meetasr server --config configs/models/zipformer_vi.yaml --port 8000
```

Model `hynt/Zipformer-30M-RNNT-6000h` sử dụng giấy phép
CC-BY-NC-ND-4.0. Cần xem lại giới hạn phi thương mại trước khi dùng ngoài mục
đích nghiên cứu hoặc đánh giá.

## Test và kiểm tra chất lượng

Backend:

```bash
pytest tests/ -q
```

Các test phải tải model hoặc gọi dịch vụ thật bị tắt mặc định. Bật khi cần:

```bash
MEETASR_RUN_LIVE_TESTS=1 pytest tests/test_real_audio.py -v
MEETASR_RUN_LIVE_TESTS=1 pytest tests/test_ollama_client_llama3.py -v
```

Frontend:

```bash
cd frontend-next
pnpm lint
pnpm build
```

## Các model chính

| Model | Key | Vai trò |
|---|---|---|
<<<<<<< HEAD
| SenseVoiceSmall | `sensevoice-small` | ASR (vi/zh/en/ja/ko) |
| Paraformer-zh | `paraformer-zh` | ASR (zh/en, fastest) |
| Qwen3-ASR 0.6B | `qwen3-asr` | Offline multilingual ASR, including Vietnamese |
| Silero VAD | `silero-vad` | Default VAD for uploaded/offline audio |
| FSMN-VAD | `fsmn-vad` | Optional fallback VAD |
| CT-Punc | `ct-punc` | Punctuation Restoration |
| CAM++ | `cam++` | Speaker Diarization |
=======
| Zipformer 30M RNNT | `zipformer-vi` | STT tiếng Việt mặc định, tối ưu CPU |
| Faster-Whisper | `faster-whisper` | STT đa ngôn ngữ tùy chọn |
| SenseVoiceSmall | `sensevoice-small` | STT đa ngôn ngữ tùy chọn |
| Silero VAD | `silero-vad` | Phát hiện vùng có tiếng nói cho file upload |
| CAM++ | `cam++` | Embedding và phân cụm người nói |
| ViBERT-CaPu | `vibert-capu` | Viết hoa và khôi phục dấu câu tiếng Việt |
>>>>>>> anhtu/integration/phase2-end-to-end

## Cấu trúc chính

```text
meetasr/                 Backend, pipeline và model adapters
frontend-next/           Giao diện Next.js
configs/models/          Preset dùng để chạy/benchmark từng model
meeting_config.example.yaml
tests/                   Unit, integration và live tests
docker-compose.yml       PostgreSQL và MinIO tùy chọn
```
