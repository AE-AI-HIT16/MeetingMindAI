# ============================================================
# STAGE 1: Builder
# Dùng để tải và biên dịch các thư viện Python (C++, Rust...)
# ============================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Cài đặt các công cụ cần thiết để biên dịch (chỉ tồn tại ở bước build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tạo Virtual Environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Cài đặt thư viện Python vào Virtual Environment
COPY pyproject.toml README.md ./
COPY meetasr ./meetasr

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir . --extra-index-url https://download.pytorch.org/whl/cpu

# ============================================================
# STAGE 2: Final Runtime
# Môi trường chạy siêu nhẹ, chỉ chứa file cần thiết
# ============================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Chỉ cài những công cụ BẮT BUỘC để chạy (như ffmpeg xử lý audio)
# Không cài git hay build-essential nữa -> Giảm hàng trăm MB
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy Virtual Environment đã được cài đặt sẵn từ bước Builder sang
COPY --from=builder /opt/venv /opt/venv

# Copy mã nguồn ứng dụng
COPY meeting_config.yaml .
COPY meetasr ./meetasr

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

CMD ["meetasr", "server", "--config", "meeting_config.yaml"]