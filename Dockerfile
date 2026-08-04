# =========================
# Base image
# =========================
FROM python:3.12-slim

# Không tạo pyc và log ra stdout ngay
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# =========================
# System packages
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Copy source
# =========================
COPY pyproject.toml .
COPY README.md .
COPY meeting_config.yaml .
COPY meetasr ./meetasr

# =========================
# Install project
# =========================
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# =========================
# Expose API
# =========================
EXPOSE 8000

# =========================
# Start server
# =========================
CMD ["meetasr", "server", "--config", "meeting_config.yaml"]