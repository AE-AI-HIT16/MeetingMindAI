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
    pip install --no-cache-dir . --extra-index-url https://download.pytorch.org/whl/cpu

# =========================
# Expose API
# =========================
EXPOSE 8000

# =========================
# Health check
# =========================
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# =========================
# Start server
# =========================
CMD ["meetasr", "server", "--config", "meeting_config.yaml"]