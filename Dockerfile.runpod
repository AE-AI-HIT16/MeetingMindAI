# ============================================================
# GPU Model Server Image for RunPod
# Uses official PyTorch runtime base image with CUDA 12.1 pre-installed
# ============================================================
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001

WORKDIR /app

# Install system dependencies (ffmpeg for audio processing, git, curl, build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata & source
COPY pyproject.toml README.md ./
COPY meeting_config.yaml config.yaml
COPY meetasr ./meetasr

# Install MeetASR dependencies without re-downloading PyTorch/CUDA wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .[gpu,vad-test] && \
    find /opt/conda -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "uvicorn", "meetasr.runpod.app:app", "--host", "0.0.0.0", "--port", "8001"]

