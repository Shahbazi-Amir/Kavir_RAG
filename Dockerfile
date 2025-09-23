# Use official Python 3.11 slim image
FROM python:3.11-slim

# --- Environment for reliable, quiet pip ---
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# --- System deps: Tesseract (fa, eng, osd) + Poppler for pdf2image + build essentials if wheels missing ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd tesseract-ocr-fas \
    poppler-utils \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Copy only requirements first to maximize layer cache ---
COPY requirements.txt /app/requirements.txt

# --- Install Python deps with BuildKit cache (enable with DOCKER_BUILDKIT=1) ---
# This keeps pip cache between builds so dependency layer is fast if requirements.txt unchanged
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /app/requirements.txt

# --- Copy source last so changes here don't invalidate deps cache ---
COPY src/ /app/src

# Expose API port
EXPOSE 8000

# Default command for dev; we'll still run via docker run
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
