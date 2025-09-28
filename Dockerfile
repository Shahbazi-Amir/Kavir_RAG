# Use official Python 3.11 slim image
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd tesseract-ocr-fas \
    poppler-utils \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- copy deps separately for better caching ---
COPY requirements.base.txt /app/requirements.base.txt
COPY requirements.app.txt  /app/requirements.app.txt

# --- install base deps (big, stable) ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /app/requirements.base.txt

# --- install app deps (smaller, may change) ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /app/requirements.app.txt

# --- source last ---
COPY src/ /app/src

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
