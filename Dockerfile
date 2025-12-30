# ================================
# Kavir_RAG — Minimal CPU Docker
# ================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

# ---- system deps (minimal, CPU-only) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- python deps (single source of truth) ----
COPY requirements.base.txt /app/requirements.base.txt
COPY requirements.app.txt  /app/requirements.app.txt

RUN pip install --no-cache-dir -r /app/requirements.base.txt \
 && pip install --no-cache-dir -r /app/requirements.app.txt

# ---- project source ----
COPY src/ /app/src

# ---- data dirs (runtime-mounted or local) ----
RUN mkdir -p /app/data/chunks /app/data/index

# ---- default: interactive CLI container ----
CMD ["bash"]
