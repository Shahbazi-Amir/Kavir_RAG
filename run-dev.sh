#!/bin/bash
docker run -it --rm \
  -v "$(pwd)/.cache/hf:/root/.cache/huggingface" \
  -v "$(pwd)/data:/app/data" \
  rag-local /bin/bash



docker stop rag-dev 2>/dev/null || true
docker run --rm -it \
  -p 8000:8000 \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/.cache/hf:/root/.cache/huggingface" \
  --name rag-dev \
  rag-local
