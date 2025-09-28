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



docker stop rag-dev 2>/dev/null || true
docker run --rm -it \
  -p 8000:8000 \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/.cache/hf:/root/.cache/huggingface" \
  --name rag-dev \
  rag-local


docker stop rag-dev 2>/dev/null || true
docker run --rm -it \
  -p 8000:8000 \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/.cache/hf:/root/.cache/huggingface" \
  -v "$(pwd)/llama.cpp:/app/llama.cpp" \
  -e LLAMA_BIN="/app/llama.cpp/build/bin/llama-cli" \
  -e LLAMA_MODEL="/app/data/models/qwen2-7b-instruct-q4_k_m.gguf" \
  --name rag-dev \
  rag-local




docker stop rag-chat 2>/dev/null || true

docker run --rm -d --name rag-chat \
  -p 8000:8000 \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/.cache/hf:/root/.cache/huggingface" \
  -v "$(pwd)/llama.cpp:/app/llama.cpp" \
  -e LLAMA_BIN="/app/llama.cpp/build/bin/llama-cli" \
  -e LLAMA_MODEL="/app/data/models/qwen2-7b-instruct-q4_k_m.gguf" \
  -e LD_LIBRARY_PATH="/app/llama.cpp/build/bin" \
  rag-local
