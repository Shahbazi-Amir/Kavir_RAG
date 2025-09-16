#!/bin/bash
docker run -it --rm \
  -v "$(pwd)/.cache/hf:/root/.cache/huggingface" \
  -v "$(pwd)/data:/app/data" \
  rag-local /bin/bash
