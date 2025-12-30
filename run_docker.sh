#!/usr/bin/env bash
set -e

IMAGE_NAME=kavir_rag
DATA_DIR="$(pwd)/data"

docker run --rm -it \
  -v "${DATA_DIR}:/app/data" \
  "${IMAGE_NAME}" \
  bash
