#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------
# 1) Check required model path
# ---------------------------------------------------------
if [ -z "${LLAMA_MODEL:-}" ]; then
  echo "ERROR: LLAMA_MODEL is not set" >&2
  exit 1
fi

echo "[entrypoint] Using model: $LLAMA_MODEL"

mkdir -p /app/logs

# ---------------------------------------------------------
# 2) Start llama-server in background
# ---------------------------------------------------------
echo "[entrypoint] Starting llama-server..."

# IMPORTANT: -t 1  → کاهش مصرف CPU روی مک 2015
/app/llama.cpp/build/bin/llama-server \
  --model "${LLAMA_MODEL}" \
  --host 0.0.0.0 \
  -c 768 \
  -t 1 \
  -b 128 \
  -ngl 0 \
  > /app/logs/llama-server.log 2>&1 &

LLAMA_PID=$!

# ---------------------------------------------------------
# 3) Health check loop
# ---------------------------------------------------------
echo "[entrypoint] Waiting for llama-server health..."
for i in {1..120}; do
  if curl -sf http://127.0.0.1:8080/health >/dev/null; then
    echo "[entrypoint] llama-server is healthy."
    break
  fi

  sleep 1

  # اگر پروسه مُرده باشد
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "ERROR: llama-server crashed. Check /app/logs/llama-server.log" >&2
    exit 1
  fi

  if [ "$i" -eq 120 ]; then
    echo "ERROR: llama-server health timeout (120s)" >&2
    exit 1
  fi
done

# ---------------------------------------------------------
# 4) Start FastAPI (uvicorn) as main process
# ---------------------------------------------------------
echo "[entrypoint] Starting Uvicorn..."
exec "$@"
