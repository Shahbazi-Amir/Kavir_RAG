#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Kavir_RAG minimal entrypoint (CLI mode)
# =========================================

echo "[entrypoint] Kavir_RAG container started"
echo "[entrypoint] Working directory: $(pwd)"

# ensure expected dirs exist
mkdir -p /app/data/chunks /app/data/index

# if a command is provided, run it
# otherwise drop into shell
if [ "$#" -gt 0 ]; then
  exec "$@"
else
  exec bash
fi
