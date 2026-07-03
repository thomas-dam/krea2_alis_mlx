#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HOST="${KREA2_API_HOST:-127.0.0.1}"
PORT="${KREA2_API_PORT:-7861}"

exec .venv/bin/python -m uvicorn api:app --host "$HOST" --port "$PORT"
