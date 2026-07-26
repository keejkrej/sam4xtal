#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  uv python install 3.12
  uv sync
fi

export SAM3_BACKEND="${SAM3_BACKEND:-mock}"
export PORT="${PORT:-9001}"

exec uv run uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
