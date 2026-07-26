#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -r requirements.txt
fi

export SAM3_BACKEND="${SAM3_BACKEND:-mock}"
export PORT="${PORT:-9001}"

exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
