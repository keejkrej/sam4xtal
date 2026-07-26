#!/usr/bin/env bash
#   ./run.sh            → real Hugging Face SAM (transformers + CUDA)
#   ./run.sh --mock     → flood-fill stub (no model weights)
#   ./run.sh --reload   → enable uvicorn auto-reload (dev only)
set -euo pipefail
cd "$(dirname "$0")"

MOCK=0
RELOAD=0
for arg in "$@"; do
  case "$arg" in
    --mock) MOCK=1 ;;
    --reload) RELOAD=1 ;;
    -h|--help)
      echo "Usage: $0 [--mock] [--reload]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--mock] [--reload]" >&2
      exit 1
      ;;
  esac
done

if [ ! -d .venv ]; then
  uv python install 3.12
  uv sync
fi

if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    key="${line%%=*}"
    val="${line#*=}"
    key="$(echo "$key" | xargs)"
    [ "$key" = "SAM3_BACKEND" ] && continue
    if [ -z "${!key+x}" ]; then
      export "$key=$val"
    fi
  done < .env
fi

if [ "$MOCK" -eq 1 ]; then
  export SAM3_BACKEND=mock
else
  export SAM3_BACKEND=transformers
fi

export PORT="${PORT:-9001}"

if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use. Stop the other sidecar first." >&2
  exit 1
fi

echo "SAM3_BACKEND=$SAM3_BACKEND PORT=$PORT"
args=(run uvicorn app.main:app --host 0.0.0.0 --port "$PORT")
if [ "$RELOAD" -eq 1 ]; then
  args+=(--reload)
fi
exec uv "${args[@]}"
