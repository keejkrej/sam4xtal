#!/usr/bin/env bash
# Start the SAM3 FastAPI sidecar inside a JupyterHub / interactive session.
#
#   ./jupyterhub/start_sidecar.sh          # real SAM (needs GPU + HF access)
#   ./jupyterhub/start_sidecar.sh --mock   # flood-fill stub, no weights
#   ./jupyterhub/start_sidecar.sh --fg     # run in foreground (logs to terminal)
#
# Then open notebooks/setup.ipynb (or start Next and use the printed node URL).

set -euo pipefail

_HUB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${_HUB_DIR}/env.sh" "$@"

FG=0
MOCK_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --fg|--foreground) FG=1 ;;
    --mock) MOCK_ARGS+=(--mock) ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
  esac
done

port_in_use() {
  local p="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :$p )" 2>/dev/null | grep -q ":$p"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1
  else
    # bash /dev/tcp probe
    (echo >/dev/tcp/127.0.0.1/"$p") >/dev/null 2>&1
  fi
}

health_ok() {
  python3 - "$SAM4XTAL_SIDECAR_URL" <<'PY' 2>/dev/null
import json, sys, urllib.request
url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=2) as r:
        data = json.loads(r.read().decode())
    print(json.dumps(data))
    sys.exit(0 if data.get("ok") else 1)
except Exception as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)
PY
}

if port_in_use "$PORT"; then
  if health_ok; then
    echo "[sam4xtal-hub] sidecar already healthy on port ${PORT}"
    health_ok || true
    exit 0
  fi
  echo "[sam4xtal-hub] port ${PORT} is in use but /health failed." >&2
  echo "  Stop the other process or set PORT=... and restart." >&2
  exit 1
fi

cd "${SAM4XTAL_ROOT}/sidecar"

if [ ! -d .venv ]; then
  if command -v uv >/dev/null 2>&1; then
    echo "[sam4xtal-hub] creating sidecar venv with uv…"
    uv python install 3.12 || true
    uv sync
  else
    echo "[sam4xtal-hub] uv not found — using python -m venv + pip (slower)."
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -U pip
    pip install -e .
  fi
fi

# Prefer venv binaries
if [ -x .venv/bin/uvicorn ]; then
  UVICORN=(.venv/bin/uvicorn)
elif command -v uv >/dev/null 2>&1; then
  UVICORN=(uv run uvicorn)
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
  UVICORN=(uvicorn)
fi

CMD=("${UVICORN[@]}" app.main:app --host "$HOST" --port "$PORT")

echo "[sam4xtal-hub] starting: ${CMD[*]}"
echo "[sam4xtal-hub] log: ${SAM4XTAL_LOG_FILE}"

if [ "$FG" -eq 1 ]; then
  exec "${CMD[@]}"
fi

nohup "${CMD[@]}" >"${SAM4XTAL_LOG_FILE}" 2>&1 &
echo $! >"${SAM4XTAL_PID_FILE}"
echo "[sam4xtal-hub] pid=$(cat "${SAM4XTAL_PID_FILE}")"

# Wait for health (weights may still be loading → 503 is ok if process is up)
for i in $(seq 1 60); do
  if health_ok >/dev/null 2>&1; then
    echo "[sam4xtal-hub] /health ok"
    health_ok || true
    echo
    echo "Open notebooks/setup.ipynb (or point Next at ${SAM4XTAL_SIDECAR_URL})."
    echo "Sidecar URL: ${SAM4XTAL_SIDECAR_URL}"
    exit 0
  fi
  if [ -f "${SAM4XTAL_PID_FILE}" ] && ! kill -0 "$(cat "${SAM4XTAL_PID_FILE}")" 2>/dev/null; then
    echo "[sam4xtal-hub] process died — last log lines:" >&2
    tail -n 40 "${SAM4XTAL_LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "[sam4xtal-hub] process is up but /health not ready yet (model download?)."
echo "  tail -f ${SAM4XTAL_LOG_FILE}"
echo "  Open the notebook anyway — it waits for ready."
exit 0
