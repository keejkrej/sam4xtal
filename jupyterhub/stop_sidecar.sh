#!/usr/bin/env bash
# Stop a sidecar started by start_sidecar.sh
set -euo pipefail

_HUB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${_HUB_DIR}/env.sh"

if [ -f "${SAM4XTAL_PID_FILE}" ]; then
  pid="$(cat "${SAM4XTAL_PID_FILE}")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[sam4xtal-hub] killing pid ${pid}"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "${SAM4XTAL_PID_FILE}"
else
  echo "[sam4xtal-hub] no pid file at ${SAM4XTAL_PID_FILE}"
fi

# Best-effort: free the port if something is still listening
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi

echo "[sam4xtal-hub] stop requested"
