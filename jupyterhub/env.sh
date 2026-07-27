#!/usr/bin/env bash
# Shared env for JupyterHub / interactive cluster sessions.
# Sourced by start_sidecar.sh; safe to source from a notebook terminal too.
#
# Usage:
#   source jupyterhub/env.sh
#   source jupyterhub/env.sh --mock

set -euo pipefail

_HUB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SAM4XTAL_ROOT="$(cd "${_HUB_DIR}/.." && pwd)"

# Prefer Lustre-ish project storage for weights / venvs (home quotas kill Hub sessions).
_data_root=""
if [ -n "${SCRATCH:-}" ] && [ -d "${SCRATCH}" ]; then
  _data_root="${SCRATCH}"
elif [ -n "${WORK:-}" ] && [ -d "${WORK}" ]; then
  _data_root="${WORK}"
else
  _data_root="${HOME}"
fi

export SAM4XTAL_DATA="${SAM4XTAL_DATA:-${_data_root}/sam4xtal}"
export HF_HOME="${HF_HOME:-${SAM4XTAL_DATA}/hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SAM4XTAL_DATA}/uv-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SAM4XTAL_DATA}/xdg-cache}"

export PORT="${PORT:-9001}"
export HOST="${HOST:-127.0.0.1}"
export SAM4XTAL_SIDECAR_URL="${SAM4XTAL_SIDECAR_URL:-http://${HOST}:${PORT}}"
export SAM4XTAL_PID_FILE="${SAM4XTAL_PID_FILE:-${SAM4XTAL_DATA}/sidecar.pid}"
export SAM4XTAL_LOG_FILE="${SAM4XTAL_LOG_FILE:-${SAM4XTAL_DATA}/sidecar.log}"

# Optional: load HF_TOKEN / overrides from sidecar/.env without clobbering shell.
if [ -f "${SAM4XTAL_ROOT}/sidecar/.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    key="${line%%=*}"
    val="${line#*=}"
    key="$(echo "$key" | xargs)"
    [ -z "$key" ] && continue
    if [ -z "${!key+x}" ]; then
      export "$key=$val"
    fi
  done < "${SAM4XTAL_ROOT}/sidecar/.env"
fi

_MOCK=0
for arg in "$@"; do
  case "$arg" in
    --mock) _MOCK=1 ;;
  esac
done

if [ "$_MOCK" -eq 1 ]; then
  export SAM3_BACKEND=mock
else
  export SAM3_BACKEND="${SAM3_BACKEND:-transformers}"
fi

mkdir -p "$SAM4XTAL_DATA" "$HF_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME"

echo "[sam4xtal-hub] root=${SAM4XTAL_ROOT}"
echo "[sam4xtal-hub] data=${SAM4XTAL_DATA}"
echo "[sam4xtal-hub] HF_HOME=${HF_HOME}"
echo "[sam4xtal-hub] sidecar=${SAM4XTAL_SIDECAR_URL} backend=${SAM3_BACKEND}"
