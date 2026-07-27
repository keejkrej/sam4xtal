#!/usr/bin/env bash
# OPTIONAL: try to start the Next.js UI next to the sidecar.
#
# On most JupyterHubs this is useless: the Hub does not reverse-proxy :3000
# (same issue as TensorBoard on this faculty Hub). The notebook workspace is
# the supported path. This script is here if your Hub has jupyter-server-proxy
# or you plan to SSH-tunnel :3000 yourself.
#
#   ./jupyterhub/start_sidecar.sh
#   ./jupyterhub/start_web_optional.sh

set -euo pipefail

_HUB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${_HUB_DIR}/env.sh"

export INFERENCE_URL="${INFERENCE_URL:-${SAM4XTAL_SIDECAR_URL}}"
export PORT_WEB="${PORT_WEB:-3000}"
export HOSTNAME="${HOSTNAME:-0.0.0.0}"

cd "${SAM4XTAL_ROOT}/web"

if [ ! -d node_modules ]; then
  if command -v pnpm >/dev/null 2>&1; then
    pnpm install --config.minimumReleaseAge=0 || pnpm install
  elif command -v npm >/dev/null 2>&1; then
    npm install
  else
    echo "[sam4xtal-hub] neither pnpm nor npm found — cannot start Next.js." >&2
    echo "  Use notebooks/setup.ipynb instead (Vite+ / Next path)." >&2
    exit 1
  fi
fi

echo "[sam4xtal-hub] INFERENCE_URL=${INFERENCE_URL}"
echo "[sam4xtal-hub] starting Next on ${HOSTNAME}:${PORT_WEB}"
echo
echo "If jupyter-server-proxy is enabled, try:"
echo "  https://<hub>/user/<you>/proxy/${PORT_WEB}/"
echo "Otherwise from your laptop (VPN):"
echo "  ssh -L ${PORT_WEB}:localhost:${PORT_WEB} <user>@<this-node-hostname>"
echo "  then open http://localhost:${PORT_WEB}"
echo
echo "This is unsupported. Prefer the notebook workspace."
echo

if command -v pnpm >/dev/null 2>&1; then
  exec env PORT="$PORT_WEB" pnpm dev --hostname "$HOSTNAME" --port "$PORT_WEB"
fi
exec env PORT="$PORT_WEB" npx next dev -H "$HOSTNAME" -p "$PORT_WEB"
