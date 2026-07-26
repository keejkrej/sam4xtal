#!/usr/bin/env bash
# Register sam4xtal as a systemd --user service (Docker Compose).
#
# Usage:
#   ./scripts/install-user-service.sh           # install + enable (does not start)
#   ./scripts/install-user-service.sh --start   # install + enable + start (--no-recreate)
#   ./scripts/install-user-service.sh --disable # disable + remove unit
#   ./scripts/install-user-service.sh --status  # show unit + compose status
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_NAME="sam4xtal.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_PATH="$UNIT_DIR/$UNIT_NAME"
TEMPLATE="$REPO_ROOT/deploy/sam4xtal.service.in"

START=0
DISABLE=0
STATUS=0

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

for arg in "$@"; do
  case "$arg" in
    --start) START=1 ;;
    --disable) DISABLE=1 ;;
    --status) STATUS=1 ;;
    -h|--help) usage 0 ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage 1
      ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: required command not found: $1" >&2
    exit 1
  }
}

if [ "$STATUS" -eq 1 ]; then
  systemctl --user status "$UNIT_NAME" --no-pager 2>&1 || true
  echo
  if [ -f "$REPO_ROOT/docker-compose.yml" ]; then
    (cd "$REPO_ROOT" && docker compose ps) 2>&1 || true
  fi
  exit 0
fi

need_cmd systemctl
need_cmd docker
docker compose version >/dev/null

if [ "$DISABLE" -eq 1 ]; then
  systemctl --user disable --now "$UNIT_NAME" 2>/dev/null || true
  rm -f "$UNIT_PATH"
  systemctl --user daemon-reload
  echo "Disabled and removed $UNIT_PATH"
  exit 0
fi

if [ ! -f "$TEMPLATE" ]; then
  echo "error: missing template: $TEMPLATE" >&2
  exit 1
fi

if [ ! -f "$REPO_ROOT/docker-compose.yml" ]; then
  echo "error: missing docker-compose.yml in $REPO_ROOT" >&2
  exit 1
fi

mkdir -p "$UNIT_DIR"

# shellcheck disable=SC2016
sed \
  -e "s|@REPO_ROOT@|${REPO_ROOT}|g" \
  -e "s|@HOME@|${HOME}|g" \
  "$TEMPLATE" >"$UNIT_PATH"

systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"

# Linger so the user service can start at boot without an interactive login.
if command -v loginctl >/dev/null 2>&1; then
  if ! loginctl show-user "$USER" -p Linger 2>/dev/null | grep -qx 'Linger=yes'; then
    if loginctl enable-linger "$USER" 2>/dev/null; then
      echo "Enabled linger for $USER (user services start at boot)."
    else
      echo "note: could not enable linger (optional). Run: loginctl enable-linger $USER" >&2
    fi
  fi
fi

echo "Installed $UNIT_PATH"
echo "Enabled $UNIT_NAME"

if [ "$START" -eq 1 ]; then
  systemctl --user start "$UNIT_NAME"
  echo "Started $UNIT_NAME (docker compose up -d --no-recreate)"
else
  echo "Not started (running containers left alone). Start later with:"
  echo "  systemctl --user start $UNIT_NAME"
  echo "  # or: $0 --start"
fi

echo
echo "Useful commands:"
echo "  systemctl --user status $UNIT_NAME"
echo "  systemctl --user stop $UNIT_NAME"
echo "  $0 --status"
