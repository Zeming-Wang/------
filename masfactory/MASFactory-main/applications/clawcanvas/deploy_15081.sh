#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${CLAWCANVAS_HOST:-0.0.0.0}"
PORT="${CLAWCANVAS_PORT:-15081}"
WORKERS="${CLAWCANVAS_WORKERS:-2}"
TIMEOUT="${CLAWCANVAS_TIMEOUT:-120}"
PYTHON_BIN="${CLAWCANVAS_PYTHON:-$APP_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python)"
fi

cd "$APP_DIR/frontend"
npm ci
npm run build

cd "$APP_DIR/backend"
if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install -r requirements.txt
elif command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON_BIN" -r requirements.txt
else
  echo "warning: pip is not available for $PYTHON_BIN; skipping dependency installation" >&2
fi

"$PYTHON_BIN" -m gunicorn --version >/dev/null 2>&1 || {
  echo "error: gunicorn is not installed for $PYTHON_BIN. Install backend requirements, then rerun this script." >&2
  exit 1
}
exec "$PYTHON_BIN" -m gunicorn "clawcanvas_backend.app:create_app()" \
  --bind "${HOST}:${PORT}" \
  --workers "${WORKERS}" \
  --timeout "${TIMEOUT}"
