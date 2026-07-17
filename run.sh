#!/usr/bin/env bash
# Launch the sketch2spice web app.
#
#   ./run.sh                # start on the default port
#   ./run.sh --server.port 9000   # extra args pass through to streamlit
#
# Requires `uv` (https://docs.astral.sh/uv/) and, for simulation, `ngspice`.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is not installed. See https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v ngspice >/dev/null 2>&1; then
  echo "warning: 'ngspice' not found — the app will load but simulation will fail." >&2
  echo "         install it with: brew install ngspice" >&2
fi

# Sync the virtualenv from pyproject/uv.lock (creates .venv on first run).
uv sync --quiet

echo "Starting sketch2spice — open the URL below in your browser (Ctrl-C to stop)."
exec uv run streamlit run app.py "$@"
