#!/usr/bin/env bash
# Launch the sketch2spice web app.
#
#   ./run.sh                                   # defaults
#   ./run.sh --vision-url http://HOST:PORT/v1  # point at your own model endpoint
#   ./run.sh --server.port 9000                # other args pass through to streamlit
#
# The vision endpoint can also be set via the LOCAL_VISION_URL env var. Either way,
# include the trailing "/v1" (it's an OpenAI-compatible base URL), e.g.
# http://127.0.0.1:8080/v1.
#
# Requires `uv` (https://docs.astral.sh/uv/) and, for simulation, `ngspice`.
set -euo pipefail

cd "$(dirname "$0")"

# Vision endpoint: --vision-url flag > LOCAL_VISION_URL env > default.
VISION_URL="${LOCAL_VISION_URL:-http://127.0.0.1:8080/v1}"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --vision-url) VISION_URL="$2"; shift 2 ;;
    --vision-url=*) VISION_URL="${1#*=}"; shift ;;
    *) args+=("$1"); shift ;;
  esac
done
export LOCAL_VISION_URL="$VISION_URL"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is not installed. See https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v ngspice >/dev/null 2>&1; then
  echo "warning: 'ngspice' not found — the app will load but simulation will fail." >&2
  echo "         install it with: brew install ngspice" >&2
fi

case "$VISION_URL" in
  */v1|*/v1/) ;;
  *) echo "hint: LOCAL_VISION_URL is '$VISION_URL' — it usually needs a trailing '/v1'." >&2 ;;
esac

# Sync the virtualenv from pyproject/uv.lock (creates .venv on first run).
uv sync --quiet

echo "Vision endpoint: $LOCAL_VISION_URL"
echo "Starting sketch2spice — open the URL below in your browser (Ctrl-C to stop)."
# ${args[@]+...} guard keeps set -u happy with an empty array on bash 3.2 (macOS).
exec uv run streamlit run app.py ${args[@]+"${args[@]}"}
