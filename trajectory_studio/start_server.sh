#!/bin/bash
set -euo pipefail

studio_app="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${GENIE_STUDIO_ENV:-}" ]; then
    source "$GENIE_STUDIO_ENV"
fi

studio_runtime="${GENIE_STUDIO_RUNTIME:-$(dirname -- "$studio_app")}"
studio_runtime="$(cd -- "$studio_runtime" && pwd)"
studio_state="${GENIE_STUDIO_STATE:-$studio_runtime/.platform-state/trajectory-studio}"
studio_exports="${GENIE_STUDIO_EXPORTS:-/data/by/genie_sim/exports}"
studio_reader="${GENIE_STUDIO_MEDIA_PYTHON:-/data/by/genie_sim/lerobot-reader-0cf8648/bin/python}"
studio_port="${GENIE_STUDIO_PORT:-18765}"
export GENIE_STUDIO_PORT="$studio_port"

studio_is_ready() {
    python3 -c 'import json, os, urllib.request; port=os.environ["GENIE_STUDIO_PORT"]; x=json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/studio/health",timeout=2)); assert x["service"]=="nexus-trajectory-studio" and x["worker_ready"]' 2>/dev/null
}

if studio_is_ready; then
    exit 0
fi
if ss -H -lnt "sport = :$studio_port" | grep -q .; then
    echo "The studio port is already in use." >&2
    exit 1
fi
export GENIE_PROJECT_ROOT="$studio_runtime"
export GENIE_COLLECTION_SLOT="${GENIE_COLLECTION_SLOT:-gpu1-extra}"
export PYTHONPATH="$studio_runtime${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$studio_runtime/.venv/bin:$PATH"
cd "$studio_runtime"
mkdir -p "$studio_state"
if [ ! -f "$studio_state/posters/0.jpg" ]; then
    "$studio_reader" "$studio_app/server.py" --runtime "$studio_runtime" --state "$studio_state" \
      --exports "$studio_exports" --media-python "$studio_reader" --posters-only >>"$studio_state/studio.log" 2>&1
fi
nohup "$studio_runtime/.venv/bin/python" -u "$studio_app/server.py" \
    --runtime "$studio_runtime" --state "$studio_state" --exports "$studio_exports" \
    --media-python "$studio_reader" --port "$studio_port" >>"$studio_state/studio.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$studio_state/server.pid"
for i in $(seq 1 30); do
    if studio_is_ready; then
        exit 0
    fi
    sleep 1
done
echo "The studio did not start. Check $studio_state/studio.log." >&2
exit 1
