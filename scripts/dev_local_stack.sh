#!/usr/bin/env bash
# Run the full pipeline on this machine: RunPod worker simulator (local GPU),
# backend and frontend. Open http://localhost:3000 and log in as guest.
#
#   scripts/dev_local_stack.sh            # start, Ctrl+C stops everything
#   FULL=1 scripts/dev_local_stack.sh     # + overlap separation (slower, more accurate)
#
# Logs: data/local_stack/{worker,backend,frontend}.log
# Data: data/local_stack/local_test.db and data/local_stack/media (not the .env database).
# Needs once: pip install runpod einops "rotary-embedding-torch>=0.8.3,<0.9"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${MEETASR_PYTHON:-$HOME/miniconda3/envs/meetasr/bin/python}"
LOGS="$ROOT/data/local_stack"
mkdir -p "$LOGS/media"
cd "$ROOT"

pids=()
cleanup() {
  trap - EXIT INT TERM
  echo "Stopping..."
  # The whole process group: npx/uvicorn spawn children of their own.
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for() {  # wait_for <name> <log> <pattern>
  echo -n "Waiting for $1"
  until grep -qE "$3" "$2" 2>/dev/null; do
    if grep -qE "Traceback|Error:|EADDRINUSE" "$2" 2>/dev/null; then
      echo; echo "$1 failed — see $2"; tail -20 "$2"; exit 1
    fi
    echo -n "."; sleep 2
  done
  echo " ready"
}

CONFIG="configs/runpod_gpu.yaml"
if [ "${FULL:-0}" = "1" ]; then
  # Full pipeline: overlap-aware turns + MossFormer2 separation (slower).
  CONFIG="$LOGS/runpod_full.yaml"
  "$PY" - "$CONFIG" <<'EOF'
import sys, yaml
config = yaml.safe_load(open("configs/runpod_gpu.yaml"))
config["pipeline"]["overlap_detection"]["refine"] = True
config["pipeline"]["overlap_separation"]["enabled"] = True
yaml.safe_dump(config, open(sys.argv[1], "w"), allow_unicode=True, sort_keys=False)
EOF
  echo "FULL pipeline: refine + overlap separation ($CONFIG)"
fi

CONFIG_PATH="$CONFIG" "$PY" meetasr/runpod/handler.py \
  --rp_serve_api --rp_api_port 8008 > "$LOGS/worker.log" 2>&1 &
pids+=($!)
wait_for "worker (models load ~20 s)" "$LOGS/worker.log" "Uvicorn running"

# The backend reads GROQ_API_KEY (comma-separated keys rotate); local .env
# files may only define GROQ_API_KEYS.
if [ -z "${GROQ_API_KEY:-}" ] && grep -q "^GROQ_API_KEYS=" .env 2>/dev/null; then
  GROQ_API_KEY="$(grep "^GROQ_API_KEYS=" .env | cut -d= -f2- | tr -d "\"'")"
  export GROQ_API_KEY
fi

DATABASE_URL="sqlite:///$LOGS/local_test.db" STORAGE_BACKEND=local \
STORAGE_LOCAL_ROOT="$LOGS/media" RUNPOD_URL=http://localhost:8008 RUNPOD_SERVERLESS=1 \
  "$PY" -m uvicorn meetasr.backend.api.app:app --host 127.0.0.1 --port 8010 \
  > "$LOGS/backend.log" 2>&1 &
pids+=($!)
wait_for "backend" "$LOGS/backend.log" "Application startup complete"

(cd frontend-next && NEXT_PUBLIC_MEETASR_API=http://127.0.0.1:8010 \
  MEETASR_API=http://127.0.0.1:8010 npx next dev -p 3000) > "$LOGS/frontend.log" 2>&1 &
pids+=($!)
wait_for "frontend" "$LOGS/frontend.log" "Ready in"

echo
echo "All up → http://localhost:3000 (log in as guest). Ctrl+C to stop."
echo "Logs: $LOGS/{worker,backend,frontend}.log"
wait
