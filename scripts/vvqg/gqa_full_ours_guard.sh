#!/usr/bin/env bash
# Guarded launcher for the GQA-native Full Ours chain.
#   * single-writer lock (flock) prevents duplicate chain processes
#   * read-only integrity verification between every stage; a failing check
#     aborts before the next stage starts
#   * delegates to the idempotent gqa_full_ours_chain.sh stage functions
# The lock fd is inherited across exec, so it is held for the whole run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE="$ROOT/outputs/gqa_native/full_ours"
PY="$ROOT/.venv/bin/python"
LOCK="$BASE/.chain.lock"
mkdir -p "$BASE/logs"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[gqa-guard $(date +%F_%H:%M:%S)] another chain holds $LOCK; refusing duplicate" >&2
  exit 1
fi
log() { echo "[gqa-guard $(date +%F_%H:%M:%S)] $*" >&2; }
log "lock acquired (pid $$)"

verify() { "$PY" scripts/vvqg/verify_gqa_full_ours.py --phase "$1" || { log "verify $1 FAILED"; exit 2; }; }

# Require the SFT stage to have completed successfully before continuing:
# both the saved adapter and the training_summary.json success marker.
sft_ready() { [ -f "$BASE/sft/checkpoint-final/adapter_config.json" ] && [ -f "$BASE/sft/training_summary.json" ]; }
if ! sft_ready; then
  log "waiting for GQA SFT to finish (adapter + training_summary.json)"
  while ! sft_ready; do sleep 120; done
fi

verify pre-downstream
log "stage score"
bash scripts/vvqg/gqa_full_ours_chain.sh score
verify post-score
log "stage d1"
bash scripts/vvqg/gqa_full_ours_chain.sh d1
verify post-d1
log "stage g2"
bash scripts/vvqg/gqa_full_ours_chain.sh g2
verify post-g2
log "stage eval"
bash scripts/vvqg/gqa_full_ours_chain.sh eval
log "chain complete"
