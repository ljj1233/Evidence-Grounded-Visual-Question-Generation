#!/usr/bin/env bash
# Top-level GQA-native, base-init, shared-10k pipeline:
#   1. Full Ours (no GQA-SFT): base -> UVFC(base)+Direct-4 -> S3 -> D1(base) -> G2 -> eval
#   2. B2/B3 retrained on the same 10k manifest (10000 records) + eval
#   3. Rebuild the combined GQA table
# Each sub-chain holds its own flock, so duplicate launches are refused.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
LOG="$ROOT/outputs/gqa_native/full_ours_base/logs"
mkdir -p "$LOG"
log() { echo "[gqa-base-pipeline $(date +%F_%H:%M:%S)] $*"; }

log "Full Ours (base) chain"
bash scripts/vvqg/gqa_full_ours_base_chain.sh all

log "B2/B3 10k retrain chain"
bash scripts/vvqg/gqa_b2b3_10k_chain.sh

log "done"
