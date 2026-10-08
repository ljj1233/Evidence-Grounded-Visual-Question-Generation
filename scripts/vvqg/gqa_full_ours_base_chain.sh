#!/usr/bin/env bash
# GQA-native Full Ours, base-init (NO GQA-SFT), shared 10k manifest.
#   base Qwen3-VL-8B-Instruct -> UVFC(base) + Direct-4(base) -> S3 -> D1(base)
#   -> G2(D1) -> Full Ours.  Single-writer lock; idempotent stages.
# Usage: bash scripts/vvqg/gqa_full_ours_base_chain.sh [stage]
#   stage in {data,uvfc,direct4,score,d1,g2,eval,all}  (default all)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
EVAL="$ROOT/.venv-vllm-cu128-v011/bin/python"

BASE="$ROOT/outputs/gqa_native/full_ours_base"
DATA="$BASE/data"
BASEZERO="$BASE/base_zero_lora"
UVFC="$BASE/uvfc"
D4="$BASE/direct4"
QSE="$BASE/qse_teacher"
D1="$BASE/distill_d1/checkpoint-final"
G2="$BASE/g2"
EV="$BASE/evaluation/dev1000"
MODEL="${VVQG_MODELS_DIR:-$HOME/models}/Qwen3-VL-8B-Instruct"
FLAT="${VVQG_MODELS_DIR:-$HOME/models}/Qwen2.5-VL-7B-Instruct"

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 RAYON_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

log() { echo "[gqa-base $(date +%F_%H:%M:%S)] $*"; }
run() { taskset -c 0-7 "$@"; }
STAGE="${1:-all}"

mkdir -p "$BASE/logs"
exec 9>"$BASE/.chain.lock"
if ! flock -n 9; then echo "another chain holds $BASE/.chain.lock" >&2; exit 1; fi

do_uvfc() {
  if [ -f "$UVFC/weights.pt" ]; then log "UVFC exists, skip"; return; fi
  mkdir -p "$UVFC"
  uv() { run "$PY" scripts/vvqg/phase26_vfc.py "$@" --adapter none --base "$MODEL" \
    --train "$DATA/gqa_train_50k.jsonl" --out "$UVFC"; }
  log "UVFC manifest"; uv manifest
  log "UVFC discovery 20000"; uv extract --split discovery
  log "UVFC calibration 5000 real"; uv extract --split calibration
  log "UVFC calibration 5000 blank"; uv extract --split calibration --blank
  log "UVFC train SAE"; uv train
  log "UVFC policy on method_train_10000"; uv policy --policy-manifest "$DATA/method_train_10000.jsonl"
  log "UVFC extract policy"; uv extract-policy
  log "UVFC calibrate v/u"; uv calibrate
}

do_direct4() {
  if [ -s "$D4/candidates.jsonl" ]; then log "Direct-4 exists, skip"; return; fi
  mkdir -p "$D4"
  log "Direct-4 from base on method_train_10000"
  run "$PY" scripts/vvqg/phase5b_direct4_generate.py --outdir "$D4" --manifest "$DATA/method_train_10000.jsonl" \
    --base "$MODEL" --sft-checkpoint "$BASEZERO"
}

do_score() {
  local covered expected
  covered="$(jq -r '.image' "$D4/candidates.jsonl" | sort -u | wc -l)"
  expected="$(wc -l < "$DATA/method_train_10000.jsonl")"
  [ "$covered" -eq "$expected" ] || { echo "Direct-4 covered $covered/$expected" >&2; exit 1; }
  log "score candidates (UVFC(base)+Flat), resumable ($covered images)"
  run "$PY" scripts/vvqg/phase5b_score_candidates.py --candidates "$D4/candidates.jsonl" --out "$D4/candidates_scored.jsonl" \
    --uvfc-dir "$UVFC" --base "$MODEL" --sft-checkpoint "$BASEZERO" --flat-model "$FLAT"
}

do_d1() {
  mkdir -p "$QSE"
  [ -f "$QSE/selected/selection_audit.json" ] || { log "S3 select tau=0.5"; run "$PY" scripts/vvqg/phase5c_select.py --scored "$D4/candidates_scored.jsonl" --selector s3 --tau 0.5 --outdir "$QSE/selected"; }
  [ -f "$QSE/targets_d1_single.jsonl" ] || { log "build D1 targets"; run "$PY" scripts/vvqg/phase5c_build_targets.py --scored "$D4/candidates_scored.jsonl" --manifest "$DATA/method_train_10000.jsonl" --outdir "$QSE" --tau 0.5; }
  if [ -f "$D1/adapter_config.json" ]; then log "D1 exists, skip"; return; fi
  log "D1 distillation from base"
  run "$PY" scripts/vvqg/phase5c_distill.py --targets "$QSE/targets_d1_single.jsonl" --outdir "$BASE/distill_d1" \
    --base "$MODEL" --init "$BASEZERO" --epochs 2
}

do_g2() {
  if [ -f "$G2/training_summary.json" ]; then log "G2 exists, skip"; return; fi
  log "G2 static-UVFC GRPO from D1"
  run "$PY" scripts/vvqg/train_online_uvfc_relative.py --config configs/gqa_native/g2_base.yaml
}

do_eval() {
  mkdir -p "$EV"
  log "dev1000 evaluation"
  run "$EVAL" scripts/vvqg/gqa_evaluate.py --predictions "$G2/greedy1000.jsonl" --refs outputs/gqa_native/manifests/dev1000_refs.json \
    --system gqa_full_ours --out "$EV" --concurrency 16 --retry-failures
}

case "$STAGE" in
  uvfc) do_uvfc ;;
  direct4) do_direct4 ;;
  score) do_score ;;
  d1) do_d1 ;;
  g2) do_g2 ;;
  eval) do_eval ;;
  all) do_uvfc; do_direct4; do_score; do_d1; do_g2; do_eval ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
log "stage '$STAGE' done"
