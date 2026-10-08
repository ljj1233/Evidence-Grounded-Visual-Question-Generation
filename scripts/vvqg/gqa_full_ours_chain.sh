#!/usr/bin/env bash
# GQA-native Full Ours chain: public backbone -> GQA-SFT -> GQA-UVFC ->
# Direct-4 -> Flat+UVFC -> S3 -> D1 -> G2 -> dev1000 evaluation.
# Resumable: every stage is skipped when its output exists.  No VQAv2 learned
# artifact is read.  See docs/GQA_NATIVE_FULL_OURS_METHOD_CONTRACT.md.
#
# Usage:  bash scripts/vvqg/gqa_full_ours_chain.sh [stage]
#   stage in {data,sft,uvfc,direct4,score,d1,g2,eval,all}  (default all)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
EVAL="$ROOT/.venv-vllm-cu128-v011/bin/python"

MAN="$ROOT/outputs/gqa_native/manifests"
BASE="$ROOT/outputs/gqa_native/full_ours"
DATA="$BASE/data"
UVFC="$BASE/uvfc"
SFT="$BASE/sft/checkpoint-final"
D4="$BASE/direct4"
QSE="$BASE/qse_teacher"
D1="$BASE/distill_d1/checkpoint-final"
G2="$BASE/g2"
EV="$BASE/evaluation/dev1000"

MODEL="${VVQG_MODELS_DIR:-$HOME/models}/Qwen3-VL-8B-Instruct"
FLAT="${VVQG_MODELS_DIR:-$HOME/models}/Qwen2.5-VL-7B-Instruct"

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 RAYON_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

log() { echo "[gqa-full-ours $(date +%F_%H:%M:%S)] $*"; }
run() { taskset -c 0-7 "$@"; }

STAGE="${1:-all}"

prepare_data() {
  mkdir -p "$DATA"
  [ -f "$DATA/sft_train_50000.jsonl" ] || jq -c '{image:.image_path, question:.question}' "$MAN/sft_train_50000.jsonl" > "$DATA/sft_train_50000.jsonl"
  [ -f "$DATA/method_train_10000.jsonl" ] || jq -c '{image:.image_path, image_id:.image_id}' "$MAN/method_train_10000.jsonl" > "$DATA/method_train_10000.jsonl"
  log "data ready: $(wc -l < "$DATA/sft_train_50000.jsonl") sft records, $(wc -l < "$DATA/method_train_10000.jsonl") method images"
}

do_sft() {
  if [ -f "$SFT/adapter_config.json" ]; then log "SFT exists, skip"; return; fi
  log "SFT 50k from public backbone"
  run "$PY" src/vvqg/training/train_sft.py --config configs/gqa_native/sft.yaml --train
}

do_uvfc() {
  if [ -f "$UVFC/weights.pt" ]; then log "UVFC exists, skip"; return; fi
  mkdir -p "$UVFC"
  uv() { run "$PY" scripts/vvqg/phase26_vfc.py "$@" --adapter "$SFT" --base "$MODEL" --train "$DATA/sft_train_50000.jsonl" --out "$UVFC"; }
  log "UVFC manifest"; uv manifest
  log "UVFC extract discovery 20000"; uv extract --split discovery
  log "UVFC extract calibration 5000 real"; uv extract --split calibration
  log "UVFC extract calibration 5000 blank"; uv extract --split calibration --blank
  log "UVFC train SAE"; uv train
  log "UVFC policy on method_train_10000"; uv policy --policy-manifest "$DATA/method_train_10000.jsonl"
  log "UVFC extract policy"; uv extract-policy
  log "UVFC calibrate v/u"; uv calibrate
}

do_direct4() {
  if [ -s "$D4/candidates.jsonl" ]; then log "Direct-4 exists, skip"; return; fi
  mkdir -p "$D4"
  log "Direct-4 on method_train_10000 (K=4 T=0.8)"
  run "$PY" scripts/vvqg/phase5b_direct4_generate.py --outdir "$D4" --manifest "$DATA/method_train_10000.jsonl" \
    --base "$MODEL" --sft-checkpoint "$SFT"
}

do_score() {
  local covered expected
  covered="$(jq -r '.image' "$D4/candidates.jsonl" | sort -u | wc -l)"
  expected="$(wc -l < "$DATA/method_train_10000.jsonl")"
  if [ "$covered" -ne "$expected" ]; then
    echo "Direct-4 covered $covered/$expected method images; refusing to score an incomplete pool" >&2
    exit 1
  fi
  log "score candidates (UVFC + Flat), $covered images covered, resumable"
  run "$PY" scripts/vvqg/phase5b_score_candidates.py --candidates "$D4/candidates.jsonl" --out "$D4/candidates_scored.jsonl" \
    --uvfc-dir "$UVFC" --base "$MODEL" --sft-checkpoint "$SFT" --flat-model "$FLAT"
}

do_d1() {
  mkdir -p "$QSE"
  if [ ! -f "$QSE/selected/selection_audit.json" ]; then
    log "S3 select tau=0.5"
    run "$PY" scripts/vvqg/phase5c_select.py --scored "$D4/candidates_scored.jsonl" --selector s3 --tau 0.5 --outdir "$QSE/selected"
  fi
  if [ ! -f "$QSE/targets_d1_single.jsonl" ]; then
    log "build D1/D2 targets tau=0.5"
    run "$PY" scripts/vvqg/phase5c_build_targets.py --scored "$D4/candidates_scored.jsonl" --manifest "$DATA/method_train_10000.jsonl" --outdir "$QSE" --tau 0.5
  fi
  if [ -f "$D1/adapter_config.json" ]; then log "D1 exists, skip"; return; fi
  log "D1 distillation from GQA-SFT"
  run "$PY" scripts/vvqg/phase5c_distill.py --targets "$QSE/targets_d1_single.jsonl" --outdir "$BASE/distill_d1" \
    --base "$MODEL" --init "$SFT" --epochs 2
}

do_g2() {
  if [ -f "$G2/training_summary.json" ]; then log "G2 exists, skip"; return; fi
  log "G2 static-UVFC GRPO from D1"
  run "$PY" scripts/vvqg/train_online_uvfc_relative.py --config configs/gqa_native/g2.yaml
}

do_eval() {
  mkdir -p "$EV"
  log "dev1000 evaluation (SeeQ-Luna + GPT-4o + traditional)"
  run "$EVAL" scripts/vvqg/gqa_evaluate.py --predictions "$G2/greedy1000.jsonl" --refs "$MAN/dev1000_refs.json" \
    --system gqa_full_ours --out "$EV" --concurrency 16 --retry-failures
}

case "$STAGE" in
  data) prepare_data ;;
  sft) prepare_data; do_sft ;;
  uvfc) do_uvfc ;;
  direct4) do_direct4 ;;
  score) do_score ;;
  d1) do_d1 ;;
  g2) do_g2 ;;
  eval) do_eval ;;
  all) prepare_data; do_sft; do_uvfc; do_direct4; do_score; do_d1; do_g2; do_eval ;;
  rest)
    while [ ! -f "$SFT/adapter_config.json" ]; do log "waiting for GQA SFT checkpoint"; sleep 120; done
    do_uvfc; do_direct4; do_score; do_d1; do_g2; do_eval
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
log "stage '$STAGE' done"
