#!/bin/bash
# GQA-native Faithful SeeQ-style reproduction chain (A1).
# Base Qwen3-VL-8B-Instruct seed (zero LoRA) -> R1 proposal/rewrite/filter -> M1
# -> R2 (M1 proposer, frozen seed-M0 rewrite/filter) -> M2. Resumable.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
VLLM="$ROOT/.venv-vllm-cu128-v011/bin/python"
TRAIN="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs/gqa_native/published_baselines/seeq"
IMAGES="$ROOT/outputs/gqa_native/manifests/method_train_10000.jsonl"
BASE="$HOME/models/Qwen3-VL-8B-Instruct"
M0="$OUT/seed_m0"
M0H="1a736f3ffc4dce18cd8f5d2d5ef1b49221b759f8ffa37ea703534889c1db2738"
M1="$OUT/round1_m1/checkpoint-final"
M2="$OUT/round2_m2/checkpoint-final"
mkdir -p "$OUT/logs"

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
log() { echo "[gqa-seeq $(date +%F_%H:%M:%S)] $*"; }

if [[ ! -f "$OUT/round1/funnel.json" ]]; then
  log "R1 generation (seed M0 proposer+reference)"
  taskset -c 0-7 "$VLLM" scripts/vvqg/run_seeq_faithful.py \
    --round 1 --images "$IMAGES" --stage-dir "$OUT/round1" \
    --base-model "$BASE" --proposer-adapter "$M0" --reference-adapter "$M0" \
    --m0-hash "$M0H" --current-m0-hash "$M0H" --retained 10000 --batch-size 256 --gpu-memory-utilization 0.8
fi

if [[ ! -f "$M1/adapter_model.safetensors" ]]; then
  log "train M1 (seed M0 -> M1)"
  taskset -c 0-7 "$TRAIN" scripts/vvqg/train_seeq_faithful.py \
    --formatted "$OUT/round1/retained/formatted.jsonl" \
    --base-model "$BASE" --start-adapter "$M0" --output "$M1" \
    --dataloader-num-workers 4 --gradient-checkpointing
  cp "$M1/training_summary.json" "$OUT/round1_m1/training_summary.json" 2>/dev/null || true
fi

if [[ ! -f "$OUT/round2/funnel.json" ]]; then
  log "R2 generation (M1 proposer, frozen seed M0 reference)"
  taskset -c 0-7 "$VLLM" scripts/vvqg/run_seeq_faithful.py \
    --round 2 --images "$IMAGES" --stage-dir "$OUT/round2" \
    --base-model "$BASE" --proposer-adapter "$M1" --reference-adapter "$M0" \
    --m0-hash "$M0H" --current-m0-hash "$M0H" --retained 10000 --batch-size 256 --gpu-memory-utilization 0.8
fi

if [[ ! -f "$M2/adapter_model.safetensors" ]]; then
  log "train M2 (M1 -> M2)"
  taskset -c 0-7 "$TRAIN" scripts/vvqg/train_seeq_faithful.py \
    --formatted "$OUT/round2/retained/formatted.jsonl" \
    --base-model "$BASE" --start-adapter "$M1" --output "$M2" \
    --dataloader-num-workers 4 --gradient-checkpointing
  cp "$M2/training_summary.json" "$OUT/round2_m2/training_summary.json" 2>/dev/null || true
fi

log "SeeQ GQA chain complete: M2 at $M2"
