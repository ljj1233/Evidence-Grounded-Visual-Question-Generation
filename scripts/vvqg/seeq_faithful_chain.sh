#!/bin/bash
# Autonomous faithful SeeQ chain: finish R1 -> train M1 -> R2 (M1 proposer) -> train M2.
# Resumable: each generation stage skips completed candidate keys; training starts
# from the frozen M0 (round 1) or M1 (round 2) adapter.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
VLLM="$ROOT/.venv-vllm-cu128-v011/bin/python"
TRAIN="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs/same_backbone_baselines/seeq_faithful"
MAN="$ROOT/outputs/same_backbone_baselines/manifests/train_unlabeled_images.jsonl"
M0="$ROOT/outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final"
M0H="5fb66c36a2255fb3a9011db198fd55a8be8b9cbb26eef611727a816bc7934e0e"
BASE="$HOME/models/Qwen3-VL-8B-Instruct"
M1="$OUT/round1_m1/checkpoint-final"
M2="$OUT/round2_m2/checkpoint-final"

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

# 1. wait for any running R1 generation
log "waiting for R1 generation to finish"
while pgrep -f 'run_seeq_faithful.py --round 1' >/dev/null; do sleep 30; done
log "R1 generation finished"

# 2. train M0 -> M1
if [[ ! -f "$M1/adapter_model.safetensors" ]]; then
  log "training M1"
  $TRAIN scripts/vvqg/train_seeq_faithful.py \
    --formatted "$OUT/round1/retained/formatted.jsonl" \
    --base-model "$BASE" --start-adapter "$M0" --output "$M1" \
    --dataloader-num-workers 8 --gradient-checkpointing
  cp "$M1/training_summary.json" "$OUT/round1_m1/training_summary.json" 2>/dev/null || true
fi
log "M1 ready at $M1"

# 3. R2 generation with M1 proposer and frozen M0 rewrite/filter
if [[ ! -f "$OUT/round2/funnel.json" ]]; then
  log "running R2 generation"
  $VLLM scripts/vvqg/run_seeq_faithful.py \
    --round 2 --images "$MAN" --stage-dir "$OUT/round2" \
    --base-model "$BASE" --proposer-adapter "$M1" --reference-adapter "$M0" \
    --m0-hash "$M0H" --current-m0-hash "$M0H" --retained 10000 --batch-size 128
fi
log "R2 generation finished"

# 4. train M1 -> M2
if [[ ! -f "$M2/adapter_model.safetensors" ]]; then
  log "training M2"
  $TRAIN scripts/vvqg/train_seeq_faithful.py \
    --formatted "$OUT/round2/retained/formatted.jsonl" \
    --base-model "$BASE" --start-adapter "$M1" --output "$M2" \
    --dataloader-num-workers 8 --gradient-checkpointing
  cp "$M2/training_summary.json" "$OUT/round2_m2/training_summary.json" 2>/dev/null || true
fi
log "M2 ready at $M2"
log "chain complete"
