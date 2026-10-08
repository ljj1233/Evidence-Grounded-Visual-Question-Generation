#!/bin/bash
# Faithful SeeQ evaluation: M0/M1/M2 greedy + sampled4 on development1000.
# Traditional metrics + local Qwen3-Embedding-4B diversity (no API); greedy
# visual-centric (gpt-5.6-luna variant) + visual support (gpt-4o) use the API.
# M0 reuses the frozen prior evaluation artifacts (same checkpoint).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
VLLM="$ROOT/.venv-vllm-cu128-v011/bin/python"
GEN="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs/same_backbone_baselines/seeq_faithful"
EVAL="$OUT/evaluation"
OLD="$ROOT/outputs/same_backbone_baselines/seeq_style/evaluation"
DEV="$ROOT/outputs/same_backbone_baselines/manifests/development1000_images.jsonl"
REF="$ROOT/outputs/evaluation/sft_greedy/val2014_sample1000.jsonl"
M0="$ROOT/outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final"
M1="$OUT/round1_m1/checkpoint-final"
M2="$OUT/round2_m2/checkpoint-final"
mkdir -p "$EVAL/visual_centric" "$EVAL/visual_support"
log() { echo "[eval $(date +%H:%M:%S)] $*"; }

reuse_m0() {
  for src in m0_greedy m0_sampled4; do
    [[ -f "$EVAL/$src.jsonl" ]] || cp "$OLD/$src.jsonl" "$EVAL/$src.jsonl"
  done
  for src in metrics_m0_greedy metrics_m0_sampled; do [[ -d "$EVAL/$src" ]] || cp -r "$OLD/$src" "$EVAL/$src"; done
  [[ -f "$EVAL/visual_centric/m0" ]] || cp "$OLD/visual_centric/m0" "$EVAL/visual_centric/m0"
  [[ -f "$EVAL/visual_support/m0.jsonl" ]] || cp "$OLD/visual_support/m0.jsonl" "$EVAL/visual_support/m0.jsonl"
}

for pair in "m1:$M1" "m2:$M2"; do
  name="${pair%%:*}"; ckpt="${pair#*:}"
  [[ -f "$EVAL/${name}_greedy.jsonl" ]] || { log "generate $name greedy"; $GEN scripts/vvqg/generate.py --data "$DEV" --checkpoint "$ckpt" --output "$EVAL/${name}_greedy.jsonl" --schema greedy --batch-size 16; }
  [[ -f "$EVAL/${name}_sampled4.jsonl" ]] || { log "generate $name sampled4"; $GEN scripts/vvqg/generate.py --data "$DEV" --checkpoint "$ckpt" --output "$EVAL/${name}_sampled4.jsonl" --schema sampled --num-samples 4 --batch-size 16; }
done
reuse_m0

for m in m0 m1 m2; do
  [[ -d "$EVAL/metrics_${m}_greedy" ]] || { log "metrics $m greedy"; $VLLM scripts/vvqg/evaluate.py --predictions "$EVAL/${m}_greedy.jsonl" --references "$REF" --out "$EVAL/metrics_${m}_greedy"; }
  [[ -d "$EVAL/metrics_${m}_sampled" ]] || { log "metrics $m sampled"; $VLLM scripts/vvqg/evaluate.py --predictions "$EVAL/${m}_sampled4.jsonl" --references "$REF" --out "$EVAL/metrics_${m}_sampled"; }
done

for pair in "m0:m1:M0M1" "m0:m2:M0M2"; do
  a="${pair%%:*}"; rest="${pair#*:}"; b="${rest%%:*}"; tag="${rest#*:}"
  [[ -f "$EVAL/diversity_${tag}/summary.json" ]] || { log "diversity $a vs $b"; $VLLM scripts/vvqg/evaluate_seeq_diversity.py --model "$HOME/models/Qwen3-Embedding-4B" --zero-shot "$EVAL/${a}_sampled4.jsonl" --sft "$EVAL/${b}_sampled4.jsonl" --output "$EVAL/diversity_${tag}"; }
done

for m in m1 m2; do
  [[ -f "$EVAL/visual_centric/$m" ]] || { log "visual-centric $m"; $VLLM scripts/vvqg/evaluate_visual_centric.py --predictions "$EVAL/${m}_greedy.jsonl" --output "$EVAL/visual_centric/$m" --model gpt-5.6-luna --allow-variant; }
done
[[ -f "$EVAL/visual_support/summary.json" ]] || { log "visual support"; $VLLM scripts/vvqg/evaluate_visual_support.py --predictions "m0=$EVAL/m0_greedy.jsonl" --predictions "m1=$EVAL/m1_greedy.jsonl" --predictions "m2=$EVAL/m2_greedy.jsonl" --out "$EVAL/visual_support"; }

log "tables"
$GEN scripts/vvqg/seeq_faithful_tables.py --root "$OUT" --out "$EVAL" || true
log "evaluation complete"
