#!/usr/bin/env bash
# GQA-native B2/B3 retrained on the shared 10k manifest (one question/image,
# 10000 records), matching Full Ours' data budget.  Fresh base init, no SFT.
# Outputs under outputs/gqa_native/baselines_10k/.  Resumable.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
EVAL="$ROOT/.venv-vllm-cu128-v011/bin/python"
PB="$ROOT/outputs/gqa_native/published_baselines"
NEW="$ROOT/outputs/gqa_native/baselines_10k"
FB="$ROOT/outputs/gqa_native/full_ours_base"
MAN="$ROOT/outputs/gqa_native/manifests"
REC_B2="$FB/data/records_10k_1per.jsonl"
REC_B3="$FB/data/train_targets_10k_1per.jsonl"
B2CACHE="$PB/b2_contrastive/cache"
B3CACHE="$PB/b3_implicit/cache"
DEV="$MAN/dev1000_images.jsonl"
REF="$MAN/dev1000_refs.json"
B2="$NEW/b2_contrastive"
B3="$NEW/b3_implicit"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "[gqa-b2b3-10k $(date +%F_%H:%M:%S)] $*"; }
mkdir -p "$NEW/logs"
exec 9>"$NEW/.chain.lock"
flock -n 9 || { echo "another b2b3 chain holds $NEW/.chain.lock" >&2; exit 1; }

# ---------------- B2 ----------------
if [ ! -f "$B2/full_fast/backbone/adapter_model.safetensors" ]; then
  log "B2 train 2 epochs on records_10k_1per (10000 rec)"
  taskset -c 0-7 "$PY" scripts/vvqg/train_b2.py --records "$REC_B2" --cache "$B2CACHE" --output "$B2/full_fast" \
    --per-device-batch-size 4 --gradient-accumulation-steps 4 --dataloader-num-workers 4 \
    --no-gradient-checkpointing --attn-implementation flash_attention_2 --processor-backend pil
fi
[ -f "$B2/greedy1000.jsonl" ] || { log "B2 greedy"; taskset -c 0-7 "$PY" scripts/vvqg/generate_b2.py --manifest "$DEV" --captions "$B2CACHE/captions.jsonl" --checkpoint "$B2/full_fast" --output "$B2/greedy1000.jsonl" --num-samples 1 --batch-size 8; }
[ -f "$B2/sampled4x1000.jsonl" ] || { log "B2 sampled4"; taskset -c 0-7 "$PY" scripts/vvqg/generate_b2.py --manifest "$DEV" --captions "$B2CACHE/captions.jsonl" --checkpoint "$B2/full_fast" --output "$B2/sampled4x1000.jsonl" --num-samples 4 --temperature 0.8 --top-p 0.9 --batch-size 8; }

# ---------------- B3 ----------------
if [ ! -f "$B3/full_fast/guided_head.pt" ]; then
  log "B3 train 2 epochs on train_targets_10k_1per (10000 rec)"
  taskset -c 0-7 "$PY" scripts/vvqg/train_b3.py --records "$REC_B3" --detector-cache "$B3CACHE/detector.jsonl" --output "$B3/full_fast" \
    --per-device-batch-size 4 --gradient-accumulation-steps 4 --dataloader-num-workers 4 \
    --no-gradient-checkpointing --attn-implementation flash_attention_2 --processor-backend pil
fi
[ -f "$B3/greedy1000.jsonl" ] || { log "B3 greedy"; taskset -c 0-7 "$PY" scripts/vvqg/generate_b3.py --manifest "$DEV" --detector-cache "$B3CACHE/detector_dev.jsonl" --checkpoint "$B3/full_fast" --output "$B3/greedy1000.jsonl" --num-samples 1 --batch-size 4; }
[ -f "$B3/sampled4x1000.jsonl" ] || { log "B3 sampled4"; taskset -c 0-7 "$PY" scripts/vvqg/generate_b3.py --manifest "$DEV" --detector-cache "$B3CACHE/detector_dev.jsonl" --checkpoint "$B3/full_fast" --output "$B3/sampled4x1000.jsonl" --num-samples 4 --temperature 0.8 --top-p 0.9 --batch-size 4; }

# ---------------- evaluation ----------------
for pair in "b2:$B2" "b3:$B3"; do
  sys="${pair%%:*}"; dir="${pair#*:}"
  [ -f "$dir/evaluation/dev1000/summary.json" ] || { log "$sys evaluate"; "$EVAL" scripts/vvqg/gqa_evaluate.py --predictions "$dir/greedy1000.jsonl" --refs "$REF" --system "$sys" --out "$dir/evaluation/dev1000" --concurrency 16 --retry-failures || true; }
  if [ ! -f "$dir/evaluation/diversity/summary.json" ]; then
    "$EVAL" - "$dir/sampled4x1000.jsonl" "$dir/evaluation/diversity_input.jsonl" <<'PY'
import json,sys
out=open(sys.argv[2],'w')
for l in open(sys.argv[1]):
    r=json.loads(l); out.write(json.dumps({"image_id":str(r["image_id"]),"generation_id":int(r.get("generation_id",0)),"question":r["question"]})+"\n")
PY
    taskset -c 0-7 "$EVAL" scripts/vvqg/evaluate_seeq_diversity.py --model "${VVQG_MODELS_DIR:-$HOME/models}/Qwen3-Embedding-4B" --zero-shot "$dir/evaluation/diversity_input.jsonl" --sft "$dir/evaluation/diversity_input.jsonl" --output "$dir/evaluation/diversity"
  fi
done
log "B2/B3 10k chain complete"
