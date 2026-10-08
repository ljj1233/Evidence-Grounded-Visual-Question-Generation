#!/bin/bash
# GQA-native B2 (ConVQG contrastive grounding) + B3 (Guiding VQG implicit guided).
# Pool = method_train_10000 (user-confirmed). Same backbone/recipe as the VQAv2 runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PB="$ROOT/outputs/gqa_native/published_baselines"
MAN="$ROOT/outputs/gqa_native/manifests"
REC=$(printf '%s' "$PB/records_method10000.jsonl")
DEV="$MAN/dev1000_images.jsonl"
B2="$PB/b2_contrastive"; B3="$PB/b3_implicit"
GEN="$ROOT/.venv/bin/python"
EVAL="$ROOT/.venv-vllm-cu128-v011/bin/python"
mkdir -p "$PB/logs"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
log() { echo "[gqa-b2b3 $(date +%F_%H:%M:%S)] $*"; }

# ---------------- B2 ----------------
if [[ ! -f "$B2/cache/captions.jsonl" || ! -f "$B2/cache/q_txt.jsonl" ]]; then
  log "B2 cache (captions/q_img/q_txt) over method_train_10000 + dev1000"
  taskset -c 0-7 "$GEN" scripts/vvqg/prepare_b2_cache.py --records "$REC" --development-manifest "$DEV" \
    --output "$B2/cache" --model "${VVQG_MODELS_DIR:-$HOME/models}/Qwen3-VL-8B-Instruct" --batch-size 8 --stage all
fi
[[ -f "$B2/cache/embeddings.json" ]] || { log "B2 embeddings"; taskset -c 0-7 "$GEN" scripts/vvqg/prepare_b2_embeddings.py --records "$REC" --image-cache "$B2/cache" --output "$B2/cache/embeddings.json"; }
if [[ ! -f "$B2/full_fast/guided_head.pt" && ! -f "$B2/full_fast/contrastive_projection.pt" ]]; then
  log "B2 train"
  taskset -c 0-7 "$GEN" scripts/vvqg/train_b2.py --records "$REC" --cache "$B2/cache" --output "$B2/full_fast" \
    --per-device-batch-size 4 --gradient-accumulation-steps 4 --dataloader-num-workers 4 --no-gradient-checkpointing --attn-implementation flash_attention_2 --processor-backend pil
fi
[[ -f "$B2/greedy1000.jsonl" ]] || { log "B2 greedy"; taskset -c 0-7 "$GEN" scripts/vvqg/generate_b2.py --manifest "$DEV" --captions "$B2/cache/captions.jsonl" --checkpoint "$B2/full_fast" --output "$B2/greedy1000.jsonl" --num-samples 1 --batch-size 8; }
[[ -f "$B2/sampled4x1000.jsonl" ]] || { log "B2 sampled4"; taskset -c 0-7 "$GEN" scripts/vvqg/generate_b2.py --manifest "$DEV" --captions "$B2/cache/captions.jsonl" --checkpoint "$B2/full_fast" --output "$B2/sampled4x1000.jsonl" --num-samples 4 --temperature 0.8 --top-p 0.9 --batch-size 8; }

# ---------------- B3 ----------------
[[ -f "$B3/cache/detector.jsonl" ]] || { log "B3 detector train cache"; taskset -c 0-7 "$GEN" scripts/vvqg/prepare_b3_detector_cache.py --manifest "$MAN/method_train_10000.jsonl" --output "$B3/cache/detector.jsonl"; }
[[ -f "$B3/cache/detector_dev.jsonl" ]] || { log "B3 detector dev cache"; taskset -c 0-7 "$GEN" scripts/vvqg/prepare_b3_detector_cache.py --manifest "$DEV" --output "$B3/cache/detector_dev.jsonl"; }
[[ -f "$B3/cache/train_targets.jsonl" ]] || { log "B3 targets"; taskset -c 0-7 "$GEN" scripts/vvqg/gqa_prepare_b3_targets.py --records "$REC" --detector-cache "$B3/cache/detector.jsonl" --output "$B3/cache/train_targets.jsonl"; }
[[ -f "$B3/full_fast/guided_head.pt" ]] || { log "B3 train"; taskset -c 0-7 "$GEN" scripts/vvqg/train_b3.py --records "$B3/cache/train_targets.jsonl" --detector-cache "$B3/cache/detector.jsonl" --output "$B3/full_fast" --per-device-batch-size 4 --gradient-accumulation-steps 4 --dataloader-num-workers 4 --no-gradient-checkpointing --attn-implementation flash_attention_2 --processor-backend pil; }
[[ -f "$B3/greedy1000.jsonl" ]] || { log "B3 greedy"; taskset -c 0-7 "$GEN" scripts/vvqg/generate_b3.py --manifest "$DEV" --detector-cache "$B3/cache/detector_dev.jsonl" --checkpoint "$B3/full_fast" --output "$B3/greedy1000.jsonl" --num-samples 1 --batch-size 4; }
[[ -f "$B3/sampled4x1000.jsonl" ]] || { log "B3 sampled4"; taskset -c 0-7 "$GEN" scripts/vvqg/generate_b3.py --manifest "$DEV" --detector-cache "$B3/cache/detector_dev.jsonl" --checkpoint "$B3/full_fast" --output "$B3/sampled4x1000.jsonl" --num-samples 4 --temperature 0.8 --top-p 0.9 --batch-size 4; }

# ---------------- evaluation ----------------
for pair in "b2:$B2" "b3:$B3"; do
  sys="${pair%%:*}"; dir="${pair#*:}"
  mkdir -p "$dir/evaluation"
  "$EVAL" scripts/vvqg/gqa_evaluate.py --predictions "$dir/greedy1000.jsonl" --refs "$MAN/dev1000_refs.json" --system "$sys" --out "$dir/evaluation/dev1000" --concurrency 16 --retry-failures || true
done
for pair in "b2:$B2" "b3:$B3"; do
  sys="${pair%%:*}"; dir="${pair#*:}"
  "$EVAL" - "$dir/sampled4x1000.jsonl" "$dir/evaluation/diversity_input.jsonl" <<'PY'
import json,sys
out=open(sys.argv[2],'w')
for l in open(sys.argv[1]):
    r=json.loads(l); out.write(json.dumps({"image_id":str(r["image_id"]),"generation_id":int(r.get("generation_id",0)),"question":r["question"]})+"\n")
PY
  [[ -f "$dir/evaluation/diversity/summary.json" ]] || taskset -c 0-7 "$EVAL" scripts/vvqg/evaluate_seeq_diversity.py --model "${VVQG_MODELS_DIR:-$HOME/models}/Qwen3-Embedding-4B" --zero-shot "$dir/evaluation/diversity_input.jsonl" --sft "$dir/evaluation/diversity_input.jsonl" --output "$dir/evaluation/diversity"
done
log "B2/B3 GQA chain complete"
