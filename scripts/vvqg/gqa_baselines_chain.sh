#!/bin/bash
# GQA published baselines: external LOVA3/Genixer + SeeQ inference + evaluation.
# Waits for the SeeQ training chain to finish, then runs everything on the single GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PB="$ROOT/outputs/gqa_native/published_baselines"
SAN="$ROOT/outputs/gqa_native/manifests/sanity50_images.jsonl"
DEV="$ROOT/outputs/gqa_native/manifests/dev1000_images.jsonl"
SANREF="$ROOT/outputs/gqa_native/manifests/sanity50_refs.json"
DEVREF="$ROOT/outputs/gqa_native/manifests/dev1000_refs.json"
LOVA3_REPO="$ROOT/third_party/LOVA3"
GENIXER_REPO="$ROOT/third_party/Genixer/Genixer_LLaVA"
LOVA3_MODEL="$HOME/models/LOVA3-llava-v1.5-7b"
GENIXER_MODEL="$HOME/models/Genixer-llava-v1.5-7b"
EMBED="$HOME/models/Qwen3-Embedding-4B"
INSTR="Please provide a clear and direct question and answer after examining the image."
M2="$PB/seeq/round2_m2/checkpoint-final"
EVAL="$ROOT/.venv-vllm-cu128-v011/bin/python"
mkdir -p "$PB/logs" "$PB/comparison"

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
log() { echo "[gqa-pub $(date +%F_%H:%M:%S)] $*"; }

if [[ "${1:-}" == "--no-wait" ]]; then
  log "skipping SeeQ-chain wait"
else
  while pgrep -f gqa_native_seeq_chain >/dev/null; do log "waiting for SeeQ training chain"; sleep 120; done
fi
log "GPU free; SeeQ M2 present=$([[ -f "$M2/adapter_model.safetensors" ]] && echo yes || echo no)"

lova3() {
  local schema="$1" out="$2" extra="$3"
  ( cd "$LOVA3_REPO" && taskset -c 0-7 "$ROOT/.venv_lova3/bin/python" "$ROOT/scripts/vvqg/published_vqg_inference.py" \
    --method lova3 --repo "$LOVA3_REPO" --model-path "$LOVA3_MODEL" \
    --manifest "$4" --out "$out" --schema "$schema" --instruction "$INSTR" \
    --conv-mode vicuna_v1 --max-new-tokens 1024 --seed 42 $extra )
}
genixer() {
  local schema="$1" out="$2" extra="$3"
  ( cd "$GENIXER_REPO" && taskset -c 0-7 "$ROOT/.venv_genixer/bin/python" "$ROOT/scripts/vvqg/published_vqg_inference.py" \
    --method genixer --repo "$GENIXER_REPO" --model-path "$GENIXER_MODEL" \
    --manifest "$4" --out "$out" --schema "$schema" --instruction "$INSTR" \
    --instruction-suffix $'\nThis is the Common VQA task.' \
    --conv-mode vicuna_v1 --max-new-tokens 1024 --seed 42 $extra )
}

# ---------- LOVA3 (external, greedy + sampled4) ----------
log "LOVA3 sanity50 greedy"
[[ -f "$PB/lova3/sanity50/predictions.jsonl" ]] || lova3 greedy "$PB/lova3/sanity50" "--temperature 0.0 --num-samples 1" "$SAN"
log "LOVA3 dev1000 greedy"
[[ -f "$PB/lova3/predictions.jsonl" ]] || lova3 greedy "$PB/lova3" "--temperature 0.0 --num-samples 1" "$DEV"
log "LOVA3 dev1000 sampled4"
[[ -f "$PB/lova3/sampled4x.jsonl" ]] || lova3 sampled4 "$PB/lova3" "--temperature 0.8 --top-p 0.9 --num-samples 4" "$DEV"
log "LOVA3 evaluate"
"$EVAL" scripts/vvqg/gqa_evaluate.py --predictions "$PB/lova3/sanity50/predictions.jsonl" --refs "$SANREF" --system lova3 --out "$PB/lova3/evaluation/sanity50" --concurrency 16
"$EVAL" scripts/vvqg/gqa_evaluate.py --predictions "$PB/lova3/predictions.jsonl" --refs "$DEVREF" --system lova3 --out "$PB/lova3/evaluation/dev1000" --concurrency 16

# ---------- Genixer (external, released T=0.2 primary + sampled4) ----------
log "Genixer sanity50 primary T=0.2"
[[ -f "$PB/genixer/sanity50/predictions.jsonl" ]] || genixer greedy "$PB/genixer/sanity50" "--temperature 0.2 --num-samples 1" "$SAN"
log "Genixer dev1000 primary T=0.2"
[[ -f "$PB/genixer/predictions.jsonl" ]] || genixer greedy "$PB/genixer" "--temperature 0.2 --num-samples 1" "$DEV"
log "Genixer dev1000 sampled4"
[[ -f "$PB/genixer/sampled4x.jsonl" ]] || genixer sampled4 "$PB/genixer" "--temperature 0.8 --top-p 0.9 --num-samples 4" "$DEV"
log "Genixer evaluate"
"$EVAL" scripts/vvqg/gqa_evaluate.py --predictions "$PB/genixer/sanity50/predictions.jsonl" --refs "$SANREF" --system genixer --out "$PB/genixer/evaluation/sanity50" --concurrency 16
"$EVAL" scripts/vvqg/gqa_evaluate.py --predictions "$PB/genixer/predictions.jsonl" --refs "$DEVREF" --system genixer --out "$PB/genixer/evaluation/dev1000" --concurrency 16

# ---------- SeeQ M2 (greedy + sampled4) ----------
if [[ -f "$M2/adapter_model.safetensors" ]]; then
log "SeeQ sanity50 greedy"
[[ -f "$PB/seeq/evaluation/sanity50/predictions.jsonl" ]] || taskset -c 0-7 .venv/bin/python scripts/vvqg/generate.py --data "$SAN" --checkpoint "$M2" --output "$PB/seeq/evaluation/sanity50/predictions.jsonl" --schema greedy --batch-size 16 --no-progress
log "SeeQ dev1000 greedy"
[[ -f "$PB/seeq/evaluation/dev1000/predictions.jsonl" ]] || taskset -c 0-7 .venv/bin/python scripts/vvqg/generate.py --data "$DEV" --checkpoint "$M2" --output "$PB/seeq/evaluation/dev1000/predictions.jsonl" --schema greedy --batch-size 16 --no-progress
log "SeeQ dev1000 sampled4"
[[ -f "$PB/seeq/evaluation/dev1000_sampled4.jsonl" ]] || taskset -c 0-7 .venv/bin/python scripts/vvqg/generate.py --data "$DEV" --checkpoint "$M2" --output "$PB/seeq/evaluation/dev1000_sampled4.jsonl" --schema sampled --num-samples 4 --batch-size 16 --no-progress
log "SeeQ evaluate"
"$EVAL" scripts/vvqg/gqa_evaluate.py --predictions "$PB/seeq/evaluation/sanity50/predictions.jsonl" --refs "$SANREF" --system seeq --out "$PB/seeq/evaluation/sanity50" --concurrency 16
"$EVAL" scripts/vvqg/gqa_evaluate.py --predictions "$PB/seeq/evaluation/dev1000/predictions.jsonl" --refs "$DEVREF" --system seeq --out "$PB/seeq/evaluation/dev1000" --concurrency 16
else
log "SeeQ M2 missing; skipping SeeQ inference/evaluation"
fi

# ---------- Embedding diversity (SemDiv) ----------
to_div() { # $1 sampled4 file, $2 out
  .venv-vllm-cu128-v011/bin/python - "$1" "$2" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
out=open(sys.argv[2],'w')
for r in rows:
    out.write(json.dumps({"image_id":str(r["image_id"]),"generation_id":int(r.get("generation_id",r.get("sample_index",0))),"question":r["question"]})+"\n")
PY
}
for name in lova3 genixer; do
  [[ -f "$PB/$name/sampled4x.jsonl" ]] || { log "$name sampled4 missing; skip diversity"; continue; }
  to_div "$PB/$name/sampled4x.jsonl" "$PB/$name/evaluation/diversity_input.jsonl"
  [[ -f "$PB/$name/evaluation/diversity/summary.json" ]] || taskset -c 0-7 "$EVAL" scripts/vvqg/evaluate_seeq_diversity.py --model "$EMBED" --zero-shot "$PB/$name/evaluation/diversity_input.jsonl" --sft "$PB/$name/evaluation/diversity_input.jsonl" --output "$PB/$name/evaluation/diversity"
done
if [[ -f "$PB/seeq/evaluation/dev1000_sampled4.jsonl" ]]; then
  to_div "$PB/seeq/evaluation/dev1000_sampled4.jsonl" "$PB/seeq/evaluation/diversity_input.jsonl"
  [[ -f "$PB/seeq/evaluation/diversity/summary.json" ]] || taskset -c 0-7 "$EVAL" scripts/vvqg/evaluate_seeq_diversity.py --model "$EMBED" --zero-shot "$PB/seeq/evaluation/diversity_input.jsonl" --sft "$PB/seeq/evaluation/diversity_input.jsonl" --output "$PB/seeq/evaluation/diversity"
fi

log "assemble table"
"$EVAL" scripts/vvqg/gqa_baseline_table.py
log "ALL DONE"
