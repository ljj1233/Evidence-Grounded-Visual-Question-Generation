#!/bin/bash
# GPU work queue of the probing route under the pinned stack (transformers 5.22.2, project .venv) and the neutral prompt plain-v2
# (all runs use the fixed prompt plain-v2). One stream per call; streams A and B may run
# side by side when the card has room. Every step writes a marker, so a restart only does what is missing.
#   A  Qwen3-VL-8B main data: pilot, expansion, distances, answering, geometry
#   B  breadth: Qwen3.5 smoke, context variants, MultiModalQA, Qwen3.5 pilot (second model)
#   C  steering screen (needs labels of A: run scripts/api_queue.sh first); C2 = answering for the survivors
# Usage: scripts/gpu_queue.sh A|B|C|C2        Only one card is used (GPU env var, default 0); a step waits until at least 30 GB of it is free.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=${GPU:-0} UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv-egvqg}
PY=.venv/bin/python
M3=/home/lijunjian/models/Qwen3-VL-8B-Instruct
M35=/home/lijunjian/models/Qwen3.5-9B
VAR=data/processed/scienceqa/context-variants-v1/materials.jsonl
MM=data/processed/multimodalqa/mmqa-pool-v1/materials.jsonl
Q=outputs/gpu-queue
mkdir -p "$Q/logs"

wait_gpu() {
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-0})" -lt 30000 ]; do sleep 120; done
}

step() {   # step NAME COMMAND...
  local name=$1; shift
  if [ -f "$Q/$name.done" ]; then echo "skip  $name"; return 0; fi
  wait_gpu
  echo "[$(date +%H:%M:%S)] start $name"
  if "$@" > "$Q/logs/$name.log" 2>&1; then touch "$Q/$name.done"; echo "[$(date +%H:%M:%S)] done  $name"
  else echo "[$(date +%H:%M:%S)] FAILED $name (log: $Q/logs/$name.log)"; exit 1; fi
}

chain() {  # chain RUN GENERATOR_MODEL "MATERIALS_ARGS" CAPTURE_ARGS...  -> capture, distances, source (local answering by Qwen3-VL-8B)
  local run=$1 model=$2 mat=$3; shift 3
  step "$run.capture"   $PY -m egvqg candidates-capture --run-id "$run" --local-model-path "$model" $mat "$@"
  step "$run.distances" $PY -m egvqg modality-distances --run-id "$run" --local-model-path "$model" $mat
  step "$run.srcinit"   $PY -m egvqg candidate-source-init --run-id "$run-src" --capture-run "$run" $mat
  step "$run.srcrun"    $PY -m egvqg candidate-source-run --run-id "$run-src" --capture-run "$run" --local-model-path "$M3"
}

case "${1:-}" in
A)
  chain qwen3vl8b-cand-pilot-v2        "$M3" "" --n-skills 20 --per-skill 3
  chain qwen3vl8b-cand-expand-v2       "$M3" "" --n-skills 40 --per-skill 8 --exclude-run qwen3vl8b-cand-pilot-v2
  step geometry-pilot-v2 $PY -m egvqg geometry-run --run-id qwen3vl8b-geometry-pilot-v2 --capture-run qwen3vl8b-cand-pilot-v2 --local-model-path "$M3"
  ;;
B)
  step q35.p0 $PY -m egvqg whitebox-capture --run-id qwen35-whitebox-p0-v1 --local-model-path "$M35" --limit 8
  chain qwen3vl8b-variants-v2 "$M3" "--materials-file $VAR" --materials-file "$VAR" --select all
  chain qwen3vl8b-mmqa-v2     "$M3" "--materials-file $MM"  --materials-file "$MM" --select sample --max-materials 60
  chain qwen35-cand-pilot-v1  "$M35" "" --n-skills 20 --per-skill 3
  ;;
C)
  T=qwen3vl8b-cand-pilot-v2:labels_qwen3vl8b-cand-pilot-v2-eval.jsonl,qwen3vl8b-cand-expand-v2:labels_qwen3vl8b-cand-expand-v2-eval.jsonl
  S=outputs/qwen3vl8b-steer-v1/steer
  step steer.prepare $PY -m egvqg steer-prepare --run-id qwen3vl8b-steer-v1 --training "$T" --layers 12,20,28 --alphas 1,2 --heldout 8
  CAP="--materials-file $S/heldout_materials.jsonl --select sample --max-materials 16 --n-candidates 4"
  step steer.ctrl.capture $PY -m egvqg candidates-capture --run-id qwen3vl8b-steer-ctrl --local-model-path "$M3" $CAP
  for spec in $S/spec_*.json; do
    cond=$(basename "$spec" .json); cond=${cond#spec_}
    step "steer.$cond.capture" $PY -m egvqg candidates-capture --run-id "qwen3vl8b-steer-$cond" --local-model-path "$M3" $CAP --steer-spec "$spec"
  done
  ALL="ctrl,$(ls $S | sed -n 's/^spec_\(.*\)\.json$/\1/p' | paste -sd,)"
  step steer.screen $PY -m egvqg steer-analyze --stage screen --prefix qwen3vl8b-steer --conditions "$ALL"
  echo "screening written to $Q/logs/steer.screen.log"
  ;;
C2)
  # answering for the control and the screening survivors (plus all random controls of surviving real conditions)
  SURV=$($PY - <<'EOF'
import json, re
log = open('outputs/gpu-queue/logs/steer.screen.log').read()
res = json.loads(log[log.index('{'):])
keep = {'ctrl'} | {c for c, v in res.items() if v.get('survives')}
for c in list(keep):
    m = re.match(r'dir[PN]_(L\d+_a[\d.]+)', c)
    if m:
        keep |= {f'randM_{m.group(1)}', f'gauss_{m.group(1)}'}
print(','.join(sorted(keep)))
EOF
)
  echo "answering for: $SURV"
  for cond in ${SURV//,/ }; do
    run=qwen3vl8b-steer-$cond
    CAPARGS="--materials-file outputs/qwen3vl8b-steer-v1/steer/heldout_materials.jsonl"
    step "$run.srcinit" $PY -m egvqg candidate-source-init --run-id "$run-src" --capture-run "$run" $CAPARGS
    step "$run.srcrun"  $PY -m egvqg candidate-source-run  --run-id "$run-src" --capture-run "$run" --local-model-path "$M3"
  done
  echo "$SURV" > "$Q/steer_survivors.txt"
  ;;
*) echo "usage: $0 A|B|C|C2"; exit 2 ;;
esac
echo "stream ${1} finished"
