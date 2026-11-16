#!/bin/bash
# API evaluation queue of the probing route (cascade labels for candidate runs whose answering is done). No GPU needed.
# Usage: scripts/api_queue.sh RUN [RUN...]        e.g. qwen3vl8b-cand-pilot-v2 qwen3vl8b-variants-v2
# Budget guard: the estimated calls of a run (2.5 per distinct question) plus the ledger so far must stay below CAP (default 9000 of the 10,000 total).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
CAP=${CAP:-9000}
Q=outputs/gpu-queue
mkdir -p "$Q/logs"

for run in "$@"; do
  ev="$run-eval"
  if [ -f "$Q/$run.api.done" ]; then echo "skip  $run"; continue; fi
  if [ ! -f "$Q/$run.srcrun.done" ]; then echo "answering of $run is not finished; run the GPU queue first"; exit 1; fi
  est=$($PY -c "import sys;print(int(sum(1 for _ in open('outputs/'+sys.argv[1]+'-src/materials.jsonl'))*5/2))" "$run")
  spent=$($PY -m egvqg api-ledger | $PY -c "import json,sys;print(json.load(sys.stdin)['calls_total'])")
  echo "[$(date +%H:%M:%S)] $run: estimated $est calls, spent so far $spent, cap $CAP"
  if [ $((spent + est)) -gt "$CAP" ]; then echo "GUARD: budget cap would be exceeded; stopping"; exit 3; fi
  $PY -m egvqg simple-baseline-eval-init --run-id "$ev" --source-run "$run-src" --cascade > "$Q/logs/$run.evinit.log" 2>&1 || { echo "eval-init failed"; exit 1; }
  $PY -m egvqg simple-baseline-eval-run --run-id "$ev" > "$Q/logs/$run.evrun.log" 2>&1 || { echo "eval-run failed (resume by calling again)"; exit 1; }
  $PY -m egvqg candidate-labels --run-id "$run" --source-run "$run-src" --eval-run "$ev" > "$Q/logs/$run.labels.log" 2>&1 || { echo "labels failed"; exit 1; }
  touch "$Q/$run.api.done"
  echo "[$(date +%H:%M:%S)] done $run"
done
$PY -m egvqg api-ledger | tail -6
