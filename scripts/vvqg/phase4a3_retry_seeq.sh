#!/usr/bin/env bash
# Phase 4A.3: retry failed SeeQ/support judge calls until coverage is complete,
# then regenerate each seed's normalized summary from the caches.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
RUN=outputs/phase4a3_rft_replay_multiseed
PY=.venv-vllm-cu128-v011/bin/python

for s in 43 44; do
  for attempt in 1 2 3 4 5 6 7 8; do
    "$PY" scripts/vvqg/phase4a2_evaluate.py --variant uvfc_violation_replay \
      --run-dir "$RUN/seed${s}" --retry --label "seed${s}" \
      > "$RUN/seed${s}/evaluate_retry.log" 2>&1
    failed=$("$PY" -c "import json;print(json.load(open('$RUN/seed${s}/evaluation/seeq/scores_runtime.json'))['failed'])")
    echo "seed${s} attempt ${attempt}: seeq failed=${failed}"
    [ "$failed" = "0" ] && break
    sleep 5
  done
done
echo "phase4a3 retry complete"
