#!/usr/bin/env bash
# Low-frequency monitor for the base-init GQA pipeline.  Appends a line to
# monitor.log on each stage transition or pipeline exit.  No polling of the GPU;
# purely file/process based.  Safe to run detached; idempotent state file.
set -uo pipefail
cd "$(dirname "$0")/../.."
BASE="outputs/gqa_native/full_ours_base"
NEW="outputs/gqa_native/baselines_10k"
LOG="$BASE/logs/monitor.log"
STATE="$BASE/logs/monitor.state"
mkdir -p "$BASE/logs"

cand_n() { wc -l < "$BASE/direct4/candidates.jsonl" 2>/dev/null || echo 0; }
scored_n() { wc -l < "$BASE/direct4/candidates_scored.jsonl" 2>/dev/null || echo 0; }

mark() { # name condition-cmd...
  local name="$1"; shift
  if eval "$@" 2>/dev/null; then
    if ! grep -qx "$name" "$STATE" 2>/dev/null; then
      echo "$name" >> "$STATE"
      echo "[monitor $(date +%F_%T)] DONE $name" >> "$LOG"
    fi
  fi
}

while true; do
  mark uvfc_weights   "[ -f $BASE/uvfc/weights.pt ]"
  mark direct4        "[ -s $BASE/direct4/candidates.jsonl ]"
  mark score          "[ \"\$(scored_n)\" -ge \"\$(cand_n)\" ] && [ \"\$(cand_n)\" -gt 0 ]"
  mark d1_selection   "[ -f $BASE/qse_teacher/selected/selection_audit.json ]"
  mark d1_targets     "[ -f $BASE/qse_teacher/targets_d1_single.jsonl ]"
  mark d1_adapter     "[ -f $BASE/distill_d1/checkpoint-final/adapter_config.json ]"
  mark g2_summary     "[ -f $BASE/g2/training_summary.json ]"
  mark full_ours_eval "[ -f $BASE/evaluation/dev1000/summary.json ]"
  mark b2_checkpoint  "[ -f $NEW/b2_contrastive/full_fast/backbone/adapter_model.safetensors ]"
  mark b3_checkpoint  "[ -f $NEW/b3_implicit/full_fast/guided_head.pt ]"
  mark b2_eval        "[ -f $NEW/b2_contrastive/evaluation/dev1000/summary.json ]"
  mark b3_eval        "[ -f $NEW/b3_implicit/evaluation/dev1000/summary.json ]"
  if ! pgrep -f "gqa_base_pipeline.sh" >/dev/null 2>&1 && ! grep -qx PIPELINE_EXIT "$STATE" 2>/dev/null; then
    echo PIPELINE_EXIT >> "$STATE"
    echo "[monitor $(date +%F_%T)] PIPELINE EXITED" >> "$LOG"
  fi
  sleep 180
done
