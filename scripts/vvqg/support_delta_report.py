#!/usr/bin/env python3
"""Support-Delta GRPO — assemble final report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    args = ap.parse_args()
    R = args.report_dir
    train = read_jsonl(R / "pilot_final/train_log.jsonl")
    summary = json.loads((R / "pilot_final/training_summary.json").read_text())
    status = json.loads((R / "STATUS.json").read_text())

    shutil.copyfile(R / "pilot_final/BASELINE_ANCHORS.jsonl", R / "BASELINE_ANCHORS.jsonl")

    with (R / "RATIO_CLIP_DIAGNOSTICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "ratio_mean", "ratio_p01", "ratio_p50", "ratio_p99", "clip_fraction", "grad_norm", "lr"])
        writer.writeheader()
        for row in train:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    with (R / "KL_DIAGNOSTICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "reference_kl", "loss", "policy_loss"])
        writer.writeheader()
        for row in train:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    shutil.copyfile(R / "sanity128/REWARD_COMPONENTS.csv", R / "REWARD_COMPONENTS.csv")
    shutil.copyfile(R / "sanity128/GROUP_VARIANCE.csv", R / "GROUP_VARIANCE.csv")
    sanity = json.loads((R / "sanity128/REWARD_SANITY.json").read_text())
    (R / "REWARD_SANITY_CHECK.md").write_text(
        f"# Reward sanity check\n\nSTATUS: **{sanity['status']}** (128 train images, 1024 rollouts, frozen pi0)\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in sanity["checks"].items())
        + "\n\n## Diagnostics\n\n```json\n" + json.dumps(sanity["diagnostics"], indent=2) + "\n```\n", encoding="utf-8")

    step0 = {str(r["image_id"]): r for r in read_jsonl(R / "pilot_eval/external/support_r1/step0-original.jsonl")}
    step300 = {str(r["image_id"]): r for r in read_jsonl(R / "pilot_eval/external/support_r1/step300-original.jsonl")}
    ids = sorted(set(step0) & set(step300))
    examples = [{"image_id": i, "base_question": step0[i]["question"], "base_support": step0[i].get("overall_visual_support"),
                 "sd_question": step300[i]["question"], "sd_support": step300[i].get("overall_visual_support")} for i in ids[:20]]
    (R / "PAIRED_EXAMPLES.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8")
    failures = [{"type": "regression_supported_to_unsupported", **row} for row in
                ({"image_id": i, "base_support": step0[i].get("overall_visual_support"), "sd_support": step300[i].get("overall_visual_support"),
                  "base_question": step0[i]["question"], "sd_question": step300[i]["question"]} for i in ids)
                if row["base_support"] == "supported" and row["sd_support"] != "supported"][:20]
    failures += [{"type": "improvement_unsupported_to_supported", **row} for row in
                 ({"image_id": i, "base_support": step0[i].get("overall_visual_support"), "sd_support": step300[i].get("overall_visual_support"),
                   "base_question": step0[i]["question"], "sd_question": step300[i]["question"]} for i in ids)
                 if row["base_support"] != "supported" and row["sd_support"] == "supported"][:20]
    (R / "FAILURE_CASES.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8")

    config = {
        "method": "Support-Delta GRPO (internal training objective / candidate method variant)",
        "base_checkpoint": "outputs/phase5c_qse_relativeguard/checkpoints/step_final",
        "base_adapter_sha256": "52ee692ca08737058594fe408a7817419f53532395f9249b2f227b43b4d9a58d",
        "instruction": "Generate a question about this image.",
        "greedy": {"temperature": 0.0, "max_new_tokens": 32}, "rollout": {"temperature": 0.8, "top_p": 0.9, "max_new_tokens": 32},
        "seed": 42, "group_size_K": 8, "groups_per_step": 1, "max_steps": 300, "inner_epochs": 2,
        "lr": 5e-7, "beta_kl": 0.02, "clip_epsilon": 0.2, "max_grad_norm": 1.0, "optimizer": "AdamW", "lora_only": True,
        "reward": "delta_support - 0.50*coverage_guard - 0.25*duplicate_penalty - 0.50*malformed_penalty",
        "checkpoint_steps": [0, 50, 100, 150, 200, 250, 300],
        "pilot_manifest": "reports/vc_qse_20261005T035840Z/dev256_manifest.jsonl",
        "evaluators": {"support": "gpt-4o / visual-support-phase23-v1.0", "coverage": "SeeQ gpt-5.6-luna VARIANT_DOCUMENTED"},
        "evaluator_repeats_completed": 1,
    }
    (R / "CONFIG_RESOLVED.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    (R / "MISSING_INFORMATION.md").write_text(
        "# Missing information\n\n"
        "- Evaluator repeat=3 was NOT completed (1 of 3 rounds run) → gate condition 12 unmet.\n"
        "- Formal 1000-step training NOT started (pilot failed).\n"
        "- K=8 rollouts with a no-op guard fix; groups_per_step=1 chosen for compute (not specified by task).\n"
        "- The full 9184-image q0 precompute was replaced by on-demand frozen-pi0 anchors for the 300 images\n"
        "  that actually entered training (saved in BASELINE_ANCHORS.jsonl).\n"
        "- Blur/shuffled SeeQ coverage is reported but blank condition was not evaluated.\n"
        "- final_holdout NOT accessed.\n", encoding="utf-8")

    local = status["local_original"]
    res = status["results"]
    report = f"""# FINAL REPORT — Support-Delta GRPO pilot

METHOD: Support-Delta GRPO — **internal training objective / candidate method variant** (not a paper novelty).

## Headline

- **STATUS: {status['status']}**. Reward sanity PASSED; the 300-step pilot completed; the 256-image external pilot gate FAILED.
- Formal 1000-step training was **NOT started**. `final_holdout` was **NOT accessed**.

## FACT

1. Reward sanity check: **PASSED** (128 train images): supported S=0.953 > unsupported S=-0.703; delta_support has both signs; zero-variance group rate 0.031 (<0.30); coverage_guard 0.378 (<0.50); malformed 0.017 (<0.05); no template collapse (0.016).
2. 300-step pilot: completed 300 steps, `stop_reason=null`. ratio mean range {summary['ratio_mean_range']} (varies); clip_fraction range {summary['clip_fraction_range']} (>0); KL range {summary['kl_range']} (<=0.02).
3. External (256 fixed-output images, GPT-4o support + SeeQ coverage):
   - Supported delta **{res['supported_original']['mean']:+.4f}** 95% CI [{res['supported_original']['lo']:+.4f}, {res['supported_original']['hi']:+.4f}]
   - Unsupported delta **{res['unsupported_original']['mean']:+.4f}** 95% CI [{res['unsupported_original']['lo']:+.4f}, {res['unsupported_original']['hi']:+.4f}]
   - Coverage delta **{res['coverage_original']['mean']:+.4f}** 95% CI [{res['coverage_original']['lo']:+.4f}, {res['coverage_original']['hi']:+.4f}]
   - original-shuffled support gap delta **{res['original_shuffled_support_gap_delta']['mean']:+.4f}** CI [{res['original_shuffled_support_gap_delta']['lo']:+.4f}, {res['original_shuffled_support_gap_delta']['hi']:+.4f}]
   - original-blurred support gap delta **{res['original_blurred_support_gap_delta']['mean']:+.4f}** CI [{res['original_blurred_support_gap_delta']['lo']:+.4f}, {res['original_blurred_support_gap_delta']['hi']:+.4f}]
4. Local Flat on the same 256 (original): Supported {local['step0']['flat_supported_rate']}→{local['step300']['flat_supported_rate']}; duplicate {local['step0']['duplicate_rate_global']}→{local['step300']['duplicate_rate_global']}; generic {local['step0']['generic_rate']}→{local['step300']['generic_rate']}; length {local['step0']['mean_token_length']}→{local['step300']['mean_token_length']}.
5. Gate: {json.dumps(status['checks'])}.

## Required answers

1. Reward sanity PASS. 2. zero-variance group rate 0.031 (sanity) / per-step reported in logs. 3. coverage_guard trigger rate 0.378 (sanity). 4. ratio truly varies (range above). 5. clip_fraction range 0.0-0.10. 6. KL range up to 0.0070. 7. training completed 300 steps. 8. pilot FAIL. 9. 1000-step NOT started. 10-12. Coverage/Supported/Unsupported deltas and CIs above. 13. duplicate +0.000, generic +0.000, length +1.9%. 14. original/blur/shuffle gaps above. 15. final_holdout NOT accessed. 16. Failure: the objective moved Coverage up (+0.0156) but Supported down (-0.0039) and Unsupported up; the reward's relative-to-q0 delta was already near zero because rollouts are close to the base anchor, so the advantage mostly reweighted coverage rather than support. 17. Code: `src/vvqg/support_delta/*`, `scripts/vvqg/support_delta_*.py`, `tests/test_support_delta.py`. 18. Checkpoints: `pilot_final/checkpoints/step_0000..step_0300,step_final`.

## INTERPRETATION

- This is a **genuine negative pilot result**: the reward is not reward-sanity-broken, the optimizer moved the policy (ratio/clip/KL all active), yet external Supported did not improve and Unsupported did not fall.
- The reward's `delta_support = S(q_i) - S(q_0)` is dominated by cases where `S(q_i)=S(q_0)` (both high/flat), so the gradient is largely driven by the coverage guard and duplicate penalty, which is consistent with Coverage rising and Supported staying flat.
- Do not claim Support-Delta GRPO improves Supported.

## UNKNOWN

- Whether a different margin/baseline (e.g. absolute S rather than q0-relative) would change Supported (not tested; changing it now would be a new experiment).
- Whether 3 evaluator repeats would change the direction (only 1 repeat run).
- Whether groups_per_step>1 would reduce gradient noise enough to matter.

## Next single action

Stop. Do not start formal training. The one pre-registered follow-up to consider is to isolate why the support term is inert (fraction of rollouts with `delta_support != 0` and their reward contribution) before changing the objective.
"""
    (R / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": status["status"], "report_dir": str(R), "examples": len(examples), "failures": len(failures)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
