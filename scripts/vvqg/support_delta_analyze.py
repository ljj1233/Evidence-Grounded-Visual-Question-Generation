#!/usr/bin/env python3
"""Support-Delta GRPO — external pilot analysis, gate, and report."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))

from vvqg.evidence_constrained.audit import image_level_bootstrap  # noqa: E402

CONDITIONS = ("original", "blurred", "shuffled")
SEEQ_MODEL = "gpt-5.6-luna"


def read_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def seeq_key(record, prompt_hash, image_hash):
    import hashlib

    payload = {"version": "seeq-single-question-v1", "model": SEEQ_MODEL, "prompt_hash": prompt_hash,
               "image_id": str(record["image_id"]), "image_hash": image_hash, "question": record["question"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--pilot-eval", type=Path, required=True)
    args = ap.parse_args()
    R, E = args.report_dir, args.pilot_eval
    support_dir = E / "external/support_r1"

    from vvqg.evaluation.visual_centric.seeq import _audit
    import hashlib

    audit = _audit()
    dim = audit["seeq_protocol"]["dimensions"][0]
    prompt_hash = hashlib.sha256((dim["prompt_system"] + dim["prompt_user"]).encode()).hexdigest()
    seeq_scores = {row["cache_key"]: row["scores"]["visual_evidence_coverage"] for row in read_jsonl(E / "external/seeq_r1.jsonl") if row.get("status") == "ok"}

    image_hash_cache: dict[str, str] = {}
    def imghash(path: str) -> str:
        if path not in image_hash_cache:
            image_hash_cache[path] = hashlib.sha256((ROOT / path if not Path(path).is_absolute() else Path(path)).read_bytes()).hexdigest()
        return image_hash_cache[path]

    per_image: dict[tuple, dict] = {}
    systems = [f"{tag}-{condition}" for tag in ("step0", "step300") for condition in CONDITIONS]
    external_rows = []
    for system in systems:
        tag, condition = system.rsplit("-", 1)
        rows = read_jsonl(support_dir / f"{system}.jsonl")
        for row in rows:
            key = (tag, condition, str(row["image_id"]))
            coverage = seeq_scores.get(seeq_key(row, prompt_hash, imghash(row["image"])))
            per_image[key] = {
                "supported": row.get("overall_visual_support") == "supported",
                "unsupported": row.get("overall_visual_support") == "unsupported",
                "uncertain": row.get("overall_visual_support") == "uncertain",
                "support_status": row.get("status"),
                "coverage": (coverage / 5.0) if coverage is not None else None,
            }
        valid = [per_image[(tag, condition, str(row["image_id"]))] for row in rows]
        external_rows.append({
            "system": system, "tag": tag, "condition": condition, "n": len(rows),
            "supported_rate": sum(v["supported"] for v in valid) / len(valid),
            "unsupported_rate": sum(v["unsupported"] for v in valid) / len(valid),
            "uncertain_rate": sum(v["uncertain"] for v in valid) / len(valid),
            "coverage": sum(v["coverage"] for v in valid if v["coverage"] is not None) / max(sum(v["coverage"] is not None for v in valid), 1),
            "parser_failures": sum(v["support_status"] != "ok" for v in valid),
        })
    with (R / "EXTERNAL_PILOT_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(external_rows[0].keys())); writer.writeheader(); writer.writerows(external_rows)

    image_ids = sorted({key[2] for key in per_image})
    def paired(metric, condition):
        deltas = [per_image[("step300", condition, i)][metric] - per_image[("step0", condition, i)][metric]
                  for i in image_ids if per_image[("step300", condition, i)][metric] is not None and per_image[("step0", condition, i)][metric] is not None]
        return image_level_bootstrap([1.0 if d else 0.0 for d in deltas] if metric in ("supported", "unsupported") else deltas)
    def paired_indicator(metric, condition):
        deltas = [float(per_image[("step300", condition, i)][metric]) - float(per_image[("step0", condition, i)][metric]) for i in image_ids]
        return image_level_bootstrap(deltas)

    results = {
        "supported_original": paired_indicator("supported", "original"),
        "unsupported_original": paired_indicator("unsupported", "original"),
        "coverage_original": paired("coverage", "original"),
        "coverage_blurred": paired("coverage", "blurred"),
        "coverage_shuffled": paired("coverage", "shuffled"),
    }
    def gap_metric(tag, condition):
        return [float(per_image[(tag, "original", i)]["supported"]) - float(per_image[(tag, condition, i)]["supported"]) for i in image_ids]
    gap0_shuf, gap3_shuf = gap_metric("step0", "shuffled"), gap_metric("step300", "shuffled")
    gap0_blur, gap3_blur = gap_metric("step0", "blurred"), gap_metric("step300", "blurred")
    results["original_shuffled_support_gap_delta"] = image_level_bootstrap([a - b for a, b in zip(gap3_shuf, gap0_shuf)])
    results["original_blurred_support_gap_delta"] = image_level_bootstrap([a - b for a, b in zip(gap3_blur, gap0_blur)])
    results["original_shuffled_gap_step0"] = sum(gap0_shuf) / len(gap0_shuf)
    results["original_shuffled_gap_step300"] = sum(gap3_shuf) / len(gap3_shuf)

    with (R / "PAIRED_BOOTSTRAP_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["metric", "mean", "ci_lo", "ci_hi", "n"])
        for key, value in results.items():
            if isinstance(value, dict):
                writer.writerow([key, value["mean"], value["lo"], value["hi"], value["n"]])
    with (R / "INTERVENTION_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["gap", "step0", "step300", "delta_mean", "ci_lo", "ci_hi"])
        writer.writerow(["original_shuffled", results["original_shuffled_gap_step0"], results["original_shuffled_gap_step300"], results["original_shuffled_support_gap_delta"]["mean"], results["original_shuffled_support_gap_delta"]["lo"], results["original_shuffled_support_gap_delta"]["hi"]])

    # local diagnostics from CHECKPOINT_METRICS.csv
    local = {row["tag"] + "-" + row["condition"]: row for row in csv.DictReader((E / "CHECKPOINT_METRICS.csv").open())}
    train = json.loads((R / "pilot_final/training_summary.json").read_text())
    local_orig0, local_orig3 = local["step0-original"], local["step300-original"]
    checks = {
        "supported_point_ge_0.015": results["supported_original"]["mean"] >= 0.015,
        "supported_ci_lower_gt_0": results["supported_original"]["lo"] > 0,
        "coverage_ci_lower_ge_-0.01": results["coverage_original"]["lo"] >= -0.01,
        "unsupported_point_lt_0": results["unsupported_original"]["mean"] < 0,
        "unsupported_ci_upper_le_0": results["unsupported_original"]["hi"] <= 0,
        "orig_shuffled_gap_increase": results["original_shuffled_support_gap_delta"]["mean"] > 0,
        "duplicate_increase_le_0.05": float(local_orig3["duplicate_rate_global"]) - float(local_orig0["duplicate_rate_global"]) <= 0.05,
        "generic_increase_le_0.03": float(local_orig3["generic_rate"]) - float(local_orig0["generic_rate"]) <= 0.03,
        "length_change_le_10pct": abs(float(local_orig3["mean_token_length"]) - float(local_orig0["mean_token_length"])) / float(local_orig0["mean_token_length"]) <= 0.10,
        "kl_le_0.01": train["kl_range"][1] <= 0.01,
        "ratio_not_identity": train["ratio_mean_range"][0] != train["ratio_mean_range"][1],
        "clip_fraction_positive": train["clip_fraction_range"][1] > 0,
        "evaluator_repeat_3": False,
        "no_final_holdout": True,
    }
    status = "PASS_PILOT" if all(checks.values()) else "FAIL_PILOT"
    (R / "STATUS.json").write_text(json.dumps({
        "status": status, "checks": checks, "results": results,
        "local_original": {"step0": local_orig0, "step300": local_orig3},
        "training": {"steps": train["steps"], "stop_reason": train["stop_reason"], "ratio_mean_range": train["ratio_mean_range"],
                      "clip_fraction_range": train["clip_fraction_range"], "kl_range": train["kl_range"]},
        "formal_started": False, "final_holdout_accessed": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "results": results, "local_original_step0": local_orig0, "local_original_step300": local_orig3}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
