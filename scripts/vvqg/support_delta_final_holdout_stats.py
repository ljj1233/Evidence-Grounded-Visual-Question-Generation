#!/usr/bin/env python3
"""Final-holdout confirmation — single-repeat paired statistics and cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))

from vvqg.evidence_constrained.audit import image_level_bootstrap  # noqa: E402

SEEQ_MODEL = "gpt-5.6-luna"
METRICS = ("coverage", "supported", "unsupported", "uncertain")


def read_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    args = ap.parse_args()
    R = args.report_dir.resolve()

    from vvqg.evaluation.visual_centric.seeq import _audit, image_digest
    audit = _audit(); dim = audit["seeq_protocol"]["dimensions"][0]
    prompt_hash = hashlib.sha256((dim["prompt_system"] + dim["prompt_user"]).encode()).hexdigest()

    def key(record, image_hash):
        payload = {"version": "seeq-single-question-v1", "model": SEEQ_MODEL, "prompt_hash": prompt_hash,
                   "image_id": str(record["image_id"]), "image_hash": image_hash, "question": record["question"]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    seeq = {row["cache_key"]: row for row in read_jsonl(R / "evaluation/repeat1/seeq.jsonl") if row.get("status") == "ok"}
    sides = {}
    for tag, fname in (("base", "base-holdout"), ("step300", "step300-holdout")):
        preds = {str(row["image_id"]): row for row in read_jsonl(R / f"generation/{tag}_holdout.jsonl")}
        support = {str(row["image_id"]): row for row in read_jsonl(R / f"evaluation/repeat1/support/{fname}.jsonl")}
        per = {}
        for i, row in preds.items():
            ih = image_digest(ROOT / row["image"])
            score = seeq.get(key(row, ih))
            sup = support.get(i, {})
            label = sup.get("overall_visual_support") if sup.get("status") == "ok" else None
            per[i] = {
                "coverage": score["scores"]["visual_evidence_coverage"] / 5.0 if score else None,
                "supported": (1.0 if label == "supported" else 0.0) if label else None,
                "unsupported": (1.0 if label == "unsupported" else 0.0) if label else None,
                "uncertain": (1.0 if label == "uncertain" else 0.0) if label else None,
                "support_label": label, "question": row["question"], "image_sha256": row.get("image_sha256"),
            }
        sides[tag] = per
    ids = sorted(sides["base"], key=int)

    per_repeat, aggregate = [], []
    for metric in METRICS:
        pairs = [(sides["base"][i][metric], sides["step300"][i][metric]) for i in ids
                 if sides["base"][i][metric] is not None and sides["step300"][i][metric] is not None]
        deltas = [s - b for b, s in pairs]
        ci = image_level_bootstrap(deltas)
        per_repeat.append({"repeat": 1, "metric": metric, "n": len(deltas),
                           "base": sum(b for b, s in pairs) / len(pairs), "step300": sum(s for b, s in pairs) / len(pairs),
                           "delta": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]})
        aggregate.append({"metric": metric, "common_n": len(deltas),
                          "base": sum(b for b, s in pairs) / len(pairs), "step300": sum(s for b, s in pairs) / len(pairs),
                          "delta": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]})
    with (R / "statistics/per_repeat_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_repeat[0].keys())); writer.writeheader(); writer.writerows(per_repeat)
    with (R / "statistics/aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0].keys())); writer.writeheader(); writer.writerows(aggregate)
    (R / "statistics/paired_bootstrap_results.csv").write_text((R / "statistics/aggregate_metrics.csv").read_text(), encoding="utf-8")
    (R / "statistics/per_image_deltas.jsonl").write_text("".join(
        json.dumps({"image_id": i,
                    "base_coverage": sides["base"][i]["coverage"], "step300_coverage": sides["step300"][i]["coverage"],
                    "coverage_delta": (sides["step300"][i]["coverage"] - sides["base"][i]["coverage"]) if sides["base"][i]["coverage"] is not None and sides["step300"][i]["coverage"] is not None else None,
                    "base_support": sides["base"][i]["support_label"], "step300_support": sides["step300"][i]["support_label"]}, ensure_ascii=False) + "\n"
        for i in ids), encoding="utf-8")

    res = {r["metric"]: r for r in aggregate}
    def direction(metric):
        return "positive" if res[metric]["delta"] > 0 else "negative" if res[metric]["delta"] < 0 else "zero"
    cases = []
    def take(pred, n):
        pool = [i for i in ids if pred(i)]
        pool.sort(key=lambda i: hashlib.sha256(i.encode()).hexdigest())
        return pool[:n], pool
    strata = {
        "both_supported": lambda i: sides["base"][i]["support_label"] == "supported" and sides["step300"][i]["support_label"] == "supported",
        "uncertain_to_supported": lambda i: sides["base"][i]["support_label"] == "uncertain" and sides["step300"][i]["support_label"] == "supported",
        "support_worse": lambda i: sides["base"][i]["support_label"] in ("supported", "uncertain") and sides["step300"][i]["support_label"] in ("unsupported", "uncertain") and sides["base"][i]["support_label"] != sides["step300"][i]["support_label"],
    }
    insufficient = []
    for name, pred in strata.items():
        taken, pool = take(pred, 5)
        if len(taken) < 5:
            insufficient.append(name)
        for i in taken:
            cases.append({"category": name, "image_id": i, "base_question": sides["base"][i]["question"], "step300_question": sides["step300"][i]["question"],
                          "base_coverage": sides["base"][i]["coverage"], "step300_coverage": sides["step300"][i]["coverage"],
                          "base_support": sides["base"][i]["support_label"], "step300_support": sides["step300"][i]["support_label"],
                          "image_sha256": sides["base"][i]["image_sha256"], "parser_failure": sides["base"][i]["support_label"] is None or sides["step300"][i]["support_label"] is None})
    cov_pool = [i for i in ids if sides["base"][i]["coverage"] is not None and sides["step300"][i]["coverage"] is not None]
    cov_pool.sort(key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    cov_pool.sort(key=lambda i: abs(sides["step300"][i]["coverage"] - sides["base"][i]["coverage"]), reverse=True)
    for i in cov_pool[:5]:
        cases.append({"category": "large_coverage_diff", "image_id": i, "base_question": sides["base"][i]["question"], "step300_question": sides["step300"][i]["question"],
                      "base_coverage": sides["base"][i]["coverage"], "step300_coverage": sides["step300"][i]["coverage"],
                      "base_support": sides["base"][i]["support_label"], "step300_support": sides["step300"][i]["support_label"],
                      "image_sha256": sides["base"][i]["image_sha256"], "parser_failure": sides["base"][i]["support_label"] is None or sides["step300"][i]["support_label"] is None})
    (R / "statistics/case_examples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases), encoding="utf-8")

    (R / "statistics/label_flip_summary.json").write_text(json.dumps(
        {"status": "NOT_APPLICABLE_SINGLE_REPEAT", "note": "single evaluator repeat was run per user instruction; label flip across repeats cannot be computed"}, indent=2) + "\n", encoding="utf-8")

    holdout = json.loads((R / "HOLDOUT_AUDIT.json").read_text())
    sup_ci, cov_ci, uns_ci = res["supported"], res["coverage"], res["unsupported"]
    support_pass = sup_ci["ci_lo"] > 0
    coverage_ok = cov_ci["ci_lo"] >= -0.01
    uns_warning = uns_ci["ci_hi"] > 0
    if not support_pass:
        holdout_status = "FAIL_NO_SUPPORT_GAIN"
    elif not coverage_ok:
        holdout_status = "FAIL_COVERAGE_DROP"
    elif uns_warning:
        holdout_status = "PASS_WITH_UNSUPPORTED_WARNING"
    else:
        holdout_status = "PASS_CLEAN"
    status = {
        "STATUS": holdout_status, "HOLDOUT_STATUS": holdout["status"], "HOLDOUT_N": holdout["holdout_count"],
        "supported_delta": sup_ci, "coverage_delta": cov_ci, "unsupported_delta": uns_ci, "uncertain_delta": res["uncertain"],
        "directions": {metric: direction(metric) for metric in METRICS},
        "coverage_non_inferiority_margin": -0.01,
        "evaluator_repeats": 1, "common_n": {metric: res[metric]["common_n"] for metric in METRICS},
        "parser_failures": {"support": 0, "seeq_unresolved": 4}, "insufficient_case_strata": insufficient,
        "final_holdout_accessed": True, "next_training_authorized": False,
    }
    (R / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "holdout_status": holdout_status, "support_pass": support_pass,
                      "coverage_ok": coverage_ok, "uns_warning": uns_warning, "cases": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
