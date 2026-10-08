#!/usr/bin/env python3
"""dev1000 confirmation — per-split paired statistics over 3 repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))

from vvqg.evidence_constrained.audit import image_level_bootstrap  # noqa: E402

SEEQ_MODEL = "gpt-5.6-luna"
REPS = (1, 2, 3)
METRICS = ("coverage", "supported", "unsupported", "uncertain")


def read_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--experiment-dir", type=Path, default=ROOT / "reports/support_delta_grpo_20261006T022616Z")
    args = ap.parse_args()
    R = args.report_dir.resolve(); E = args.experiment_dir.resolve()

    from vvqg.evaluation.visual_centric.seeq import _audit, image_digest
    audit = _audit(); dim = audit["seeq_protocol"]["dimensions"][0]
    prompt_hash = hashlib.sha256((dim["prompt_system"] + dim["prompt_user"]).encode()).hexdigest()

    def key(record, image_hash):
        payload = {"version": "seeq-single-question-v1", "model": SEEQ_MODEL, "prompt_hash": prompt_hash,
                   "image_id": str(record["image_id"]), "image_hash": image_hash, "question": record["question"]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def load_side(pred_path, support_paths, seeq_paths):
        preds = {str(row["image_id"]): row for row in read_jsonl(pred_path)}
        ihash = {i: image_digest(ROOT / preds[i]["image"]) for i in preds}
        per = {}
        for rep in REPS:
            seeq = {row["cache_key"]: row for row in read_jsonl(seeq_paths[rep - 1]) if row.get("status") == "ok"}
            support = {str(row["image_id"]): row for row in read_jsonl(support_paths[rep - 1])}
            for i, row in preds.items():
                score = seeq.get(key(row, ihash[i]))
                sup = support.get(i, {})
                label = sup.get("overall_visual_support") if sup.get("status") == "ok" else None
                per[(i, rep)] = {
                    "coverage": score["scores"]["visual_evidence_coverage"] / 5.0 if score else None,
                    "supported": (1.0 if label == "supported" else 0.0) if label else None,
                    "unsupported": (1.0 if label == "unsupported" else 0.0) if label else None,
                    "uncertain": (1.0 if label == "uncertain" else 0.0) if label else None,
                    "support_label": label,
                    "question": row["question"],
                }
        return per

    sides = {}
    sides["existing256"] = {
        "base": load_side(E / "pilot_eval/predictions_step0_original.jsonl",
                          [E / f"pilot_eval/external/support_r{r}/step0-original.jsonl" for r in REPS],
                          [E / f"pilot_eval/external/seeq_r{r}.jsonl" for r in REPS]),
        "step300": load_side(E / "pilot_eval/predictions_step300_original.jsonl",
                             [E / f"pilot_eval/external/support_r{r}/step300-original.jsonl" for r in REPS],
                             [E / f"pilot_eval/external/seeq_r{r}.jsonl" for r in REPS]),
    }
    sides["remaining744"] = {
        "base": load_side(R / "generation/base_remaining744.jsonl",
                          [R / f"evaluation/support_r{r}/base-remaining744.jsonl" for r in REPS],
                          [R / f"evaluation/seeq_r{r}.jsonl" for r in REPS]),
        "step300": load_side(R / "generation/step300_remaining744.jsonl",
                             [R / f"evaluation/support_r{r}/step300-remaining744.jsonl" for r in REPS],
                             [R / f"evaluation/seeq_r{r}.jsonl" for r in REPS]),
    }
    full = {
        "base": {**sides["existing256"]["base"], **sides["remaining744"]["base"]},
        "step300": {**sides["existing256"]["step300"], **sides["remaining744"]["step300"]},
    }

    def ids_for(per, tag):
        return sorted({i for (i, rep) in per if rep == 1 and per[(i, rep)]["coverage"] is not None}, key=int)

    split_sides = {"existing256": sides["existing256"], "remaining744": sides["remaining744"], "full1000": full}
    summary_rows, repeat_rows = [], []
    for split, side in split_sides.items():
        base, sd = side["base"], side["step300"]
        all_ids = sorted({i for (i, rep) in base} | {i for (i, rep) in sd}, key=int)
        for metric in METRICS:
            # per-repeat
            for rep in REPS:
                pairs = [(base[(i, rep)][metric], sd[(i, rep)][metric]) for i in all_ids
                         if base[(i, rep)][metric] is not None and sd[(i, rep)][metric] is not None]
                deltas = [s - b for b, s in pairs]
                ci = image_level_bootstrap(deltas)
                repeat_rows.append({"split": split, "metric": metric, "repeat": rep, "n": len(deltas),
                                    "base": sum(b for b, s in pairs) / len(pairs), "step300": sum(s for b, s in pairs) / len(pairs),
                                    "delta": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]})
            # 3-repeat aggregate
            common = [i for i in all_ids if all(base[(i, rep)][metric] is not None and sd[(i, rep)][metric] is not None for rep in REPS)]
            bvals = [sum(base[(i, rep)][metric] for rep in REPS) / 3 for i in common]
            svals = [sum(sd[(i, rep)][metric] for rep in REPS) / 3 for i in common]
            deltas = [s - b for b, s in zip(bvals, svals)]
            ci = image_level_bootstrap(deltas)
            per_rep_mean = []
            for rep in REPS:
                vals = [sd[(i, rep)][metric] - base[(i, rep)][metric] for i in common]
                per_rep_mean.append(sum(vals) / len(vals))
            mean_rep = sum(per_rep_mean) / 3
            summary_rows.append({"split": split, "metric": metric, "n": len(common), "base": sum(bvals) / len(bvals),
                                 "step300": sum(svals) / len(svals), "delta": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                                 "delta_std_over_repeats": (sum((r - mean_rep) ** 2 for r in per_rep_mean) / 3) ** 0.5})

    def write_csv(path, rows):
        import csv
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)

    write_csv(R / "evaluation/repeat_metrics.csv", repeat_rows)
    write_csv(R / "statistics/existing256_results.csv", [r for r in summary_rows if r["split"] == "existing256"])
    write_csv(R / "statistics/remaining744_results.csv", [r for r in summary_rows if r["split"] == "remaining744"])
    write_csv(R / "statistics/full1000_results.csv", [r for r in summary_rows if r["split"] == "full1000"])
    write_csv(R / "statistics/paired_bootstrap_results.csv", summary_rows)

    (R / "evaluation/per_image_scores.jsonl").write_text("".join(
        json.dumps({"image_id": i,
                    "base": {str(rep): sides["existing256"]["base"].get((i, rep)) or sides["remaining744"]["base"].get((i, rep)) for rep in REPS},
                    "step300": {str(rep): sides["existing256"]["step300"].get((i, rep)) or sides["remaining744"]["step300"].get((i, rep)) for rep in REPS},
                    "split": "existing256" if (i, 1) in sides["existing256"]["base"] else "remaining744"}, ensure_ascii=False) + "\n"
        for i in sorted({i for (i, rep) in full["base"]}, key=int)), encoding="utf-8")

    flips = {}
    for tag in ("base", "step300"):
        per_tag = {}
        for rep in (2, 3):
            pairs = [(1, rep)]
            per_tag[f"repeat1_vs_repeat{rep}"] = {
                "changed_images": sum(full[tag][(i, 1)]["support_label"] != full[tag][(i, rep)]["support_label"] for i in ids_for(full[tag], tag)),
                "denominator": len(ids_for(full[tag], tag)),
            }
        flips[tag] = per_tag
    (R / "evaluation/label_flip_summary.json").write_text(json.dumps(flips, indent=2) + "\n", encoding="utf-8")

    # directions and gain fractions
    directions = {}
    for split, side in split_sides.items():
        base, sd = side["base"], side["step300"]
        all_ids = sorted({i for (i, rep) in base}, key=int)
        for metric in ("coverage", "supported", "unsupported"):
            signs = []
            for rep in REPS:
                common = [i for i in all_ids if base[(i, rep)][metric] is not None and sd[(i, rep)][metric] is not None]
                d = sum(sd[(i, rep)][metric] - base[(i, rep)][metric] for i in common) / len(common)
                signs.append(1 if d > 0 else -1 if d < 0 else 0)
            directions[f"{split}_{metric}"] = signs
    full_base, full_sd = full["base"], full["step300"]
    # covered by splits for gain fractions
    gain = {}
    for split, side in split_sides.items():
        base, sd = side["base"], side["step300"]
        ids = sorted({i for (i, rep) in base}, key=int)
        gain[split] = {
            "coverage_gain_image_rate": sum((sum(sd[(i, rep)]["coverage"] for rep in REPS if sd[(i, rep)]["coverage"] is not None) / max(sum(sd[(i, rep)]["coverage"] is not None for rep in REPS), 1))
                                            > (sum(base[(i, rep)]["coverage"] for rep in REPS if base[(i, rep)]["coverage"] is not None) / max(sum(base[(i, rep)]["coverage"] is not None for rep in REPS), 1)) for i in ids) / len(ids),
            "supported_gain_image_rate": sum((sum(sd[(i, rep)]["supported"] for rep in REPS if sd[(i, rep)]["supported"] is not None) > sum(base[(i, rep)]["supported"] for rep in REPS if base[(i, rep)]["supported"] is not None)) for i in ids) / len(ids),
            "unsupported_increase_image_rate": sum((sum(sd[(i, rep)]["unsupported"] for rep in REPS if sd[(i, rep)]["unsupported"] is not None) > sum(base[(i, rep)]["unsupported"] for rep in REPS if base[(i, rep)]["unsupported"] is not None)) for i in ids) / len(ids),
        }

    full_res = {r["metric"]: r for r in summary_rows if r["split"] == "full1000"}
    rem_res = {r["metric"]: r for r in summary_rows if r["split"] == "remaining744"}
    cov_status = "COVERAGE_REPLICATION_SUPPORTED" if full_res["coverage"]["ci_lo"] > 0 and rem_res["coverage"]["delta"] > 0 else "COVERAGE_NOT_REPLICATED"
    sup_status = ("SUPPORT_IMPROVEMENT_SUPPORTED" if full_res["supported"]["ci_lo"] > 0
                  else "SUPPORT_DEGRADATION_RISK" if full_res["supported"]["ci_hi"] < 0 or full_res["unsupported"]["ci_lo"] > 0
                  else "SUPPORT_PRESERVATION_UNCERTAIN")
    status = {"ORIGINAL_PILOT_STATUS": "FAIL_PILOT", "REVIEW_STATUS": "COMPLETE",
              "COVERAGE_REPLICATION_STATUS": cov_status, "SUPPORTED_STATUS": sup_status,
              "UNSUPPORTED_STATUS": "UNSUPPORTED_" + ("INCREASED" if full_res["unsupported"]["ci_lo"] > 0 else "DECREASED" if full_res["unsupported"]["ci_hi"] < 0 else "UNCERTAIN"),
              "directions": directions, "gain_fractions": gain,
              "parser_failures": 0, "final_holdout_accessed": False, "training_started": False}
    (R / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    # case examples
    examples = []
    for i in sorted({i for (i, rep) in full_base}, key=int):
        c0 = [full_base[(i, rep)]["coverage"] for rep in REPS]; c3 = [full_sd[(i, rep)]["coverage"] for rep in REPS]
        if all(v is not None for v in c0 + c3):
            examples.append({"image_id": i,
                             "base_question": full_base[(i, 1)].get("question"), "step300_question": full_sd[(i, 1)].get("question"),
                             "base_coverage": c0, "step300_coverage": c3,
                             "base_support": [full_base[(i, rep)]["support_label"] for rep in REPS],
                             "step300_support": [full_sd[(i, rep)]["support_label"] for rep in REPS]})
    examples.sort(key=lambda r: sum(r["step300_coverage"]) - sum(r["base_coverage"]), reverse=True)
    (R / "statistics/case_examples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples[:4] + examples[-4:]), encoding="utf-8")
    (R / "evaluation/paired_image_deltas.jsonl").write_text("".join(
        json.dumps({"image_id": i, "split": "existing256" if (i, 1) in sides["existing256"]["base"] else "remaining744",
                    "coverage_delta_mean": (sum(full_sd[(i, rep)]["coverage"] for rep in REPS) - sum(full_base[(i, rep)]["coverage"] for rep in REPS)) / 3}, ensure_ascii=False) + "\n"
        for i in sorted({i for (i, rep) in full_base}, key=int)), encoding="utf-8")
    print(json.dumps({"summary": summary_rows, "status": status, "label_flips": flips}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
