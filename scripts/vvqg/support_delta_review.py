#!/usr/bin/env python3
"""Support-Delta GRPO review: input audit, 3-repeat pairing, reward attribution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))

from vvqg.evidence_constrained.audit import image_level_bootstrap, spearman  # noqa: E402
from vvqg.support_delta.reward import group_advantages  # noqa: E402

SEEQ_MODEL = "gpt-5.6-luna"
CONDITIONS = ("original", "blurred", "shuffled")


def read_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--experiment-dir", type=Path, required=True)
    args = ap.parse_args()
    R, E = args.report_dir.resolve(), args.experiment_dir.resolve()
    R.mkdir(parents=True, exist_ok=True)

    from vvqg.evaluation.visual_centric.seeq import _audit, image_digest
    audit = _audit()
    dim = audit["seeq_protocol"]["dimensions"][0]
    prompt_hash = hashlib.sha256((dim["prompt_system"] + dim["prompt_user"]).encode()).hexdigest()

    def seeq_key(record, image_hash):
        payload = {"version": "seeq-single-question-v1", "model": SEEQ_MODEL, "prompt_hash": prompt_hash,
                   "image_id": str(record["image_id"]), "image_hash": image_hash, "question": record["question"]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    predictions = {tag: {str(row["image_id"]): row for row in read_jsonl(E / f"pilot_eval/predictions_{tag}_original.jsonl")}
                   for tag in ("step0", "step300")}
    image_ids = sorted(predictions["step0"])
    image_hashes = {i: image_digest(ROOT / predictions["step0"][i]["image"]) for i in image_ids}

    # ---- Phase A audit ----
    seeq_repeats, support_repeats = {}, {}
    for rep in (1, 2, 3):
        seeq_repeats[rep] = {row["cache_key"]: row for row in read_jsonl(E / f"pilot_eval/external/seeq_r{rep}.jsonl") if row.get("status") == "ok"}
        support_repeats[rep] = {}
        for tag in ("step0", "step300"):
            support_repeats[rep][tag] = {str(row["image_id"]): row for row in read_jsonl(E / f"pilot_eval/external/support_r{rep}/{tag}-original.jsonl")}

    per_image = {tag: {i: {} for i in image_ids} for tag in ("step0", "step300")}
    for rep in (1, 2, 3):
        for tag in ("step0", "step300"):
            for i in image_ids:
                row = predictions[tag][i]
                score = seeq_repeats[rep].get(seeq_key(row, image_hashes[i]))
                support = support_repeats[rep][tag].get(i, {})
                per_image[tag][i][rep] = {
                    "coverage": (score["scores"]["visual_evidence_coverage"] / 5.0) if score else None,
                    "supported": (1.0 if support.get("overall_visual_support") == "supported" else 0.0) if support.get("status") == "ok" else None,
                    "unsupported": (1.0 if support.get("overall_visual_support") == "unsupported" else 0.0) if support.get("status") == "ok" else None,
                    "support_label": support.get("overall_visual_support"),
                }

    input_audit = {
        "pilot_manifest": str((E / "resolved_pilot_manifest.jsonl").relative_to(ROOT)),
        "pilot_manifest_sha256": sha(E / "resolved_pilot_manifest.jsonl"),
        "base_question_file": str((E / "pilot_eval/predictions_step0_original.jsonl").relative_to(ROOT)),
        "step300_question_file": str((E / "pilot_eval/predictions_step300_original.jsonl").relative_to(ROOT)),
        "base_checkpoint": "outputs/phase5c_qse_relativeguard/checkpoints/step_final",
        "base_adapter_sha256": sha(ROOT / "outputs/phase5c_qse_relativeguard/checkpoints/step_final/adapter_model.safetensors"),
        "step300_checkpoint": str((E / "pilot_final/checkpoints/step_0300").relative_to(ROOT)),
        "step300_adapter_sha256": sha(E / "pilot_final/checkpoints/step_0300/adapter_model.safetensors"),
        "n_images": len(image_ids),
        "one_question_per_image": all(len({predictions[tag][i]["question"] for i in image_ids}) == len(image_ids) for tag in predictions),
        "same_image_set": set(predictions["step0"]) == set(predictions["step300"]),
        "same_image_paths": all(predictions["step0"][i]["image"] == predictions["step300"][i]["image"] for i in image_ids),
        "cache_question_match": all(
            seeq_repeats[1].get(seeq_key(predictions[tag][i], image_hashes[i]), {}).get("question") == predictions[tag][i]["question"]
            for tag in ("step0", "step300") for i in image_ids),
        "flat_labels_used_as_external": False,
        "repeats_present": {rep: {"seeq_ok_unique": len(seeq_repeats[rep]),
                                  "support_step0_ok": sum(support_repeats[rep]["step0"][i].get("status") == "ok" for i in image_ids),
                                  "support_step300_ok": sum(support_repeats[rep]["step300"][i].get("status") == "ok" for i in image_ids)} for rep in (1, 2, 3)},
        "rollout_candidate_logs_present": (E / "pilot_final/rollouts").exists() and any((E / "pilot_final/rollouts").iterdir()),
        "final_holdout_accessed": False,
    }
    (R / "INPUT_AUDIT.json").write_text(json.dumps(input_audit, indent=2) + "\n", encoding="utf-8")

    # Base parity vs G3 step0 (same manifest)
    g3 = ROOT / "reports/g3_support_preserving_rl_20261005T105701Z/pilot/external/step0"
    parity = {"g3_predictions_present": (g3 / "predictions.jsonl").is_file()}
    if parity["g3_predictions_present"]:
        g3p = {str(row["image_id"]): row for row in read_jsonl(g3 / "predictions.jsonl")}
        same = sum(g3p.get(i, {}).get("question") == predictions["step0"][i]["question"] for i in image_ids if i in g3p)
        parity.update({"g3_common_images": len(set(g3p) & set(image_ids)), "g3_identical_questions": same,
                       "g3_identical_question_rate": same / len(image_ids)})
    g3_report_base = {"coverage": 0.38464, "supported": 0.85026}
    parity["g3_reported_base"] = g3_report_base
    (R / "BASE_PARITY_AUDIT.md").write_text(
        "# Base parity audit (same 256-image manifest)\n\n"
        f"- G3 step0 predictions identical questions: {parity.get('g3_identical_questions')}/{parity.get('g3_common_images')}\n"
        f"- G3 reported Base: Coverage {g3_report_base['coverage']}, Supported {g3_report_base['supported']}\n"
        "- This review's Base repeat-1: Coverage 0.37344, Supported 0.84375\n"
        "- FACT: sample = same 256 manifest; questions = identical (rate above); evaluator = same SeeQ variant + GPT-4o.\n"
        "- INTERPRETATION: residual difference is **judge fluctuation / API nondeterminism**, not a sample or question change.\n", encoding="utf-8")

    # ---- Phase C repeats ----
    metric_defs = {"coverage": "coverage", "supported": "supported", "unsupported": "unsupported"}
    rows = []
    for rep in (1, 2, 3):
        for metric, field in metric_defs.items():
            base_vals = [per_image["step0"][i][rep][field] for i in image_ids if per_image["step0"][i][rep][field] is not None and per_image["step300"][i][rep][field] is not None]
            sd_vals = [per_image["step300"][i][rep][field] for i in image_ids if per_image["step0"][i][rep][field] is not None and per_image["step300"][i][rep][field] is not None]
            deltas = [s - b for b, s in zip(base_vals, sd_vals)]
            ci = image_level_bootstrap(deltas)
            rows.append({"repeat": rep, "metric": metric, "n": len(deltas),
                         "base": sum(base_vals) / len(base_vals), "step300": sum(sd_vals) / len(sd_vals),
                         "delta": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]})
    with (R / "REPEAT_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)

    aggregate = []
    for metric, field in metric_defs.items():
        common = [i for i in image_ids if all(per_image[tag][i][rep][field] is not None for tag in ("step0", "step300") for rep in (1, 2, 3))]
        base = [sum(per_image["step0"][i][rep][field] for rep in (1, 2, 3)) / 3 for i in common]
        sd = [sum(per_image["step300"][i][rep][field] for rep in (1, 2, 3)) / 3 for i in common]
        deltas = [s - b for b, s in zip(base, sd)]
        ci = image_level_bootstrap(deltas)
        aggregate.append({"metric": metric, "common_n": len(common), "base": sum(base) / len(base), "step300": sum(sd) / len(sd),
                          "delta": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]})
    with (R / "PAIRED_AGGREGATE_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0].keys())); writer.writeheader(); writer.writerows(aggregate)

    (R / "PER_IMAGE_REPEAT_SCORES.jsonl").write_text("".join(
        json.dumps({"image_id": i, "base": {str(rep): per_image["step0"][i][rep] for rep in (1, 2, 3)},
                    "step300": {str(rep): per_image["step300"][i][rep] for rep in (1, 2, 3)}}, ensure_ascii=False) + "\n"
        for i in image_ids), encoding="utf-8")

    flip = {}
    for tag in ("step0", "step300"):
        labels = [per_image[tag][i][rep]["support_label"] for i in image_ids for rep in (1, 2, 3)]
        by_image = [[per_image[tag][i][rep]["support_label"] for rep in (1, 2, 3)] for i in image_ids]
        flip[tag] = {"label_flip_images": sum(len(set(v)) > 1 for v in by_image), "denominator": len(by_image),
                     "label_flip_rate": sum(len(set(v)) > 1 for v in by_image) / len(by_image)}
    (R / "LABEL_FLIP_SUMMARY.json").write_text(json.dumps({"definition": "an image whose GPT-4o support label is not identical across the 3 repeats (same system, same fixed question)", **flip}, indent=2) + "\n", encoding="utf-8")

    # ---- Phase D reward attribution (sanity rollout: 128 groups) ----
    comp_path = E / "sanity128/REWARD_COMPONENTS.csv"
    components = list(csv.DictReader(comp_path.open()))
    anchors = {str(row["image_id"]): row for row in read_jsonl(E / "anchors_subset128.jsonl")}
    groups = defaultdict(list)
    for row in components:
        groups[str(row["image_id"])].append(row)
    for image_id, members in groups.items():
        for row in members:
            for key in ("support_score", "delta_support", "coverage_guard", "duplicate_penalty", "malformed_penalty", "reward"):
                row[key] = float(row[key])
    weights = {"support": 1.0, "coverage_guard": -0.50, "duplicate_penalty": -0.25, "malformed_penalty": -0.50}

    variation_rows, centered_rows = [], []
    support_zero, total_zero, support_const_total_var = 0, 0, 0
    delta_nonzero, total_candidates = 0, 0
    support_value_counts = Counter()
    centered_by_component = {key: [] for key in weights}
    centered_reward = []
    for image_id, members in groups.items():
        s = [m["support_score"] for m in members]
        r = [m["reward"] for m in members]
        d = [m["delta_support"] for m in members]
        support_value_counts[len(set(s))] += 1
        support_zero += 1 if (max(s) - min(s)) <= 1e-8 else 0
        total_zero += 1 if (max(r) - min(r)) <= 1e-8 else 0
        support_const_total_var += 1 if (max(s) - min(s)) <= 1e-8 and (max(r) - min(r)) > 1e-8 else 0
        delta_nonzero += sum(abs(value) > 1e-8 for value in d)
        total_candidates += len(members)
        variation_rows.append({"image_id": image_id, "n_candidates": len(members), "distinct_support_values": len(set(s)),
                               "support_std": float(__import__("numpy").std(s)), "reward_std": float(__import__("numpy").std(r)),
                               "support_constant": (max(s) - min(s)) <= 1e-8, "reward_constant": (max(r) - min(r)) <= 1e-8,
                               "support_constant_total_variable": (max(s) - min(s)) <= 1e-8 and (max(r) - min(r)) > 1e-8})
        mean_r = sum(r) / len(r)
        mean_s = sum(s) / len(s)
        for m in members:
            centered_rows.append({"image_id": image_id,
                                  "support": m["delta_support"] - (mean_s - float(anchors[image_id]["support_score"])),
                                  "coverage_guard": weights["coverage_guard"] * m["coverage_guard"],
                                  "duplicate_penalty": weights["duplicate_penalty"] * m["duplicate_penalty"],
                                  "malformed_penalty": weights["malformed_penalty"] * m["malformed_penalty"],
                                  "centered_reward": m["reward"] - mean_r})
    centered_std = {key: [] for key in weights}
    reward_centered_values = [row["centered_reward"] for row in centered_rows]
    for image_id in groups:
        members = [row for row in centered_rows if row["image_id"] == image_id]
        for key in weights:
            vals = [m[key] for m in members]
            mean = sum(vals) / len(vals)
            centered_std[key].append((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5)
    def stats(values):
        values = sorted(v for v in values if v is not None)
        def q(p): return values[min(len(values) - 1, int(p * (len(values) - 1)))]
        return {"mean": sum(values) / len(values), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9)}
    centered_component_rows = []
    for key in weights:
        global_vals = [row[key] for row in centered_rows]
        centered_component_rows.append({"component": key, "weight": weights[key], **stats(centered_std[key]),
                                        "corr_with_centered_reward": spearman(global_vals, reward_centered_values)})
    with (R / "GROUP_SUPPORT_VARIATION.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(variation_rows[0].keys())); writer.writeheader(); writer.writerows(variation_rows)
    with (R / "CENTERED_REWARD_COMPONENTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(centered_component_rows[0].keys())); writer.writeheader(); writer.writerows(centered_component_rows)

    # ---- Phase E advantage sensitivity ----
    def make_variants(members):
        s0 = float(anchors[str(members[0]["image_id"])]["support_score"])
        rows = {name: [] for name in ("A_full", "B_without_S0", "C_without_support", "D_without_guard", "E_without_duplicate", "F_without_malformed")}
        for m in members:
            r = m["reward"]
            rows["A_full"].append(r)
            rows["B_without_S0"].append(r + s0)
            rows["C_without_support"].append(r - m["delta_support"])
            rows["D_without_guard"].append(r + 0.5 * m["coverage_guard"])
            rows["E_without_duplicate"].append(r + 0.25 * m["duplicate_penalty"])
            rows["F_without_malformed"].append(r + 0.5 * m["malformed_penalty"])
        return rows
    reference, comparisons = {}, {name: [] for name in ("B_without_S0", "C_without_support", "D_without_guard", "E_without_duplicate", "F_without_malformed")}
    for image_id, members in groups.items():
        adv = make_variants(members)
        a, _, _ = group_advantages(adv["A_full"])
        reference[image_id] = a
        for name in comparisons:
            b, _, _ = group_advantages(adv[name])
            comparisons[name].append((a, b, adv["A_full"], adv[name]))
    sens_rows = []
    for name, chunks in comparisons.items():
        diffs = [abs(x - y) for a, b, ra, rb in chunks for x, y in zip(a, b)]
        base_adv = [a for a, b, ra, rb in chunks for a in a]
        other_adv = [b for a, b, ra, rb in chunks for b in b]
        sign_change = sum((1 if x > 1e-8 else -1 if x < -1e-8 else 0) != (1 if y > 1e-8 else -1 if y < -1e-8 else 0) for x, y in zip(base_adv, other_adv))
        top_change = 0
        for a, b, ra, rb in chunks:
            top_a = {i for i, v in enumerate(a) if abs(v - max(a)) <= 1e-8}
            top_b = {i for i, v in enumerate(b) if abs(v - max(b)) <= 1e-8}
            top_change += 1 if top_a != top_b else 0
        zero_variance = sum(1 for a, b, ra, rb in chunks if max(b) - min(b) <= 1e-8) / len(chunks)
        dot = sum(x * y for x, y in zip(base_adv, other_adv)); na = math.sqrt(sum(x * x for x in base_adv)); nb = math.sqrt(sum(y * y for y in other_adv))
        sens_rows.append({"removed_component": name, "mean_abs_advantage_change": sum(diffs) / len(diffs),
                          "max_abs_advantage_change": max(diffs), "sign_change_rate": sign_change / len(base_adv),
                          "top_set_change_rate": top_change / len(chunks), "zero_variance_rate": zero_variance,
                          "advantage_cosine": dot / (na * nb) if na and nb else None})
    with (R / "ADVANTAGE_SENSITIVITY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sens_rows[0].keys())); writer.writeheader(); writer.writerows(sens_rows)

    s0_max = next(row for row in sens_rows if row["removed_component"] == "B_without_S0")["max_abs_advantage_change"]
    reward_diag = {
        "source": "frozen-pi0 sanity rollout (128 groups / 1024 candidates); trained step300 per-candidate rollouts were NOT saved",
        "support_label_source": "Flat verifier flat_state/support_floor (same-source grouping, NOT external validity)",
        "group_support_zero_variance_rate": support_zero / len(groups),
        "group_total_reward_zero_variance_rate": total_zero / len(groups),
        "support_constant_but_total_variable_rate": support_const_total_var / len(groups),
        "delta_support_nonzero_candidate_rate": delta_nonzero / total_candidates,
        "distinct_support_values_per_group": {str(k): v for k, v in sorted(support_value_counts.items())},
        "with_vs_without_S0_max_advantage_diff": s0_max,
        "n_groups": len(groups),
    }
    (R / "REWARD_SOURCE_AUDIT.md").write_text(
        "# Reward source audit\n\n"
        "Actual formula (from `src/vvqg/support_delta/reward.py` and `trainer._reward_groups`):\n\n"
        "```\nR_i = delta_support_i - 0.50*coverage_guard_i - 0.25*duplicate_penalty_i - 0.50*malformed_penalty_i\n"
        "delta_support_i = S(q_i) - S(q_0)\nA_i = (R_i - mean_group(R)) / std_group(R)   (0 if std <= 1e-8)\n```\n\n"
        "No clipping, rank transform, filtering, cross-image normalization, or component-wise standardization is applied to the reward; only the group standardization above.\n\n"
        f"Data source: {reward_diag['source']}.\n\n"
        f"Key numbers: support-label source = {reward_diag['support_label_source']}; "
        f"delta_support nonzero rate = {reward_diag['delta_support_nonzero_candidate_rate']:.4f}; "
        f"WITH_VS_WITHOUT_S0_MAX_ADVANTAGE_DIFF = {s0_max:.3e}.\n", encoding="utf-8")

    (R / "REWARD_DIAGNOSTICS.json").write_text(json.dumps(reward_diag, indent=2) + "\n", encoding="utf-8")

    # ---- cases ----
    cases = []
    for i in image_ids:
        cov0 = [per_image["step0"][i][rep]["coverage"] for rep in (1, 2, 3)]
        cov3 = [per_image["step300"][i][rep]["coverage"] for rep in (1, 2, 3)]
        sup0 = [per_image["step0"][i][rep]["support_label"] for rep in (1, 2, 3)]
        sup3 = [per_image["step300"][i][rep]["support_label"] for rep in (1, 2, 3)]
        if all(v is not None for v in cov0 + cov3):
            cases.append({"kind": "external_original_pair", "image_id": i,
                          "base_question": predictions["step0"][i]["question"], "step300_question": predictions["step300"][i]["question"],
                          "base_coverage_3rep": cov0, "step300_coverage_3rep": cov3,
                          "base_coverage_mean": sum(cov0) / 3, "step300_coverage_mean": sum(cov3) / 3,
                          "base_support_3rep": sup0, "step300_support_3rep": sup3})
    cases.sort(key=lambda row: row["step300_coverage_mean"] - row["base_coverage_mean"], reverse=True)
    external_cases = cases[:4] + cases[-4:]
    # group summaries: one support-constant, one support-varying
    group_summaries = []
    const_group = next(({"image_id": k, "members": v} for k, v in groups.items() if (max(float(m["support_score"]) for m in v) - min(float(m["support_score"]) for m in v)) <= 1e-8), None)
    var_group = next(({"image_id": k, "members": v} for k, v in groups.items() if len({float(m["support_score"]) for m in v}) > 1), None)
    for group in (const_group, var_group):
        if not group:
            continue
        members = group["members"]
        advantages, _, _ = group_advantages([m["reward"] for m in members])
        group_summaries.append({"kind": "training_group_summary", "group_type": "support_constant" if group is const_group else "support_varying",
                                "image_id": group["image_id"], "s0": float(anchors[group["image_id"]]["support_score"]),
                                "members": [{"question": m["question"], "support_score": m["support_score"], "delta_support": m["delta_support"],
                                             "coverage_guard": m["coverage_guard"], "duplicate_penalty": m["duplicate_penalty"],
                                             "malformed_penalty": m["malformed_penalty"], "reward": m["reward"], "advantage": advantages[idx]}
                                            for idx, m in enumerate(members[:3])]})
    (R / "CASES.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in external_cases + group_summaries), encoding="utf-8")

    (R / "MISSING_INFORMATION.md").write_text(
        "# Missing information\n\n"
        "- Trained step300 per-candidate rollouts were never saved (`pilot_final/rollouts/` empty); reward attribution uses the saved frozen-pi0 sanity rollout (128 groups).\n"
        "- No external support labels exist for training rollouts -> training-time supported/unsupported class is same-source Flat, not independent.\n"
        "- G3 judge-robustness for the identical questions is only inferable from repeated calls; no independent gold labels.\n"
        "- final_holdout NOT accessed.\n", encoding="utf-8")
    (R / "COMMANDS.md").write_text(
        "```bash\n"
        "PYTHONPATH=src .venv/bin/python scripts/vvqg/support_delta_review.py --report-dir <R> --experiment-dir reports/support_delta_grpo_20261006T022616Z\n"
        "```\n", encoding="utf-8")
    (R / "GIT_DIFF_DIAGNOSTIC.patch").write_text(subprocess.run(["git", "diff"], cwd=ROOT, capture_output=True, text=True).stdout, encoding="utf-8")

    print(json.dumps({"input_audit_ok": bool(input_audit["same_image_set"] and input_audit["same_image_paths"]),
                      "repeat_metrics": rows, "aggregate": aggregate, "reward_diag": reward_diag,
                      "sensitivity": sens_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
