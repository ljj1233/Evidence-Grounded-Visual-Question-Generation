#!/usr/bin/env python3
"""Paired intervention-gap statistics: Full Ours vs SeeQ M2 (offline only).

Reuses existing per-image evaluator outputs. No generation, no API, no training.
Pairing unit is image_id. Bootstrap resamples images, seed 42, 5000 resamples.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.evaluation.visual_centric import seeq  # noqa: E402

CONDITIONS = ("original", "blurred", "blank", "shuffled")
TRANSFORMS = ("blurred", "blank", "shuffled")


def read(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def norm_q(q):
    return " ".join(q.strip().lower().split())


def ned(a, b):
    return 1 - difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def boot(d, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
            "pos": int((d > 0).sum()), "neg": int((d < 0).sum()), "tie": int((d == 0).sum()), "n": int(len(d))}


def desc(d):
    d = np.asarray(d, dtype=float)
    return {"mean": float(d.mean()), "median": float(np.median(d)), "std": float(d.std(ddof=0)),
            "p25": float(np.percentile(d, 25)), "p75": float(np.percentile(d, 75)),
            "pos": int((d > 0).sum()), "neg": int((d < 0).sum()), "tie": int((d == 0).sum()), "n": int(len(d))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fo-report", type=Path, required=True)
    ap.add_argument("--pb-report", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    FO, PB, OUT = args.fo_report, args.pb_report, args.outdir
    (OUT / "statistics").mkdir(parents=True, exist_ok=True)
    S = OUT / "statistics"

    dim = seeq._audit()["seeq_protocol"]["dimensions"][0]
    ph = hashlib.sha256((dim["prompt_system"] + dim["prompt_user"]).encode()).hexdigest()
    cache = {r["cache_key"]: r for r in read(FO / "evaluation/seeq_cache.jsonl") if r.get("status") == "ok"}

    # ---- Full Ours ----
    fo_gen, fo_cov, fo_sup = {}, {}, {}
    for c in CONDITIONS:
        gen_rows = read(FO / f"generation/full_ours_{c}.jsonl")
        fo_gen[c] = {str(r["image_id"]): r["question"] for r in gen_rows}
        fo_img = {str(r["image_id"]): r["image"] for r in gen_rows}
        fo_cov[c] = {}
        for i, q in fo_gen[c].items():
            key = seeq.cache_key({"image_id": i, "question": q, "image": fo_img[i]}, "gpt-5.6-luna", ph, seeq.image_digest(ROOT / fo_img[i]))
            row = cache.get(key)
            fo_cov[c][i] = None if row is None else row["scores"]["visual_evidence_coverage"] / 5.0
        fo_sup[c] = {str(r["image_id"]): r for r in read(FO / f"evaluation/support/{c}.jsonl")}

    # ---- SeeQ M2 ----
    m2_gen, m2_cov, m2_sup = {}, {}, {}
    for c in CONDITIONS:
        preds = {str(r["image_id"]): r["question"] for r in read(PB / f"metrics/seeq_m2/{c}/preds_original_image.jsonl")}
        m2_gen[c] = preds
        scores = read(PB / f"metrics/seeq_m2/{c}/seeq/scores.jsonl")
        m2_cov[c] = {str(r["image_id"]): (r["scores"]["visual_evidence_coverage"] / 5.0 if r.get("status") == "ok" else None) for r in scores}
        m2_sup[c] = {str(r["image_id"]): r for r in read(PB / f"metrics/seeq_m2/{c}/support/seeq_m2__{c}.jsonl")}

    def sup_ok(store, c, i):
        r = store[c].get(i)
        return r is not None and r.get("status") == "ok"

    def sup_label(store, c, i):
        return store[c][i].get("overall_visual_support")

    # ---- common sets ----
    common = {}
    excluded = {}
    for c in TRANSFORMS:
        for metric in ("coverage", "support"):
            if metric == "coverage":
                fo = {i for i in fo_cov["original"] if fo_cov["original"][i] is not None and fo_cov[c][i] is not None}
                m2 = {i for i in m2_cov["original"] if m2_cov["original"][i] is not None and m2_cov[c][i] is not None}
            else:
                fo = {i for i in fo_sup["original"] if sup_ok(fo_sup, "original", i) and sup_ok(fo_sup, c, i)}
                m2 = {i for i in m2_sup["original"] if sup_ok(m2_sup, "original", i) and sup_ok(m2_sup, c, i)}
            inter = fo & m2
            common[f"{c}_{metric}"] = sorted(inter, key=int)
            excluded[f"{c}_{metric}"] = {"fo_valid": len(fo), "m2_valid": len(m2), "common": len(inter),
                                         "dropped_from_fo": sorted(fo - m2, key=int), "dropped_from_m2": sorted(m2 - fo, key=int),
                                         "reason": "image not valid (original or transformed) in both methods"}

    # ---- paired rows ----
    rows = []
    for c in CONDITIONS:
        for i in sorted(set(fo_gen[c]) | set(m2_gen[c]), key=int):
            rows.append({
                "method": "full_ours", "condition": c, "image_id": i, "question": fo_gen[c].get(i),
                "coverage_raw": None if fo_cov[c].get(i) is None else round(fo_cov[c][i] * 5),
                "coverage_normalized": fo_cov[c].get(i),
                "support_label": sup_label(fo_sup, c, i) if sup_ok(fo_sup, c, i) else None,
                "supported": int(sup_label(fo_sup, c, i) == "supported") if sup_ok(fo_sup, c, i) else None,
                "unsupported": int(sup_label(fo_sup, c, i) == "unsupported") if sup_ok(fo_sup, c, i) else None,
                "uncertain": int(sup_label(fo_sup, c, i) == "uncertain") if sup_ok(fo_sup, c, i) else None,
                "coverage_valid": fo_cov[c].get(i) is not None,
                "support_valid": sup_ok(fo_sup, c, i),
            })
            rows.append({
                "method": "seeq_m2", "condition": c, "image_id": i, "question": m2_gen[c].get(i),
                "coverage_raw": None if m2_cov[c].get(i) is None else round(m2_cov[c][i] * 5),
                "coverage_normalized": m2_cov[c].get(i),
                "support_label": sup_label(m2_sup, c, i) if sup_ok(m2_sup, c, i) else None,
                "supported": int(sup_label(m2_sup, c, i) == "supported") if sup_ok(m2_sup, c, i) else None,
                "unsupported": int(sup_label(m2_sup, c, i) == "unsupported") if sup_ok(m2_sup, c, i) else None,
                "uncertain": int(sup_label(m2_sup, c, i) == "uncertain") if sup_ok(m2_sup, c, i) else None,
                "coverage_valid": m2_cov[c].get(i) is not None,
                "support_valid": sup_ok(m2_sup, c, i),
            })
    with (S / "paired_intervention_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- absolute metrics (own valid denominators) ----
    abs_rows = []
    for method, gen, cov, sup in (("full_ours", fo_gen, fo_cov, fo_sup), ("seeq_m2", m2_gen, m2_cov, m2_sup)):
        for c in CONDITIONS:
            cv = [cov[c][i] for i in gen[c] if cov[c].get(i) is not None]
            sv = [sup[c][i] for i in gen[c] if sup_ok(sup, c, i)]
            abs_rows.append({
                "method": method, "condition": c, "coverage_valid": len(cv), "support_valid": len(sv),
                "coverage": statistics.fmean(cv) if cv else None,
                "supported": statistics.fmean(r["overall_visual_support"] == "supported" for r in sv) if sv else None,
                "unsupported": statistics.fmean(r["overall_visual_support"] == "unsupported" for r in sv) if sv else None,
                "uncertain": statistics.fmean(r["overall_visual_support"] == "uncertain" for r in sv) if sv else None,
            })
    with (S / "absolute_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(abs_rows[0])); w.writeheader(); w.writerows(abs_rows)

    # ---- gaps ----
    gap_rows, boot_rows = [], []
    boot_rows.append({"quantity": "original_full_ours_minus_seeq_coverage", **boot([fo_cov["original"][i] - m2_cov["original"][i]
        for i in sorted(set(fo_cov["original"]) & set(m2_cov["original"]), key=int) if fo_cov["original"][i] is not None and m2_cov["original"][i] is not None])})
    boot_rows.append({"quantity": "original_full_ours_minus_seeq_supported", **boot([(sup_label(fo_sup, "original", i) == "supported") - (sup_label(m2_sup, "original", i) == "supported")
        for i in sorted(set(fo_sup["original"]) & set(m2_sup["original"]), key=int) if sup_ok(fo_sup, "original", i) and sup_ok(m2_sup, "original", i)])})
    boot_rows.append({"quantity": "original_full_ours_minus_seeq_unsupported", **boot([(sup_label(fo_sup, "original", i) == "unsupported") - (sup_label(m2_sup, "original", i) == "unsupported")
        for i in sorted(set(fo_sup["original"]) & set(m2_sup["original"]), key=int) if sup_ok(fo_sup, "original", i) and sup_ok(m2_sup, "original", i)])})

    for c in TRANSFORMS:
        cc = common[f"{c}_coverage"]
        sc = common[f"{c}_support"]
        drop_cov_fo = {i: fo_cov["original"][i] - fo_cov[c][i] for i in cc}
        drop_cov_m2 = {i: m2_cov["original"][i] - m2_cov[c][i] for i in cc}
        gap_cov = [drop_cov_fo[i] - drop_cov_m2[i] for i in cc]
        drop_sup_fo = {i: int(sup_label(fo_sup, "original", i) == "supported") - int(sup_label(fo_sup, c, i) == "supported") for i in sc}
        drop_sup_m2 = {i: int(sup_label(m2_sup, "original", i) == "supported") - int(sup_label(m2_sup, c, i) == "supported") for i in sc}
        gap_sup = [drop_sup_fo[i] - drop_sup_m2[i] for i in sc]
        inc_uns_fo = {i: int(sup_label(fo_sup, c, i) == "unsupported") - int(sup_label(fo_sup, "original", i) == "unsupported") for i in sc}
        inc_uns_m2 = {i: int(sup_label(m2_sup, c, i) == "unsupported") - int(sup_label(m2_sup, "original", i) == "unsupported") for i in sc}
        gap_uns = [inc_uns_fo[i] - inc_uns_m2[i] for i in sc]
        change_fo = {i: int(norm_q(fo_gen["original"][i]) != norm_q(fo_gen[c][i])) for i in cc}
        change_m2 = {i: int(norm_q(m2_gen["original"][i]) != norm_q(m2_gen[c][i])) for i in cc}
        gap_change = [change_fo[i] - change_m2[i] for i in cc]
        ned_fo = {i: ned(fo_gen["original"][i], fo_gen[c][i]) for i in cc}
        ned_m2 = {i: ned(m2_gen["original"][i], m2_gen[c][i]) for i in cc}
        gap_ned = [ned_fo[i] - ned_m2[i] for i in cc]
        for name, arr in (("gap_cov", gap_cov), ("gap_sup", gap_sup), ("gap_unsup", gap_uns), ("gap_change", gap_change), ("gap_ned", gap_ned)):
            gap_rows.append({"condition": c, "gap": name, **desc(arr)})
        for name, arr in (("coverage_intervention_gap", gap_cov), ("support_intervention_gap", gap_sup),
                          ("unsupported_intervention_gap", gap_uns), ("question_change_gap", gap_change), ("ned_gap", gap_ned)):
            boot_rows.append({"quantity": f"{c}_{name}", **boot(arr)})
    with (S / "intervention_gap_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gap_rows[0])); w.writeheader(); w.writerows(gap_rows)
    with (S / "bootstrap_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(boot_rows[0])); w.writeheader(); w.writerows(boot_rows)

    (S / "common_image_sets.json").write_text(json.dumps({"common": {k: len(v) for k, v in common.items()},
                                                          "sets": {k: v for k, v in common.items()}}, indent=2) + "\n")
    (S / "missing_and_excluded_ids.json").write_text(json.dumps(excluded, indent=2) + "\n")

    # ---- repeat stability ----
    def load_fo_repeat(cond, run, kind):
        if kind == "seeq":
            return {str(r["image_id"]): (r["scores"]["visual_evidence_coverage"] / 5.0 if r.get("status") == "ok" else None)
                    for r in read(FO / f"metrics_repeat/full_ours/{cond}/run{run}/seeq.jsonl")}
        return {str(r["image_id"]): (r.get("overall_visual_support") if r.get("status") == "ok" else None)
                for r in read(FO / f"metrics_repeat/full_ours/{cond}/run{run}/support/{cond}.jsonl")}

    def load_m2_repeat(cond, run, kind):
        base = PB / f"metrics_repeat/seeq_m2/{cond}/run{run}" if run > 1 else PB / f"metrics/seeq_m2/{cond}"
        if kind == "seeq":
            return {str(r["image_id"]): (r["scores"]["visual_evidence_coverage"] / 5.0 if r.get("status") == "ok" else None)
                    for r in read(base / "seeq/scores.jsonl")}
        return {str(r["image_id"]): (r.get("overall_visual_support") if r.get("status") == "ok" else None)
                for r in read(base / f"support/seeq_m2__{cond}.jsonl")}

    repeat_rows = []
    for method, loader in (("full_ours", load_fo_repeat), ("seeq_m2", load_m2_repeat)):
        for cond in ("original", "blank"):
            for kind in ("seeq", "support"):
                runs = [loader(cond, r, kind) for r in (1, 2, 3)]
                ids = sorted(set(runs[0]) & set(runs[1]) & set(runs[2]), key=int)
                ids = [i for i in ids if all(runs[k].get(i) is not None for k in range(3))]
                if not ids:
                    continue
                if kind == "seeq":
                    per_var = [np.var([runs[k][i] for k in range(3)]) for i in ids]
                    agg = [statistics.fmean(runs[k][i] for i in ids) for k in range(3)]
                    repeat_rows.append({"method": method, "condition": cond, "metric": "coverage", "n": len(ids),
                                        "agg_mean": statistics.fmean(agg), "agg_std": statistics.pstdev(agg),
                                        "agg_min": min(agg), "agg_max": max(agg),
                                        "mean_within_image_variance": statistics.fmean(per_var), "flip_rate": None})
                else:
                    flips = sum(len({runs[k][i] for k in range(3)}) > 1 for i in ids)
                    majority = [statistics.mode([runs[k][i] for k in range(3)]) for i in ids]
                    agg = [statistics.fmean(runs[k][i] == "supported" for i in ids) for k in range(3)]
                    repeat_rows.append({"method": method, "condition": cond, "metric": "supported", "n": len(ids),
                                        "agg_mean": statistics.fmean(agg), "agg_std": statistics.pstdev(agg),
                                        "agg_min": min(agg), "agg_max": max(agg),
                                        "mean_within_image_variance": None, "flip_rate": flips / len(ids)})
    with (S / "repeat_stability.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(repeat_rows[0])); w.writeheader(); w.writerows(repeat_rows)

    summary = {"bootstrap": {"seed": 42, "resamples": 5000, "unit": "image_id", "ci": "percentile 2.5/97.5"},
               "common_image_counts": {k: len(v) for k, v in common.items()},
               "absolute_metrics": abs_rows, "gaps": gap_rows, "bootstrap_results": boot_rows,
               "repeat_stability": repeat_rows}
    (S / "statistics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"common": summary["common_image_counts"], "bootstrap": boot_rows}, default=str)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
