#!/usr/bin/env python3
"""Full Ours intervention analysis: per-condition metrics, VDG, paired bootstrap
vs SeeQ M2, and paper tables. Offline only (no GPU/API)."""

from __future__ import annotations

import argparse
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


def rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def boot_ci(d: np.ndarray, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), \
        int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--pb-report", type=Path, required=True)
    args = ap.parse_args()
    R, PB = args.report, args.pb_report

    dim = seeq._audit()["seeq_protocol"]["dimensions"][0]
    ph = hashlib.sha256((dim["prompt_system"] + dim["prompt_user"]).encode()).hexdigest()
    cache = {}
    for r in rows(R / "evaluation/seeq_cache.jsonl"):
        if r.get("status") == "ok":
            cache[r["cache_key"]] = r["scores"]

    def coverage_for(cond):
        out = {}
        for r in rows(R / f"evaluation/{cond}_predictions.jsonl"):
            s = cache.get(seeq.cache_key(r, "gpt-5.6-luna", ph, seeq.image_digest(ROOT / r["image"])))
            out[str(r["image_id"])] = None if s is None else {k: v / 5.0 for k, v in s.items()}
        return out

    def support_for(cond):
        out = {}
        for r in rows(R / f"evaluation/support/{cond}.jsonl"):
            out[str(r["image_id"])] = r
        return out

    cov = {c: coverage_for(c) for c in CONDITIONS}
    sup = {c: support_for(c) for c in CONDITIONS}
    ids = [str(r["image_id"]) for r in rows(R / "generation/full_ours_original.jsonl")]

    def metric_table():
        table = {}
        for c in CONDITIONS:
            valid_cov = [cov[c][i] for i in ids if cov[c][i] is not None]
            valid_sup = [sup[c][i] for i in ids if sup[c][i].get("status") == "ok"]
            table[c] = {
                "n": len(ids), "seeq_valid": len(valid_cov), "support_valid": len(valid_sup),
                "search": statistics.fmean(v["visual_search_difficulty"] for v in valid_cov),
                "coverage": statistics.fmean(v["visual_evidence_coverage"] for v in valid_cov),
                "context": statistics.fmean(v["visual_context_reasoning"] for v in valid_cov),
                "spatial": statistics.fmean(v["visual_spatial_reasoning"] for v in valid_cov),
                "supported": statistics.fmean(s["overall_visual_support"] == "supported" for s in valid_sup),
                "unsupported": statistics.fmean(s["overall_visual_support"] == "unsupported" for s in valid_sup),
                "uncertain": statistics.fmean(s["overall_visual_support"] == "uncertain" for s in valid_sup),
            }
        return table

    table = metric_table()

    def questions(cond):
        return {str(r["image_id"]): r["question"] for r in rows(R / f"generation/full_ours_{cond}.jsonl")}

    q = {c: questions(c) for c in CONDITIONS}
    allq = {c: [x.strip().lower() for x in q[c].values()] for c in CONDITIONS}
    change = {}
    for c in ("blurred", "blank", "shuffled"):
        changed = sum(q["original"][i].strip().lower() != q[c][i].strip().lower() for i in ids)
        ne = []
        for i in ids:
            a, b = q["original"][i].split(), q[c][i].split()
            import difflib
            ne.append(1 - difflib.SequenceMatcher(None, a, b).ratio())
        change[c] = {"change_rate": changed / len(ids), "mean_ned": statistics.fmean(ne)}
    summary = {
        "full_ours": {
            "per_condition": table,
            "unique_questions": {c: len(set(allq[c])) for c in CONDITIONS},
            "duplicate_excess": {c: 1 - len(set(allq[c])) / len(allq[c]) for c in CONDITIONS},
            "empty": {c: sum(x == "" for x in allq[c]) for c in CONDITIONS},
            "question_change": change,
            "vdg_coverage": {c: table["original"]["coverage"] - table[c]["coverage"] for c in ("blurred", "blank", "shuffled")},
            "vdg_supported": {c: table["original"]["supported"] - table[c]["supported"] for c in ("blurred", "blank", "shuffled")},
        }
    }

    # ---- SeeQ M2 per-image from published per_image_paired_deltas.csv ----
    import csv
    m2 = {c: {} for c in CONDITIONS}
    with (PB / "metrics/per_image_paired_deltas.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row["method"] != "seeq_m2":
                continue
            i = row["image_id"]
            m2["original"][i] = (float(row["original_coverage"]), int(row["original_supported"]))
            m2[row["condition"]][i] = (float(row["transformed_coverage"]), int(row["transformed_supported"]))
    m2_unsup = {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        for r in rows(PB / f"metrics/seeq_m2/{c}/support/seeq_m2__{c}.jsonl"):
            if r.get("status") == "ok":
                m2_unsup[c][str(r["image_id"])] = int(r["overall_visual_support"] == "unsupported")

    fo_cov = {c: {i: cov[c][i]["visual_evidence_coverage"] for i in ids if cov[c][i]} for c in CONDITIONS}
    fo_sup = {c: {i: int(sup[c][i]["overall_visual_support"] == "supported") for i in ids if sup[c][i].get("status") == "ok"} for c in CONDITIONS}
    fo_unsup = {c: {i: int(sup[c][i]["overall_visual_support"] == "unsupported") for i in ids if sup[c][i].get("status") == "ok"} for c in CONDITIONS}

    paired = {}
    for c in CONDITIONS:
        common = sorted(set(fo_cov[c]) & set(m2[c]) & set(fo_sup[c]) & set(m2_unsup[c]), key=int)
        dcov = np.array([fo_cov[c][i] - m2[c][i][0] for i in common])
        dsup = np.array([fo_sup[c][i] - m2[c][i][1] for i in common])
        duns = np.array([fo_unsup[c][i] - m2_unsup[c][i] for i in common])
        paired[c] = {"n_common": len(common),
                     "coverage": dict(zip(("mean", "lo", "hi", "pos", "neg", "tie"), boot_ci(dcov))),
                     "supported": dict(zip(("mean", "lo", "hi", "pos", "neg", "tie"), boot_ci(dsup))),
                     "unsupported": dict(zip(("mean", "lo", "hi", "pos", "neg", "tie"), boot_ci(duns)))}
    summary["full_ours_vs_seeq_m2"] = paired
    # VDG paired differences (Full Ours - SeeQ) for blank/shuffle
    for c in ("blank", "shuffled"):
        common = sorted(set(fo_cov["original"]) & set(fo_cov[c]) & set(m2["original"]) & set(m2[c]), key=int)
        fo_vdg = np.array([(fo_cov["original"][i] - fo_cov[c][i]) for i in common])
        m2_vdg = np.array([(m2["original"][i][0] - m2[c][i][0]) for i in common])
        d = fo_vdg - m2_vdg
        summary.setdefault("vdg_paired_vs_seeq", {})[c] = {"n_common": len(common),
            "full_ours_vdg": float(fo_vdg.mean()), "seeq_vdg": float(m2_vdg.mean()),
            "diff": dict(zip(("mean", "lo", "hi", "pos", "neg", "tie"), boot_ci(d)))}

    (R / "analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["full_ours"]["per_condition"], indent=2))
    print(json.dumps(paired, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
