#!/usr/bin/env python3
"""Full Ours intervention tables (A/B) and case files. Offline only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.evaluation.visual_centric import seeq  # noqa: E402

CONDITIONS = ("original", "blurred", "blank", "shuffled")
BASELINE_LABELS = {
    "mostafazadeh_nn": "Mostafazadeh NN-generator",
    "jain_creativity_vae": "Jain Creativity VAE",
    "lova3": "LOVA3 GenQA",
    "genixer": "Genixer Common-VQA",
    "seeq_m2": "SeeQ (faithful M2)",
}


def rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def dup_excess(questions):
    qs = [q.strip().lower() for q in questions]
    return 1 - len(set(qs)) / len(qs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--pb-report", type=Path, required=True)
    args = ap.parse_args()
    R, PB = args.report, args.pb_report
    analysis = json.loads((R / "analysis.json").read_text())
    fo = analysis["full_ours"]["per_condition"]

    # baselines from published table
    base = {}
    with (PB / "metrics/published_baseline_intervention_table.csv").open() as fh:
        for row in csv.DictReader(fh):
            base[row["Method"]] = row
    pub_uncertain = {"Mostafazadeh NN-generator": 0.168, "Jain Creativity VAE": 0.1273,
                     "LOVA3 GenQA": 0.136, "Genixer Common-VQA": 0.1003, "SeeQ (faithful M2)": 0.1962}

    def dup_for(key):
        p = PB / f"baseline_outputs/{key}/original.jsonl"
        return dup_excess([r["question"] for r in rows(p)]) if p.exists() else None

    table_a = []
    for key, label in BASELINE_LABELS.items():
        b = base[label]
        table_a.append({"Method": label, "Type": "published/external", "Orig Coverage": float(b["Original Coverage"]),
                        "Orig Supported": float(b["Original Supported"]), "Orig Unsupported": float(b["Unsupported"]),
                        "Orig Uncertain": pub_uncertain[label], "Duplicate": dup_for(key)})
    table_a.append({"Method": "Full Ours", "Type": "ours", "Orig Coverage": fo["original"]["coverage"],
                    "Orig Supported": fo["original"]["supported"], "Orig Unsupported": fo["original"]["unsupported"],
                    "Orig Uncertain": fo["original"]["uncertain"],
                    "Duplicate": dup_excess([r["question"] for r in rows(R / "generation/full_ours_original.jsonl")])})

    table_b = []
    for key, label in BASELINE_LABELS.items():
        b = base[label]
        table_b.append({"Method": label, "Type": "published/external",
                        "VDG Blank Coverage": float(b["VDG Coverage Blank"]), "VDG Shuffle Coverage": float(b["VDG Coverage Shuffle"]),
                        "VDG Blank Supported": float(b["VDG Blank"]), "VDG Shuffle Supported": float(b["VDG Shuffle"])})
    table_b.append({"Method": "Full Ours", "Type": "ours",
                    "VDG Blank Coverage": fo["original"]["coverage"] - fo["blank"]["coverage"],
                    "VDG Shuffle Coverage": fo["original"]["coverage"] - fo["shuffled"]["coverage"],
                    "VDG Blank Supported": fo["original"]["supported"] - fo["blank"]["supported"],
                    "VDG Shuffle Supported": fo["original"]["supported"] - fo["shuffled"]["supported"]})

    with (R / "TABLE_A.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table_a[0])); w.writeheader(); w.writerows(table_a)
    with (R / "TABLE_B.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table_b[0])); w.writeheader(); w.writerows(table_b)
    (R / "tables.json").write_text(json.dumps({"TABLE_A": table_a, "TABLE_B": table_b}, indent=2) + "\n")

    # ---- cases ----
    dim = seeq._audit()["seeq_protocol"]["dimensions"][0]
    ph = hashlib.sha256((dim["prompt_system"] + dim["prompt_user"]).encode()).hexdigest()
    cache = {r["cache_key"]: r for r in rows(R / "evaluation/seeq_cache.jsonl") if r.get("status") == "ok"}
    gen = {c: {str(r["image_id"]): r for r in rows(R / f"generation/full_ours_{c}.jsonl")} for c in CONDITIONS}
    sup = {c: {str(r["image_id"]): r for r in rows(R / f"evaluation/support/{c}.jsonl") if r.get("status") == "ok"} for c in CONDITIONS}
    cov = {}
    for c in CONDITIONS:
        cov[c] = {}
        for i, r in gen[c].items():
            s = cache.get(seeq.cache_key(r, "gpt-5.6-luna", ph, seeq.image_digest(ROOT / r["image"])))
            cov[c][i] = None if s is None else s["scores"]["visual_evidence_coverage"] / 5.0
    ids = sorted(gen["original"], key=int)

    def rec(i, c):
        return {"image_id": i, "image": gen["original"][i]["image"], "condition": c,
                "question": gen[c][i]["question"], "coverage": cov[c][i],
                "support": sup[c][i].get("overall_visual_support") if i in sup[c] else None,
                "original_question": gen["original"][i]["question"], "original_coverage": cov["original"][i],
                "original_support": sup["original"][i].get("overall_visual_support") if i in sup["original"] else None}

    intervention = []
    for c in ("blank", "shuffled"):
        ranked = sorted([i for i in ids if cov[c][i] is not None and cov["original"][i] is not None],
                        key=lambda i: cov["original"][i] - cov[c][i], reverse=True)
        for i in ranked[:10]:
            intervention.append(rec(i, c))
    failure = []
    for c in ("blank", "shuffled"):
        generic = [i for i in ids if i in sup[c] and sup[c][i].get("overall_visual_support") == "supported"
                   and gen["original"][i]["question"].strip().lower() != gen[c][i]["question"].strip().lower()]
        for i in sorted(generic, key=lambda i: (cov[c][i] or 0), reverse=True)[:10]:
            failure.append({**rec(i, c), "failure_type": "transformed_question_still_supported"})
    for c in ("blank", "shuffled"):
        for i in [i for i in ids if i in sup[c] and sup[c][i].get("overall_visual_support") == "unsupported"][:0]:
            failure.append({**rec(i, c), "failure_type": "unsupported"})
    # judge instability case: blank support disagreed across the 3 fresh runs
    fresh = []
    for r in (1, 2, 3):
        fresh.append({str(x["image_id"]): x.get("overall_visual_support")
                      for x in rows(R / f"metrics_repeat/full_ours/blank/run{r}/support/blank.jsonl") if x.get("status") == "ok"})
    unstable = [i for i in ids if len({fresh[0].get(i), fresh[1].get(i), fresh[2].get(i)}) > 1]
    for i in unstable[:10]:
        failure.append({**rec(i, "blank"), "failure_type": "blank_support_judge_unstable",
                        "run1": fresh[0].get(i), "run2": fresh[1].get(i), "run3": fresh[2].get(i)})

    for name, items in (("intervention_cases", intervention), ("failure_cases", failure)):
        with (R / "cases" / f"{name}.jsonl").open("w") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"table_a": table_a, "table_b": table_b, "n_intervention": len(intervention),
                      "n_failure": len(failure), "blank_judge_unstable_images": len(unstable)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
