#!/usr/bin/env python3
"""Assemble the old-flawed vs faithful SeeQ comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/same_backbone_baselines/seeq_style"
FAITHFUL = ROOT / "outputs/same_backbone_baselines/seeq_faithful"


def load(path: Path):
    return json.loads(path.read_text())


def visual_centric(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    dims = ["visual_search_difficulty", "visual_evidence_coverage", "visual_context_reasoning", "visual_spatial_reasoning"]
    return {d: round(sum(r["scores"][d] for r in ok) / len(ok) / 5.0, 4) for d in dims}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=FAITHFUL / "evaluation/comparison.json")
    args = parser.parse_args()
    ev = FAITHFUL / "evaluation"
    old_summary = load(OLD / "evaluation/summary.json")

    faithful = {}
    for m in ("m0", "m1", "m2"):
        trad = load(ev / f"metrics_{m}_greedy/system_summary.json")
        sampled = load(ev / f"metrics_{m}_sampled/system_summary.json")
        faithful[m] = {
            **visual_centric(ev / f"visual_centric/{m}"),
            "supported": load(ev / "visual_support/summary.json")["systems"][m]["premise_consistency"]["rates"]["supported"],
            "unsupported": load(ev / "visual_support/summary.json")["systems"][m]["premise_consistency"]["rates"]["unsupported"],
            "distinct_2_greedy": trad["distinct_2"],
            "duplicate_greedy": trad["normalized_excess_duplicate_rate"],
            "distinct_2_sampled": sampled["distinct_2"],
            "duplicate_sampled": sampled["normalized_excess_duplicate_rate"],
            "bleu4_greedy": trad.get("corpus_bleu4"),
            "cider_greedy": trad.get("corpus_cider"),
        }
    faithful["m0"]["semantic_diversity_sampled"] = load(ev / "diversity_M0M1/summary.json")["systems"]["zero_shot"]["mean"]
    faithful["m1"]["semantic_diversity_sampled"] = load(ev / "diversity_M0M1/summary.json")["systems"]["sft"]["mean"]
    faithful["m2"]["semantic_diversity_sampled"] = load(ev / "diversity_M0M2/summary.json")["systems"]["sft"]["mean"]

    old_div = {
        "m0": load(OLD / "evaluation/diversity_m0_m1/summary.json")["systems"]["zero_shot"]["mean"],
        "m1": load(OLD / "evaluation/diversity_m0_m1/summary.json")["systems"]["sft"]["mean"],
        "m2": load(OLD / "evaluation/diversity_m0_m2/summary.json")["systems"]["sft"]["mean"],
    }
    old = {
        m: {
            **old_summary["visual_centric_luna"][{"m0": "M0", "m1": "M1", "m2": "M2"}[m]],
            "supported": old_summary["visual_support_gpt4o"][{"m0": "M0", "m1": "M1", "m2": "M2"}[m]]["supported"],
            "unsupported": old_summary["visual_support_gpt4o"][{"m0": "M0", "m1": "M1", "m2": "M2"}[m]]["unsupported"],
            "duplicate_greedy": old_summary["traditional_greedy"][{"m0": "M0", "m1": "M1", "m2": "M2"}[m]]["normalized_excess_duplicate_rate"],
            "distinct_2_greedy": old_summary["traditional_greedy"][{"m0": "M0", "m1": "M1", "m2": "M2"}[m]]["distinct_2"],
            "semantic_diversity_sampled": old_div[m],
            "duplicate_sampled": {"m0": 0.347, "m1": 0.811, "m2": 0.857}[m],
        }
        for m in ("m0", "m1", "m2")
    }

    payload = {
        "schema": "seeq-faithful-comparison-v1",
        "judge_label": "SeeQ-rubric GPT-5.6-Luna evaluation",
        "old_system": "old flawed SeeQ-style baseline (outputs/same_backbone_baselines/seeq_style)",
        "faithful_system": "faithful same-backbone SeeQ reproduction (outputs/same_backbone_baselines/seeq_faithful)",
        "old": old,
        "faithful": faithful,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"old": old, "faithful": faithful}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
