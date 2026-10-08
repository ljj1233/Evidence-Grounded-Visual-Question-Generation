#!/usr/bin/env python3
"""Assemble the published-baseline comparison table from existing canonical
artifacts only (no metric re-computation). One row per system; missing
measurements are reported as null and counted in completeness.json. Rows are
never ranked.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/published_vqg_baselines/comparison"
PUB = ROOT / "outputs/published_vqg_baselines"

COLUMNS = ["BLEU-4", "CIDEr", "Q-BLEU4-v", "Search", "Coverage", "Context", "Spatial",
           "Supported", "Unsupported", "Uncertain", "Distinct-2", "Dup", "SemDiv"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def from_phase5a_summary(path: Path, semantic_diversity):
    d = load(path)
    t, vc, vs = d["traditional"], d["visual_centric"], d["visual_support"]
    return row(t["corpus_bleu4"], t["corpus_cider"], t["corpus_qbleu4_variant"], vc, vs, t["distinct_2"],
               t["normalized_excess_duplicate_rate"], semantic_diversity)


def row(bleu4, cider, qbleu, vc, vs, distinct2, dup, semdiv):
    return {
        "BLEU-4": bleu4, "CIDEr": cider, "Q-BLEU4-v": qbleu,
        "Search": vc["visual_search_difficulty"], "Coverage": vc["visual_evidence_coverage"],
        "Context": vc["visual_context_reasoning"], "Spatial": vc["visual_spatial_reasoning"],
        "Supported": vs["supported"], "Unsupported": vs["unsupported"], "Uncertain": vs["uncertain"],
        "Distinct-2": distinct2, "Dup": dup, "SemDiv": semdiv,
    }


def faithful_m2():
    d = load(ROOT / "outputs/same_backbone_baselines/seeq_faithful/evaluation/comparison.json")["faithful"]["m2"]
    support = load(ROOT / "outputs/same_backbone_baselines/seeq_faithful/evaluation/visual_support/summary.json")["systems"]["m2"]["premise_consistency"]["rates"]
    semdiv = load(ROOT / "outputs/same_backbone_baselines/seeq_faithful/evaluation/diversity_M0M2/summary.json")["systems"]["sft"]["mean"]
    vc = {"visual_search_difficulty": d["visual_search_difficulty"], "visual_evidence_coverage": d["visual_evidence_coverage"],
          "visual_context_reasoning": d["visual_context_reasoning"], "visual_spatial_reasoning": d["visual_spatial_reasoning"]}
    vs = {"supported": support["supported"], "unsupported": support["unsupported"], "uncertain": support["uncertain"]}
    return row(d["bleu4_greedy"], d["cider_greedy"], _m2_qbleu(), vc, vs, d["distinct_2_greedy"], d["duplicate_greedy"], semdiv)


def _m2_qbleu():
    return load(ROOT / "outputs/same_backbone_baselines/seeq_faithful/evaluation/metrics_m2_greedy/system_summary.json")["corpus_qbleu4_variant"]


def static_uvfc():
    d = load(ROOT / "outputs/phase4b_online_uvfc/static_uvfc/evaluation/summary.json")
    vc = {**d["seeq"], "visual_search_difficulty": d["seeq"]["visual_search_difficulty"], "visual_evidence_coverage": d["seeq"]["visual_evidence_coverage"],
          "visual_context_reasoning": d["seeq"]["visual_context_reasoning"], "visual_spatial_reasoning": d["seeq"]["visual_spatial_reasoning"]}
    return row(d["traditional"]["corpus_bleu4"], d["traditional"]["corpus_cider"], d["traditional"]["corpus_qbleu4_variant"],
               vc, d["support"], d["traditional"]["distinct_2"], d["traditional"]["normalized_excess_duplicate_rate"], d["semantic_diversity"])


def published(method: str):
    # sampled4 diversity summary stores the two published systems under legacy keys
    key = {"lova3": "zero_shot", "genixer": "sft"}[method]
    diversity = OUT / "diversity/summary.json"
    semdiv = load(diversity)["systems"][key]["mean"] if diversity.is_file() else None
    return from_phase5a_summary(PUB / method / "evaluation/summary.json", semdiv)


def main() -> int:
    rows = {
        "SFT": from_phase5a_summary(ROOT / "outputs/phase5a_baselines/evaluation/sft/summary.json",
                                    load(ROOT / "outputs/phase4b_online_uvfc/comparison/final_table.json")["SFT"]["SemDiv"]),
        "GPT-5.6-Luna Direct": from_phase5a_summary(ROOT / "outputs/phase5a_baselines/evaluation/luna_direct/summary.json", None),
        "Prompt-CoT": from_phase5a_summary(ROOT / "outputs/phase5a_baselines/evaluation/prompt_cot/summary.json", None),
        "Faithful SeeQ M2": faithful_m2(),
        "LOVA3 GenQA": published("lova3"),
        "Genixer Common-VQA": published("genixer"),
        "Static UVFC GRPO": static_uvfc(),
    }
    labels = {
        "LOVA3 GenQA": "LOVA3 GenQA (released LLaVA-v1.5-7B checkpoint)",
        "Genixer Common-VQA": "Genixer-L Common-VQA Generator (released LLaVA-v1.5-7B checkpoint)",
    }
    table = {
        "schema": "published-vqg-comparison-v1",
        "unit": "development1000; corpus BLEU/CIDEr/Q-BLEU4-v/Distinct-2/Dup; image-mean SeeQ dims and support rates",
        "columns": COLUMNS,
        "rows": rows,
        "labels": labels,
        "categories": {
            "A_controlled_same_backbone": ["SFT", "Prompt-CoT", "Faithful SeeQ M2", "Static UVFC GRPO"],
            "B_released_published_models": ["GPT-5.6-Luna Direct", "LOVA3 GenQA", "Genixer Common-VQA"],
        },
        "notes": [
            "No ranking and no winner is claimed.",
            "SemDiv is null where no 4-question within-image embedding diversity artifact exists for that system.",
            "SeeQ visual-centric uses the frozen SeeQ-rubric GPT-5.6-Luna judge; visual support uses the frozen independent GPT-4o evaluator.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")

    coverage = {}
    for name, values in rows.items():
        coverage[name] = {"present": sum(v is not None for v in values.values()), "total": len(COLUMNS),
                          "missing": [c for c in COLUMNS if values[c] is None]}
        for method in ("lova3", "genixer"):
            pass
    completeness = {
        "schema": "published-vqg-completeness-v1",
        "rows": coverage,
        "seeq_support": {
            m: {"seeq_valid": load(PUB / m / "evaluation/visual_centric.json")["valid"],
                "support_valid": load(PUB / m / "evaluation/visual_support.json")["valid"]}
            for m in ("lova3", "genixer")
        },
        "integrity": {m: load(PUB / m / "integrity.json") for m in ("lova3", "genixer")},
    }
    (OUT / "completeness.json").write_text(json.dumps(completeness, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
