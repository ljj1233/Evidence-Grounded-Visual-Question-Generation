#!/usr/bin/env python3
"""Consolidate every method's development1000 metrics into one table.

SeeQ dims are image means (0-1); support are rates; BLEU/CIDEr/Dup are corpus;
SemDiv is within-image semantic diversity. Read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/analysis"
DIMS = {"Search": "visual_search_difficulty", "Coverage": "visual_evidence_coverage",
        "Context": "visual_context_reasoning", "Spatial": "visual_spatial_reasoning"}


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if Path(p).is_file() else []


def seeq(path: Path) -> dict:
    rows = [r for r in read(path) if r.get("status") == "ok"]
    if not rows:
        return {}
    return {name: float(np.mean([r["scores"][field] / 5.0 for r in rows])) for name, field in DIMS.items()} | {"n": len(rows)}


def support(path: Path) -> dict:
    rows = [r for r in read(path) if r.get("status") == "ok"]
    if not rows:
        return {}
    return {label: float(np.mean([r.get("overall_visual_support") == label for r in rows])) for label in ("supported", "unsupported", "uncertain")} | {"n": len(rows)}


def traditional(path: Path) -> dict:
    d = json.loads(path.read_text()) if Path(path).is_file() else {}
    return {"BLEU": d.get("corpus_bleu4"), "CIDEr": d.get("corpus_cider"), "Q-BLEU4": d.get("corpus_qbleu4_variant"),
            "Dup": d.get("normalized_excess_duplicate_rate"), "Distinct-2": d.get("distinct_2")}


def semdiv(path: Path, key: str = "sft") -> float | None:
    d = json.loads(path.read_text()) if Path(path).is_file() else {}
    return d.get("systems", {}).get(key, {}).get("mean")


def run_summary(eval_dir: Path) -> dict:
    s = json.loads((eval_dir / "summary.json").read_text())
    seeq_ = {name: s["seeq"][field] for name, field in DIMS.items()} | {"n": s["seeq"].get("valid")}
    sup = {k: s["support"].get(k) for k in ("supported", "unsupported", "uncertain")} | {"n": s["support"].get("valid")}
    trad = s["traditional"]
    return {"seeq": seeq_, "support": sup,
            "traditional": {"BLEU": trad.get("corpus_bleu4"), "CIDEr": trad.get("corpus_cider"), "Q-BLEU4": trad.get("corpus_qbleu4_variant"),
                            "Dup": trad.get("normalized_excess_duplicate_rate"), "Distinct-2": trad.get("distinct_2")},
            "semdiv": s.get("semantic_diversity")}


def p1strong() -> dict:
    m = ROOT / "outputs/phase3a_p1strong_eval/metrics"
    return {"seeq": seeq(m / "seeq_scores.jsonl"), "support": support(m / "support/system.jsonl"),
            "traditional": traditional(m / "traditional/system_summary.json"), "semdiv": semdiv(m / "diversity_semantic/summary.json")}


def main() -> int:
    P = ROOT / "outputs"
    methods = {
        "Zero-shot Qwen3-VL": {"init": "base", "train": "none", "anchor": "—",
            "seeq": seeq(P / "phase24/evaluation/seeq_luna/zero_shot_scores.jsonl"),
            "support": support(P / "phase23/evaluation/visual_support/zero_shot.jsonl"),
            "traditional": traditional(P / "phase23/evaluation/zero_shot_greedy/system_summary.json"),
            "semdiv": semdiv(P / "phase24/evaluation/seeq_diversity/summary.json", "zero_shot")},
        "SFT": {"init": "SFT", "train": "SFT", "anchor": "—",
            "seeq": seeq(P / "phase24/evaluation/seeq_luna/sft_scores.jsonl"),
            "support": support(P / "phase23/evaluation/visual_support/sft.jsonl"),
            "traditional": traditional(P / "phase23/evaluation/sft_greedy/system_summary.json"),
            "semdiv": semdiv(P / "phase24/evaluation/seeq_diversity/summary.json", "sft")},
        "Fixed-rollout P1strong": {"init": "SFT", "train": "offline GRPO", "anchor": "SFT", **p1strong()},
        "Static UVFC GRPO": {"init": "SFT", "train": "online GRPO", "anchor": "SFT", **run_summary(P / "phase4b_online_uvfc/static_uvfc/evaluation")},
        "Dynamic UVFC GRPO": {"init": "SFT", "train": "online GRPO", "anchor": "SFT-dynamic", **run_summary(P / "phase4b_online_uvfc/dynamic_uvfc/evaluation")},
        "Base UVFC GRPO (SFT anchor)": {"init": "base", "train": "online GRPO", "anchor": "SFT", **run_summary(P / "phase4b_online_uvfc/base_uvfc/evaluation")},
        "Base UVFC GRPO (base anchor)": {"init": "base", "train": "online GRPO", "anchor": "base", **run_summary(P / "phase4b_online_uvfc/base_uvfc_baseanchor/evaluation")},
        "Prompt-CoT Qwen3-VL": {"init": "base", "train": "none (CoT prompt)", "anchor": "—",
            "seeq": {name: json.loads((P / "phase5a_baselines/evaluation/prompt_cot/summary.json").read_text())["visual_centric"][field] for name, field in DIMS.items()} | {"n": json.loads((P/"phase5a_baselines/evaluation/prompt_cot/summary.json").read_text())["visual_centric"]["valid"]},
            "support": {k: json.loads((P/"phase5a_baselines/evaluation/prompt_cot/summary.json").read_text())["visual_support"].get(k) for k in ("supported","unsupported","uncertain")} | {"n": json.loads((P/"phase5a_baselines/evaluation/prompt_cot/summary.json").read_text())["visual_support"]["valid"]},
            "traditional": traditional(P / "phase5a_baselines/evaluation/prompt_cot/traditional/system_summary.json"),
            "semdiv": None},
        "GPT-5.6-Luna Direct": {"init": "external", "train": "none", "anchor": "—",
            "seeq": {name: json.loads((P / "phase5a_baselines/evaluation/luna_direct/summary.json").read_text())["visual_centric"][field] for name, field in DIMS.items()} | {"n": json.loads((P/"phase5a_baselines/evaluation/luna_direct/summary.json").read_text())["visual_centric"]["valid"]},
            "support": {k: json.loads((P/"phase5a_baselines/evaluation/luna_direct/summary.json").read_text())["visual_support"].get(k) for k in ("supported","unsupported","uncertain")} | {"n": json.loads((P/"phase5a_baselines/evaluation/luna_direct/summary.json").read_text())["visual_support"]["valid"]},
            "traditional": traditional(P / "phase5a_baselines/evaluation/luna_direct/traditional/system_summary.json"),
            "semdiv": None},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "all_methods_metrics.json").write_text(json.dumps(methods, indent=2) + "\n", encoding="utf-8")

    cols = ["init", "train", "anchor", "Search", "Coverage", "Context", "Spatial", "supported", "unsupported", "BLEU", "CIDEr", "Q-BLEU4", "Dup", "Distinct-2", "semdiv"]
    print(f"{'method':32s} " + " ".join(f"{c:>9s}" for c in cols))
    for name, m in methods.items():
        vals = []
        for c in cols:
            v = m.get(c)
            if v is None:
                v = m["seeq"].get(c) if c in m["seeq"] else m["support"].get(c) if c in m["support"] else m["traditional"].get(c) if c in m["traditional"] else None
            vals.append("—" if v is None else (f"{v:.3f}" if isinstance(v, (int, float)) else str(v)))
        print(f"{name:32s} " + " ".join(f"{v:>9s}" for v in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
