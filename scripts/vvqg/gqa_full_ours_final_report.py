#!/usr/bin/env python3
"""Assemble the final GQA table (Full Ours base-init + baselines), paired
image-level bootstrap (Full Ours vs each baseline, common valid ids) and 10
fixed-rule examples.  Read-only."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PB = ROOT / "outputs/gqa_native/published_baselines"
NEW = ROOT / "outputs/gqa_native/baselines_10k"
FO = ROOT / "outputs/gqa_native/full_ours_base"
OUT = ROOT / "outputs/gqa_native/final_comparison"
DIMS = ("visual_search_difficulty", "visual_evidence_coverage", "visual_context_reasoning", "visual_spatial_reasoning")
SYS = {
    "Full Ours (base)": (FO / "g2/greedy1000.jsonl", FO / "evaluation/dev1000", "gqa_full_ours"),
    "A1 SeeQ-style": (PB / "seeq/evaluation/dev1000/predictions.jsonl", PB / "seeq/evaluation/dev1000", "seeq"),
    "B2 ConVQG": (NEW / "b2_contrastive/greedy1000.jsonl", NEW / "b2_contrastive/evaluation/dev1000", "b2"),
    "B3 Guiding-VQG": (NEW / "b3_implicit/greedy1000.jsonl", NEW / "b3_implicit/evaluation/dev1000", "b3"),
    "LOVA3 GenQA": (PB / "lova3/predictions.jsonl", PB / "lova3/evaluation/dev1000", "lova3"),
    "Genixer Common-VQA": (PB / "genixer/predictions.jsonl", PB / "genixer/evaluation/dev1000", "genixer"),
}


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.is_file() else []


def seeq_per_image(eval_dir: Path) -> dict[str, dict[str, float]]:
    return {str(r["image_id"]): {k: float(r["scores"][k]) / 5.0 for k in DIMS}
            for r in read(eval_dir / "seeq/scores.jsonl") if r.get("status") == "ok"}


def support_per_image(eval_dir: Path, system: str) -> dict[str, str]:
    return {str(r["image_id"]): r["overall_visual_support"]
            for r in read(eval_dir / f"support/{system}.jsonl")
            if r.get("status") == "ok" and r.get("overall_visual_support")}


def semdiv(eval_dir: Path):
    for x in (eval_dir.parent / "diversity/summary.json", eval_dir / "diversity/summary.json"):
        if x.is_file():
            return json.loads(x.read_text())["systems"]["zero_shot"]["mean"]
    return None


def row(name: str):
    pred, ev, system = SYS[name]
    trad = json.loads((ev / "traditional/system_summary.json").read_text())
    vc = seeq_per_image(ev)
    sup = support_per_image(ev, system)
    return {
        "BLEU-4": trad["corpus_bleu4"], "CIDEr": trad["corpus_cider"], "Q-BLEU4-v": trad["corpus_qbleu4_variant"],
        "Search": float(np.mean([v["visual_search_difficulty"] for v in vc.values()])),
        "Coverage": float(np.mean([v["visual_evidence_coverage"] for v in vc.values()])),
        "Context": float(np.mean([v["visual_context_reasoning"] for v in vc.values()])),
        "Spatial": float(np.mean([v["visual_spatial_reasoning"] for v in vc.values()])),
        "Supported": float(np.mean([s == "supported" for s in sup.values()])),
        "Unsupported": float(np.mean([s == "unsupported" for s in sup.values()])),
        "Uncertain": float(np.mean([s == "uncertain" for s in sup.values()])),
        "Distinct-2": trad["distinct_2"], "Dup": trad["normalized_excess_duplicate_rate"],
        "SemDiv": semdiv(ev), "seeq_valid": len(vc), "support_valid": len(sup),
    }


def boot(diffs: list[float], seed: int = 42, n: int = 5000):
    rng = np.random.default_rng(seed)
    a = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    means = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(fo_name: str, other: str):
    _, fo_ev, fo_sys = SYS[fo_name]
    _, o_ev, o_sys = SYS[other]
    fo_vc, o_vc = seeq_per_image(fo_ev), seeq_per_image(o_ev)
    fo_sup, o_sup = support_per_image(fo_ev, fo_sys), support_per_image(o_ev, o_sys)
    common_vc = sorted(set(fo_vc) & set(o_vc))
    common_sup = sorted(set(fo_sup) & set(o_sup))
    res = {"seeq_common_n": len(common_vc), "support_common_n": len(common_sup), "metrics": {}}
    for d in DIMS:
        res["metrics"][d] = boot([fo_vc[i][d] - o_vc[i][d] for i in common_vc])
    res["metrics"]["supported"] = boot([int(fo_sup[i] == "supported") - int(o_sup[i] == "supported") for i in common_sup])
    res["metrics"]["unsupported"] = boot([int(fo_sup[i] == "unsupported") - int(o_sup[i] == "unsupported") for i in common_sup])
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    table = {name: row(name) for name in SYS}
    (OUT / "table.json").write_text(json.dumps(table, indent=2) + "\n")
    comparisons = {name: paired("Full Ours (base)", name) for name in SYS if name != "Full Ours (base)"}
    (OUT / "paired.json").write_text(json.dumps(comparisons, indent=2) + "\n")

    # 10 fixed-rule examples: ascending image_id, first 10
    pred, ev, system = SYS["Full Ours (base)"]
    fo = {str(r["image_id"]): r["question"] for r in read(pred)}
    vc = seeq_per_image(ev)
    sup = support_per_image(ev, system)
    ids = sorted(fo, key=lambda x: (len(x), x))[:10]
    examples = [{"image_id": i, "question": fo[i],
                 "seeq_mean_over_4dims": round(sum(vc[i].values()) / 4, 4) if i in vc else None,
                 "support": sup.get(i)} for i in ids]
    (OUT / "examples10.json").write_text(json.dumps(examples, indent=2) + "\n")

    print(json.dumps(table, indent=2))
    print("--- paired (full ours - X) ---")
    print(json.dumps(comparisons, indent=2))
    print("--- examples ---")
    print(json.dumps(examples, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
