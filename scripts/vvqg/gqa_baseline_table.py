#!/usr/bin/env python3
"""Assemble the GQA published-baseline table and completeness from frozen artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PB = ROOT / "outputs/gqa_native/published_baselines"
OUT = PB / "comparison"
COLUMNS = ["BLEU-4", "CIDEr", "Q-BLEU4-v", "Search", "Coverage", "Context", "Spatial",
           "Supported", "Unsupported", "Uncertain", "Distinct-2", "Dup", "SemDiv"]
DIRS = {"seeq": "seeq", "lova3": "lova3", "genixer": "genixer", "b2": "b2_contrastive", "b3": "b3_implicit"}
LABELS = {
    "seeq": "A1 Faithful SeeQ-style Reproduction (GQA-native)",
    "b2": "B2 ConVQG-inspired Contrastive Grounding (GQA-native)",
    "b3": "B3 Guiding-VQG Implicit Guided (GQA-native)",
    "lova3": "LOVA3 GenQA (released external checkpoint)",
    "genixer": "Genixer-L Common-VQA Generator (released external checkpoint)",
}
VQAV2 = {"seeq": "Faithful SeeQ M2", "lova3": "LOVA3 GenQA", "genixer": "Genixer Common-VQA"}
DIMENSIONS = ("visual_search_difficulty", "visual_evidence_coverage", "visual_context_reasoning", "visual_spatial_reasoning")


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


def d(system: str) -> Path:
    return PB / DIRS[system]


def eval_dir(system: str) -> Path:
    return d(system) / "evaluation/dev1000"


def predictions_path(system: str) -> Path:
    if system == "seeq":
        return d(system) / "evaluation/dev1000/predictions.jsonl"
    if system in ("b2", "b3"):
        return d(system) / "greedy1000.jsonl"
    return d(system) / "predictions.jsonl"


def semdiv(system: str):
    x = load(d(system) / "evaluation/diversity/summary.json")
    return x["systems"]["zero_shot"]["mean"] if x else None


def visual_centric(system: str):
    rows = [r for r in load_jsonl(eval_dir(system) / "seeq/scores.jsonl") if r.get("status") == "ok"]
    if not rows:
        return None, 0
    return {k: sum(r["scores"][k] for r in rows) / len(rows) / 5.0 for k in DIMENSIONS}, len(rows)


def visual_support(system: str):
    raw = load_jsonl(eval_dir(system) / f"support/{system}.jsonl")
    rows = [r for r in raw if r.get("status") == "ok"]
    if not rows:
        return None, 0, len(raw)
    return {lab: sum(r.get("overall_visual_support") == lab for r in rows) / len(rows)
            for lab in ("supported", "unsupported", "uncertain")}, len(rows), len(raw)


def traditional(system: str):
    return load(eval_dir(system) / "traditional/system_summary.json")


def row(system: str):
    t, vc, vs = traditional(system), visual_centric(system)[0], visual_support(system)[0]
    if t is None or vc is None or vs is None:
        return None
    return {
        "BLEU-4": t["corpus_bleu4"], "CIDEr": t["corpus_cider"], "Q-BLEU4-v": t["corpus_qbleu4_variant"],
        "Search": vc["visual_search_difficulty"], "Coverage": vc["visual_evidence_coverage"],
        "Context": vc["visual_context_reasoning"], "Spatial": vc["visual_spatial_reasoning"],
        "Supported": vs["supported"], "Unsupported": vs["unsupported"], "Uncertain": vs["uncertain"],
        "Distinct-2": t["distinct_2"], "Dup": t["normalized_excess_duplicate_rate"], "SemDiv": semdiv(system),
    }


def parser_audit(system: str) -> dict:
    preds = load_jsonl(predictions_path(system))
    empty = sum(not str(r.get("question", "")).strip() for r in preds)
    malformed = empty if system in ("seeq", "b2", "b3") else sum(not str(r.get("question", "")).strip().endswith("?") for r in preds)
    return {"generations": len(preds), "unique_image_ids": len({str(r["image_id"]) for r in preds}),
            "empty": empty, "malformed": malformed,
            "pass": len(preds) == 1000 and empty == 0 and malformed == 0}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, completeness = {}, {}
    for system, label in LABELS.items():
        rows[label] = row(system)
        _, seeq_n = visual_centric(system)
        _, support_ok, support_n = visual_support(system)
        pa = parser_audit(system)
        present = [] if rows[label] is None else [c for c in COLUMNS if rows[label].get(c) is not None]
        complete = pa["pass"] and seeq_n == 1000 and support_ok == 1000 and len(present) == len(COLUMNS)
        completeness[system] = {
            "label": label, "generation": f"{pa['generations']}/1000", "seeq": f"{seeq_n}/1000",
            "support": f"{support_ok}/{support_n}", "support_residual_failures": support_n - support_ok,
            "present": len(present), "total": len(COLUMNS),
            "missing": [c for c in COLUMNS if c not in present], "parser_audit": pa, "complete": complete,
        }
    status = "COMPLETE" if all(c["complete"] for c in completeness.values()) else "INCOMPLETE"

    vq = load(ROOT / "outputs/published_vqg_baselines/comparison/table.json")
    cross = {}
    if vq:
        for system, label in LABELS.items():
            v = vq["rows"].get(VQAV2.get(system, ""))
            g = rows.get(label)
            if v and g:
                cross[label] = {k: {"VQAv2": v.get(k), "GQA": g.get(k)}
                                for k in ("Search", "Coverage", "Context", "Spatial", "Supported", "Unsupported", "Uncertain")}

    (OUT / "table.json").write_text(json.dumps({
        "schema": "gqa-published-vqg-comparison-v1",
        "unit": "GQA dev1000; corpus BLEU/CIDEr/Q-BLEU4-v/Distinct-2/Dup; image-mean SeeQ dims and GPT-4o support rates (valid-only denominators; see completeness.json)",
        "columns": COLUMNS, "rows": rows, "labels": LABELS, "status": status,
        "note": "Rows A1/B2/B3 are GQA-native same-backbone adaptations; LOVA3/Genixer are released external checkpoints (not controlled). Compare only within the same GQA split.",
    }, indent=2) + "\n")
    (OUT / "completeness.json").write_text(json.dumps({
        "schema": "gqa-published-vqg-completeness-v1", "status": status, "rows": completeness,
        "cross_dataset_descriptive": cross,
    }, indent=2) + "\n")
    print(json.dumps({"status": status}, indent=2))
    for k, v in completeness.items():
        print(k, "gen", v["generation"], "seeq", v["seeq"], "support", v["support"], "residual", v["support_residual_failures"], "complete", v["complete"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
