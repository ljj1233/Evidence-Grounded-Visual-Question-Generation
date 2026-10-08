"""Shared reporting helpers for Phase 4A comparison tables and paired bootstraps.

Consolidates the previously duplicated read/normalize/dimension/support/system-row
and bootstrap-pair logic used by the phase4a2 / phase4a3 comparison scripts. No
metric semantics change: identical normalization (NFKC + casefold), identical
image-id alignment (sorted shared ids), identical 5,000-resample seed-42 paired
bootstrap.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np

from vvqg.evaluation.common.bootstrap import paired_bootstrap_diff

DIMS = {
    "Search": "visual_search_difficulty",
    "Coverage": "visual_evidence_coverage",
    "Context": "visual_context_reasoning",
    "Spatial": "visual_spatial_reasoning",
}
QBLEU = "corpus_qbleu4_variant"


def norm_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def read_json(path: Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def dim_vector(seeq_path: Path, field: str) -> dict[str, float]:
    return {str(r["image_id"]): r["scores"][field] / 5.0 for r in read_jsonl(seeq_path) if r.get("status") == "ok" and "scores" in r}


def support_vector(support_path: Path, label: str) -> dict[str, float]:
    return {str(r["image_id"]): float(r.get("overall_visual_support") == label) for r in read_jsonl(support_path) if r.get("status") == "ok"}


def system_row(paths: dict) -> dict:
    greedy = read_jsonl(paths["greedy"])
    traditional = read_json(paths["traditional"])
    row = {
        "BLEU": traditional.get("corpus_bleu4"),
        "CIDEr": traditional.get("corpus_cider"),
        "QBLEU": traditional.get(QBLEU),
        "Dup": traditional.get("normalized_excess_duplicate_rate"),
        "Unique": len({norm_text(r["question"]) for r in greedy}) / len(greedy) if greedy else None,
        "n": len(greedy),
    }
    for label, field in DIMS.items():
        values = dim_vector(paths["seeq"], field)
        row[label] = float(np.mean(list(values.values()))) if values else None
    for label in ("supported", "unsupported"):
        values = support_vector(paths["support"], label)
        row[label.capitalize()] = float(np.mean(list(values.values()))) if values else None
    row["SemDiv"] = read_json(paths["diversity"]).get("systems", {}).get("sft", {}).get("mean")
    return row


def bootstrap_pair(left_paths: dict, right_paths: dict, resamples: int = 5000, seed: int = 42) -> dict:
    entry = {}
    for label, field in DIMS.items():
        left, right = dim_vector(left_paths["seeq"], field), dim_vector(right_paths["seeq"], field)
        keys = sorted(set(left) & set(right))
        entry[label] = paired_bootstrap_diff([left[k] for k in keys], [right[k] for k in keys], resamples, seed) if keys else None
    for label in ("supported", "unsupported"):
        left, right = support_vector(left_paths["support"], label), support_vector(right_paths["support"], label)
        keys = sorted(set(left) & set(right))
        entry[label.capitalize()] = paired_bootstrap_diff([left[k] for k in keys], [right[k] for k in keys], resamples, seed) if keys else None
    return entry
