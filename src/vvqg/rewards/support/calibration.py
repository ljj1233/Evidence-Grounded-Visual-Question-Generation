"""Phase 2.1 visual-support labels and deterministic calibration splits."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

LABELS = {
    "premise_consistency": ("supported", "unsupported", "uncertain"),
    "reference_clarity": ("clear", "ambiguous", "not_applicable", "uncertain"),
    "visual_answerability": ("answerable", "unanswerable", "uncertain"),
    "failure_type": ("none", "missing_entity", "wrong_attribute", "wrong_relation", "ambiguous_reference", "unobservable_target", "external_knowledge_required", "other", "uncertain"),
    "overall_visual_support": ("supported", "unsupported", "uncertain"),
}

def overall_support(premise: str, reference: str, answerability: str) -> str:
    if premise == "unsupported" or reference == "ambiguous" or answerability == "unanswerable":
        return "unsupported"
    if premise == "supported" and reference in ("clear", "not_applicable") and answerability == "answerable":
        return "supported"
    return "uncertain"

def aggregate_requirements(structure: dict, statuses: list[str]) -> dict:
    n_p = len(structure.get("presuppositions", []))
    n_r = len(structure.get("reference_constraints", []))
    p = statuses[:n_p]
    r = statuses[n_p:n_p+n_r]
    a = statuses[n_p+n_r:]
    premise = "unsupported" if "unsupported" in p else "uncertain" if "uncertain" in p else "supported"
    reference = "ambiguous" if "ambiguous" in r else "uncertain" if "uncertain" in r else "clear" if r else "not_applicable"
    answerability = "unanswerable" if "unanswerable" in a else "uncertain" if "uncertain" in a else "answerable"
    return {"premise_consistency": premise, "reference_clarity": reference,
            "visual_answerability": answerability,
            "overall_visual_support": overall_support(premise, reference, answerability)}

def parse_labels(text: str) -> dict:
    data = json.loads(text)
    for key, values in LABELS.items():
        if key != "overall_visual_support" and data.get(key) not in values:
            raise ValueError(f"invalid {key}: {data.get(key)!r}")
    return data

def select_pairs(rows: list[dict]) -> list[dict]:
    by_image = {}
    for row in rows:
        by_image.setdefault(str(row["image_id"]), {})[int(row["generation_id"])] = row
    out = []
    for i, image_id in enumerate(sorted(by_image)):
        out.append(by_image[image_id][i % 4])
    return out

def split_pairs(pairs: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    return {"dev": shuffled[:600], "val": shuffled[600:800], "test": shuffled[800:1000]}

def entropy(values: list[str]) -> float:
    import math
    c = Counter(values); n = len(values)
    return round(-sum(v/n * math.log2(v/n) for v in c.values()), 6) if n else 0.0

def cache_key(image: str, question: str, model: str, prompt_version: str) -> str:
    return hashlib.sha256("\x1f".join((image, question, model, prompt_version)).encode()).hexdigest()
