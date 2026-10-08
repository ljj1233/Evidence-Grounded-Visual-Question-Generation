"""VC_SELECTOR — concept-balanced candidate selection (not a baseline).

Rules (fixed, no tuning): drop confirmed-unsupported and adherence=no; prefer
supported over uncertain; prefer higher concept confidence; no duplicate
question within a concept type; keep at most one candidate per concept type per
image; fall back to Direct-4 when no concept candidate survives. The final
single question is the highest-confidence supported candidate (uncertain only if
no supported), with a deterministic sha256 tie-break.
"""

from __future__ import annotations

import hashlib

SUPPORTED = "NO_CONFIRMED_VIOLATION"
UNSUPPORTED = "CONFIRMED_VIOLATION"
UNCERTAIN = "UNCERTAIN"
TYPE_ORDER = ("object", "attribute", "relation", "scene")
_TYPE_KEY = {"object": "objects", "attribute": "attributes", "relation": "relations", "scene": "scene"}
_RANK = {SUPPORTED: 2.0, UNCERTAIN: 1.0, "FAILED": 0.5}


def _tie(row: dict) -> str:
    key = f"{row.get('image_id')}:{row.get('generation_id')}:{row.get('question')}"
    return hashlib.sha256(key.encode()).hexdigest()


def _concept(row: dict) -> tuple:
    return (row.get("concept_type", "?"), (row.get("concept_text") or "").lower())


def eligible(rows: list[dict]) -> list[dict]:
    keep = [
        r for r in rows
        if r.get("flat_state") != UNSUPPORTED and r.get("concept_adherence") != "no"
    ]
    out, seen_concept, seen_question = [], set(), {}
    for r in sorted(
        keep,
        key=lambda r: (-_RANK.get(r.get("flat_state"), 0.0), -float(r.get("concept_confidence", 0.0)), _tie(r)),
    ):
        ckey = _concept(r)
        if ckey in seen_concept:
            continue
        qkey = (r.get("concept_type", "?"), (r.get("question") or "").strip().lower())
        if qkey in seen_question:
            continue
        seen_concept.add(ckey)
        seen_question[qkey] = True
        out.append(r)
    return out


def select_image(rows: list[dict], direct4_fallback: dict | None = None) -> dict:
    """Row dict maps one concept candidate; falls back to Direct-4 if empty."""
    kept = eligible(rows)
    by_type, type_used = [], set()
    for r in kept:
        ctype = r.get("concept_type")
        if ctype in type_used or ctype not in TYPE_ORDER:
            continue
        type_used.add(ctype)
        by_type.append(r)
    supported = [r for r in by_type if r.get("flat_state") == SUPPORTED]
    uncertain = [r for r in by_type if r.get("flat_state") == UNCERTAIN]
    winner = max(
        supported or uncertain or [],
        key=lambda r: (float(r.get("concept_confidence", 0.0)), _tie(r)),
        default=None,
    )
    fallback = winner is None
    final = direct4_fallback if fallback else winner
    return {
        "selected": by_type,
        "final": final,
        "fallback": fallback,
        "n_eligible": len(kept),
    }
