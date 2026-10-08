#!/usr/bin/env python3
"""Phase 4A: rejection-sampling selection from the frozen train-only candidate pool.

Candidate pool: outputs/phase3a_grpo_pilot/preflight/{rollouts_uvfc,rollouts_flat}.jsonl
(2,000 train2014 images x 4; UVFC + Flat state; image-disjoint from development1000
and final_holdout).  Selection: UVFC-only (A) and UVFC+violation (B).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRE = ROOT / "outputs/phase3a_grpo_pilot/preflight"
OUT = ROOT / "outputs/phase4a_uvfc_rft"
SEED = 42
ADMISSIBLE = {"NO_CONFIRMED_VIOLATION", "UNCERTAIN", "FAILED"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tie_hash(image_id: str, generation_id: int) -> str:
    return hashlib.sha256(f"{SEED}:{image_id}#{generation_id}".encode()).hexdigest()


def select(candidates: list[dict], violation_first: bool) -> dict:
    ordered = sorted(
        candidates,
        key=lambda row: (
            0 if (not violation_first or row["state"] in ADMISSIBLE) else 1,
            -float(row["UVFC"]),
            tie_hash(row["image_id"], int(row["generation_id"])),
        ),
    )
    return ordered[0]


def main() -> int:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    uvfc = read(PRE / "rollouts_uvfc.jsonl")
    flat = {}
    for row in read(PRE / "rollouts_flat.jsonl"):
        flat[(row["image"], int(row["generation_id"]))] = row["state"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in uvfc:
        groups[row["image"]].append({**row, "state": flat[(row["image"], int(row["generation_id"]))]})
    print(f"images {len(groups)} candidates {sum(len(v) for v in groups.values())}")

    audit = {}
    for name, violation_first in (("uvfc_only", False), ("uvfc_violation", True)):
        chosen, all_uvfc = [], []
        for image in sorted(groups):
            members = groups[image]
            all_uvfc.extend(float(m["UVFC"]) for m in members)
            chosen.append(select(members, violation_first))
        mean_uvfc = sum(float(c["UVFC"]) for c in chosen) / len(chosen)
        random_uvfc = sum(all_uvfc) / len(all_uvfc)
        rows = [
            {
                "image_id": c["image"],
                "image": c["image"],
                "question": c["question"],
                "generation_id": int(c["generation_id"]),
                "UVFC": float(c["UVFC"]),
                "flat_state": c["state"],
                "candidate_id": f"{c['image']}#{int(c['generation_id'])}",
            }
            for c in chosen
        ]
        (OUT / f"selected_{name}_rft.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        (OUT / "data" / f"{name}_train.jsonl").write_text(
            "".join(json.dumps({"image": r["image"], "question": r["question"]}, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        audit[name] = {
            "images": len(rows),
            "mean_uvfc": mean_uvfc,
            "random_from_4_mean_uvfc": random_uvfc,
            "mean_uvfc_gain_vs_random": mean_uvfc - random_uvfc,
            "flat_state_counts": dict(Counter(r["flat_state"] for r in rows)),
            "unique_questions": len({r["question"] for r in rows}),
            "selected_candidate_id_sha256": hashlib.sha256("".join(sorted(r["candidate_id"] for r in rows)).encode()).hexdigest(),
            "supported_unsupported_labels": "unavailable (train2014 pool has no independent GPT-4o labels)",
            "offline_luna_search_coverage": "unavailable (train2014 pool was not Luna-evaluated)",
        }
    audit["candidate_source"] = {
        "pool": "outputs/phase3a_grpo_pilot/preflight/rollouts_{uvfc,flat}.jsonl",
        "images": len(groups),
        "candidates": sum(len(v) for v in groups.values()),
        "note": "spec's reward_calibration pool equals development1000; replaced with image-disjoint train-only pool to avoid leakage",
        "seed": SEED,
    }
    (OUT / "selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
