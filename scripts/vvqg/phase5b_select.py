#!/usr/bin/env python3
"""Phase 5B selectors over a frozen candidate pool.

S0 existing Static-UVFC rule; S1 SupportFirst (weakest-requirement support);
S2 SupportTop2+UVFC.  No tunable weights; candidate pool is fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
VIOLATION = "CONFIRMED_VIOLATION"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tie_hash(image_id: str, generation_id: int) -> str:
    return hashlib.sha256(f"{SEED}:{image_id}#{generation_id}".encode()).hexdigest()


def s0(members: list[dict]) -> dict:
    return sorted(members, key=lambda r: (
        0 if r["flat_state"] != VIOLATION else 1, -float(r["UVFC"]), tie_hash(r["image_id"], r["generation_id"])
    ))[0]


def s1(members: list[dict]) -> dict:
    return sorted(members, key=lambda r: (
        0 if r["flat_state"] != VIOLATION else 1,
        -float(r["support_floor"]),
        -float(r["UVFC"]),
        tie_hash(r["image_id"], r["generation_id"]),
    ))[0]


def s2(members: list[dict]) -> dict:
    admissible = [r for r in members if r["flat_state"] != VIOLATION]
    if not admissible:
        return s0(members)
    if len(admissible) == 1:
        return admissible[0]
    top2 = sorted(admissible, key=lambda r: (
        -float(r["support_floor"]), tie_hash(r["image_id"], r["generation_id"])
    ))[:2]
    return sorted(top2, key=lambda r: (
        -float(r["UVFC"]), tie_hash(r["image_id"], r["generation_id"])
    ))[0]


SELECTORS = {"s0": s0, "s1": s1, "s2": s2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--selector", choices=sorted(SELECTORS), required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    rows = read(args.scored)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_id"])].append(row)
    for members in groups.values():
        members.sort(key=lambda r: int(r["generation_id"]))

    choose = SELECTORS[args.selector]
    selected, baseline = [], []
    for image_id in sorted(groups):
        members = groups[image_id]
        selected.append(choose(members))
        baseline.append(s0(members))

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "greedy1000.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps({
                "image_id": str(row["image_id"]), "image": row["image"], "question": row["question"],
                "generation_id": int(row["generation_id"]), "UVFC": row["UVFC"],
                "flat_state": row["flat_state"], "support_floor": row["support_floor"],
                "source": f"phase5b_{args.selector}",
            }, ensure_ascii=False) + "\n")

    floors = [float(row["support_floor"]) for row in selected]
    changed = sum(
        int(a["generation_id"]) != int(b["generation_id"]) or a["question"] != b["question"]
        for a, b in zip(selected, baseline)
    )
    audit = {
        "selector": args.selector,
        "images": len(selected),
        "candidates": len(rows),
        "mean_candidates_per_image": len(rows) / len(selected),
        "selected_index_distribution": dict(sorted(Counter(int(r["generation_id"]) for r in selected).items())),
        "selected_violation_rate": sum(r["flat_state"] == VIOLATION for r in selected) / len(selected),
        "selected_failed_rate": sum(r["flat_state"] == "FAILED" for r in selected) / len(selected),
        "mean_support_floor": float(np.mean(floors)),
        "p10_support_floor": float(np.percentile(floors, 10)),
        "p25_support_floor": float(np.percentile(floors, 25)),
        "median_support_floor": float(np.median(floors)),
        "mean_uvfc": float(np.mean([float(r["UVFC"]) for r in selected])),
        "fraction_changed_vs_s0": changed / len(selected),
        "selected_candidate_id_sha256": hashlib.sha256(
            "".join(sorted(f"{r['image_id']}#{int(r['generation_id'])}" for r in selected)).encode()
        ).hexdigest(),
    }
    (args.outdir / "selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
