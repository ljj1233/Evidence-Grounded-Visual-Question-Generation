#!/usr/bin/env python3
"""Phase 5C selectors over a frozen candidate pool.

S0 existing UVFC; S1 SupportFirst; S3 ConstraintFirst (maximize UVFC subject to
no confirmed violation and support_floor >= tau, with S1 fallback).  No tunable
weights; the candidate pool is fixed and tau comes from the frozen calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vvqg"))

from phase5b_select import s0, s1, tie_hash  # noqa: E402

VIOLATION = "CONFIRMED_VIOLATION"


def s3(members: list[dict], tau: float) -> dict:
    admissible = [r for r in members if r["flat_state"] != VIOLATION]
    eligible = [r for r in admissible if float(r["support_floor"]) >= tau]
    if not eligible:
        return s1(members)
    return sorted(eligible, key=lambda r: (-float(r["UVFC"]), tie_hash(r["image_id"], r["generation_id"])))[0]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--selector", choices=["s0", "s1", "s3"], required=True)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    rows = read(args.scored)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_id"])].append(row)
    for members in groups.values():
        members.sort(key=lambda r: int(r["generation_id"]))

    if args.selector == "s3":
        if args.tau is None:
            raise SystemExit("s3 requires --tau")
        choose = lambda members: s3(members, args.tau)  # noqa: E731
    else:
        choose = {"s0": s0, "s1": s1}[args.selector]

    selected, baseline = [], []
    fallback = 0
    for image_id in sorted(groups):
        members = groups[image_id]
        selected.append(choose(members))
        baseline.append(s0(members))
        if args.selector == "s3":
            if not any(r["flat_state"] != VIOLATION and float(r["support_floor"]) >= args.tau for r in members):
                fallback += 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "greedy1000.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps({
                "image_id": str(row["image_id"]), "image": row["image"], "question": row["question"],
                "generation_id": int(row["generation_id"]), "UVFC": row["UVFC"],
                "flat_state": row["flat_state"], "support_floor": row["support_floor"],
                "source": f"phase5c_{args.selector}",
            }, ensure_ascii=False) + "\n")

    floors = [float(r["support_floor"]) for r in selected]
    changed = sum(a["question"] != b["question"] or int(a["generation_id"]) != int(b["generation_id"])
                  for a, b in zip(selected, baseline))
    audit = {
        "selector": args.selector, "tau": args.tau, "images": len(selected), "candidates": len(rows),
        "selected_index_distribution": dict(sorted(Counter(int(r["generation_id"]) for r in selected).items())),
        "selected_violation_rate": sum(r["flat_state"] == VIOLATION for r in selected) / len(selected),
        "mean_support_floor": float(np.mean(floors)),
        "p10_support_floor": float(np.percentile(floors, 10)),
        "p25_support_floor": float(np.percentile(floors, 25)),
        "median_support_floor": float(np.median(floors)),
        "mean_uvfc": float(np.mean([float(r["UVFC"]) for r in selected])),
        "fraction_changed_vs_s0": changed / len(selected),
        "s3_fallback_rate": fallback / len(selected) if args.selector == "s3" else None,
        "selected_sha256": hashlib.sha256(
            "".join(sorted(f"{r['image_id']}#{int(r['generation_id'])}" for r in selected)).encode()).hexdigest(),
    }
    (args.outdir / "selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
