#!/usr/bin/env python3
"""Phase 5C-8: build the grounded positive set and D1/D2 distillation targets.

Positives per image = non-violation candidates with support_floor >= tau, capped
at 3 by UVFC (frozen).  D1 repeats the teacher's best_question for both epochs;
D2 uses two distinct positives (frozen order) when available.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vvqg"))

from phase5c_select import s3  # noqa: E402

VIOLATION = "CONFIRMED_VIOLATION"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--max-positives", type=int, default=3)
    args = parser.parse_args()

    rows = read(args.scored)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_id"])].append(row)
    for members in groups.values():
        members.sort(key=lambda r: int(r["generation_id"]))

    manifest_rows = read(args.manifest)
    missing = [str(r["image_id"]) for r in manifest_rows if str(r["image_id"]) not in groups]
    if missing:
        raise RuntimeError(
            f"scored pool covers {len(groups)} images but the manifest needs "
            f"{len(manifest_rows)}; {len(missing)} image(s) have no candidates "
            f"(e.g. {missing[:5]}). Refusing to build an incomplete D1 target."
        )

    positives: dict[str, list[str]] = {}
    best: dict[str, str] = {}
    for image_id, members in groups.items():
        feasible = [r for r in members if r["flat_state"] != VIOLATION and float(r["support_floor"]) >= args.tau]
        feasible.sort(key=lambda r: -float(r["UVFC"]))
        q = [r["question"] for r in feasible[: args.max_positives]]
        if not q:
            q = [s3(members, args.tau)["question"]]
        positives[image_id] = q
        best[image_id] = s3(members, args.tau)["question"]

    manifest = manifest_rows
    args.outdir.mkdir(parents=True, exist_ok=True)
    d1, d2, teacher = [], [], []
    for record in manifest:
        image_id = str(record["image_id"])
        pos = positives[image_id]
        b = best[image_id]
        d1.append({"image": record["image"], "q_epoch1": b, "q_epoch2": b})
        q1 = pos[0]
        q2 = pos[1] if len(pos) > 1 else pos[0]
        d2.append({"image": record["image"], "q_epoch1": q1, "q_epoch2": q2})
        teacher.append({
            "image_id": image_id, "image": record["image"],
            "candidates": groups[image_id],
            "positive_questions": pos, "best_question": b,
        })
    for name, data in (("d1_single", d1), ("d2_multipos", d2)):
        (args.outdir / f"targets_{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
    (args.outdir / "train_10000.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in teacher), encoding="utf-8")

    counts = defaultdict(int)
    for record in manifest:
        counts[len(positives[str(record["image_id"])])] += 1
    audit = {
        "tau": args.tau, "manifest_records": len(manifest), "unique_images": len(groups),
        "positive_count_distribution": dict(sorted(counts.items())),
        "mean_positives": sum(len(v) for v in positives.values()) / len(positives),
        "records_using_distinct_pair": sum(1 for r in d2 if r["q_epoch1"] != r["q_epoch2"]),
        "d1_unique_targets": len({r["q_epoch1"] for r in d1}),
        "d2_unique_targets": len({r["q_epoch1"] for r in d2} | {r["q_epoch2"] for r in d2}),
    }
    (args.outdir / "target_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
