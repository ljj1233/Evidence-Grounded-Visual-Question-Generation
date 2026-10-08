#!/usr/bin/env python3
"""Build the shared 10k-manifest training records used by B2/B3 (one question
per method_train_10000 image, seed 42), and slice the B3 target cache to the
same (image_id, question) pairs so B2/B3 and Full Ours share one data budget.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PB = ROOT / "outputs/gqa_native/published_baselines"
OUT = ROOT / "outputs/gqa_native/full_ours_base/data"


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, default=PB / "records_method10000.jsonl")
    p.add_argument("--b3-targets", type=Path, default=PB / "b3_implicit/cache/train_targets.jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = read(args.records)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(str(row["image_id"]), row)
    selected = [seen[k] for k in sorted(seen, key=lambda x: (len(x), x))]
    records_out = OUT / "records_10k_1per.jsonl"
    records_out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    keys = {(str(r["image_id"]), r["question"]) for r in selected}

    targets = read(args.b3_targets)
    sliced = [t for t in targets if (str(t["image_id"]), t["question"]) in keys]
    targets_out = OUT / "train_targets_10k_1per.jsonl"
    targets_out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sliced))
    print(json.dumps({
        "records_in": len(rows), "records_out": len(selected), "unique_images": len(seen),
        "b3_targets_in": len(targets), "b3_targets_out": len(sliced),
        "records_path": str(records_out), "targets_path": str(targets_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
