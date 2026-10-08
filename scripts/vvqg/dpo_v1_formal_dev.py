#!/usr/bin/env python3
"""Freeze the fresh 1000-image development split for formal Offline DPO-v1.

Candidate pool = every val2014 image in data/vqav2/processed/val.jsonl.  Excluded
sets (IDs only; final_holdout image/question contents are never read):

  - canonical development1000 == reward_calibration (outputs/evaluation/sft_greedy/val2014_sample1000.jsonl)
  - val_debug (data/vqav2/metadata/val_debug_manifest.jsonl)
  - final_holdout (src/vvqg/data/evaluation/final_holdout_manifest.jsonl)

All prior val2014 evaluations use only development1000 u val_debug, and
train2014 (SFT/UVFC/DPO train) is image-disjoint from val2014, so this yields a
seed-42 frozen dev that no stage of the VQAv2 method has ever consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAL_POOL = ROOT / "data/vqav2/processed/val.jsonl"
CANONICAL_DEV = ROOT / "outputs/evaluation/sft_greedy/val2014_sample1000.jsonl"
VAL_DEBUG = ROOT / "data/vqav2/metadata/val_debug_manifest.jsonl"
HOLDOUT = ROOT / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl"
RAW_VAL = ROOT / "data/vqav2/raw/images/val2014"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_of(path: Path, key: str = "image_id") -> set[str]:
    return {str(row[key]).lstrip("0") or "0" for row in read_jsonl(path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    pool = ids_of(VAL_POOL)
    canonical = ids_of(CANONICAL_DEV)
    val_debug = ids_of(VAL_DEBUG)
    holdout = ids_of(HOLDOUT)

    excluded = canonical | val_debug | holdout
    candidates = sorted((pool - excluded), key=int)
    chosen = sorted(random.Random(args.seed).sample(candidates, args.n), key=int)

    f = lambda v: "0" * (12 - len(v)) + v
    rows = [{"image_id": i, "image_path": f"data/vqav2/raw/images/val2014/COCO_val2014_{f(i)}.jpg", "seed": args.seed}
            for i in chosen]
    missing = [r["image_path"] for r in rows if not (ROOT / r["image_path"]).is_file()]
    if missing:
        raise SystemExit(f"missing {len(missing)} images, e.g. {missing[:3]}")

    (out / "dev1000_images.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def inter(a: set[str], b: set[str]) -> int:
        return len(a & b)

    audit = {
        "dev_n": len(rows),
        "dev_role": "FRESH_DEVELOPMENT",
        "end_to_end_untouched": True,
        "seed": args.seed,
        "val_pool_images": len(pool),
        "intersections": {
            "dev_canonical_development1000": inter(set(chosen), canonical),
            "dev_val_debug": inter(set(chosen), val_debug),
            "dev_final_holdout": inter(set(chosen), holdout),
        },
        "excluded_counts": {"canonical_development1000": len(canonical), "val_debug": len(val_debug),
                            "final_holdout": len(holdout), "union": len(excluded)},
        "candidate_pool_after_exclusion": len(candidates),
        "sources": {
            "val_pool": str(VAL_POOL.relative_to(ROOT)), "val_pool_sha256": sha256_file(VAL_POOL),
            "canonical_dev": str(CANONICAL_DEV.relative_to(ROOT)), "canonical_dev_sha256": sha256_file(CANONICAL_DEV),
            "val_debug": str(VAL_DEBUG.relative_to(ROOT)), "val_debug_sha256": sha256_file(VAL_DEBUG),
            "final_holdout_manifest": str(HOLDOUT.relative_to(ROOT)), "final_holdout_sha256": sha256_file(HOLDOUT),
        },
        "dev_image_ids_sha256": hashlib.sha256(json.dumps(chosen).encode()).hexdigest(),
        "construction": f"random.Random({args.seed}).sample(sorted(val2014 - (dev1000 u val_debug u final_holdout)), {args.n})",
    }
    (out / "DEV_SPLIT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
