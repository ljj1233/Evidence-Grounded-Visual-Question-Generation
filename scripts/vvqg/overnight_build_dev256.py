#!/usr/bin/env python3
"""Freeze the deterministic 256-image development subset for the overnight run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: int(r["image_id"]))
    chosen = sorted(random.Random(args.seed).sample(rows, args.n), key=lambda r: int(r["image_id"]))
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in chosen:
            image = row.get("image_path") or row["image"]
            handle.write(json.dumps({"image_id": str(row["image_id"]), "image_path": image,
                                     "image_sha256": sha256_file(ROOT / image), "seed": args.seed}, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(chosen), "source": str(args.source), "source_sha256": sha256_file(args.source),
                      "out": str(out), "out_sha256": sha256_file(out),
                      "construction": f"random.Random({args.seed}).sample(sorted(dev1000), {args.n})"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
