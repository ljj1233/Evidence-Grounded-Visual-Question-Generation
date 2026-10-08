#!/usr/bin/env python3
"""Phase 4B data preparation: feasibility (2000) and main (10000) train manifests.

The SFT train split is train2014 only; development1000, the reward-calibration
pool and final_holdout are val2014, so disjointness is structural.  The repo has
9,184 unique train2014 images (not 10,000), so the "10000 SFT train images"
manifest is 10,000 records covering all 9,184 unique images plus 816 repeats.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/phase4b_online_uvfc/data"
TRAIN = ROOT / "src/vvqg/data/sft/train.jsonl"
FEASIBILITY = ROOT / "outputs/phase3a_grpo_pilot/preflight/train_subset_2000.jsonl"
EXCLUDED = {
    "development1000": ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl",
    "reward_calibration": ROOT / "outputs/evaluation/sft_greedy/val2014_sample1000.jsonl",
    "final_holdout": ROOT / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl",
}
SEED = 42


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_set(path: Path) -> set[str]:
    return {row.get("image_path") or row["image"] for row in read(path)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    excluded_images = set().union(*(image_set(path) for path in EXCLUDED.values()))

    feasibility = [
        {"image_id": row["image_id"], "image": row["image"], "question": row["question"]}
        for row in read(FEASIBILITY)
    ]
    (OUT / "train_2000.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in feasibility), encoding="utf-8"
    )

    rows = read(TRAIN)
    by_image: dict[str, list[dict]] = {}
    for row in rows:
        by_image.setdefault(row["image"], []).append(row)
    rng = random.Random(SEED)
    images = sorted(by_image)
    rng.shuffle(images)
    main_rows = [rng.choice(by_image[image]) for image in images]
    extras = rng.sample(rows, 10_000 - len(main_rows)) if len(main_rows) < 10_000 else []
    main_rows.extend(extras)
    rng.shuffle(main_rows)
    manifest = [{"image_id": row["image"], "image": row["image"], "question": row["question"]} for row in main_rows]
    (OUT / "train_10000.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8"
    )

    for name, path in (("feasibility", OUT / "train_2000.jsonl"), ("main", OUT / "train_10000.jsonl")):
        parsed = read(path)
        images = {row["image"] for row in parsed}
        overlap = {key: len(images & image_set(other)) for key, other in EXCLUDED.items()}
        stats = {
            "records": len(parsed),
            "unique_images": len(images),
            "excluded_overlap": overlap,
            "source": str(TRAIN.relative_to(ROOT)),
            "seed": SEED,
        }
        (OUT / f"{name}_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(name, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
