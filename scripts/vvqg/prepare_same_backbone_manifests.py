#!/usr/bin/env python3
"""Freeze the permitted same-backbone image universe and disjointness checks."""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

from vvqg.baselines.common import BACKBONE, SEED, assert_disjoint, image_ids, lora_kwargs, read_jsonl, stable_hash, write_jsonl


def ids_from(path: Path, field: str = "image_id") -> set[str]:
    return {str(row[field]) for row in read_jsonl(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/same_backbone_baselines/manifests")
    args = parser.parse_args()
    sft = ROOT / "src/vvqg/data/sft"
    train_rows = read_jsonl(sft / "train.jsonl")
    train_manifest = read_jsonl(sft / "train_manifest.jsonl")
    if len(train_rows) != 50_000:
        raise AssertionError(f"SFT train record count changed: {len(train_rows)}")
    if len(train_manifest) != len(set(str(row["image_id"]) for row in train_manifest)):
        raise AssertionError("train_manifest must contain one row per canonical image_id")
    train_ids = image_ids(train_manifest)
    dev_manifest = ROOT / "outputs/evaluation/sft_greedy/val2014_sample1000.jsonl"
    dev_rows = read_jsonl(dev_manifest)
    dev_by_id = {}
    for row in dev_rows:
        iid = str(int(row["image_id"]))
        dev_by_id.setdefault(iid, {"image_id": iid, "image_path": row["image_path"], "seed": SEED})
    dev_rows = [dev_by_id[i] for i in sorted(dev_by_id)]
    reward_rows = read_jsonl(ROOT / "outputs/reward_calibration/sft_candidates_4x.jsonl")
    reward_ids = image_ids(reward_rows)
    holdout_ids = ids_from(ROOT / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl")
    for name, ids in {"development1000": image_ids(dev_rows), "reward_calibration": reward_ids, "final_holdout": holdout_ids}.items():
        assert_disjoint({"train50k": train_ids, name: ids})
    assert len(dev_rows) == 1000
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "train50k_records.jsonl", train_rows)
    write_jsonl(args.output / "train50k_images.jsonl", sorted(train_manifest, key=lambda row: str(row["image_id"])))
    write_jsonl(args.output / "train_unlabeled_images.jsonl", (
        {"image_id": str(row["image_id"]), "image_path": row["image_path"], "seed": SEED}
        for row in sorted(train_manifest, key=lambda row: int(row["image_id"]))
    ))
    write_jsonl(args.output / "development1000_images.jsonl", sorted(dev_rows, key=lambda row: str(row["image_id"])))
    payload = {
        "schema": "same-backbone-baselines-manifest-v1",
        "seed": SEED,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "base_model": local_model_path('Qwen3-VL-8B-Instruct'),
        "lora": lora_kwargs(),
        "backbone": BACKBONE,
        "train_records": len(train_rows),
        "train_unique_images": len(train_ids),
        "development_images": len(dev_rows),
        "reward_calibration_images": len(reward_ids),
        "final_holdout_images": len(holdout_ids),
        "train_records_sha256": stable_hash(train_rows),
        "train_image_ids_sha256": stable_hash(sorted(train_ids)),
        "development_image_ids_sha256": stable_hash(sorted(image_ids(dev_rows))),
        "intersections": {"train50k_development1000": 0, "train50k_reward_calibration": 0, "train50k_final_holdout": 0},
        "source": {"train": str(sft / "train.jsonl"), "train_manifest": str(sft / "train_manifest.jsonl"), "development": str(dev_manifest)},
    }
    (args.output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
