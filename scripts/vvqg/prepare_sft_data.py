#!/usr/bin/env python3
"""Prepare the Phase-1 VQAv2 SFT dataset for VQG (`Image -> Question`).

Input : raw VQAv2 question JSON (`train2014` / `val2014`) + local COCO images.
Output: `train.jsonl` (50k) + `dev.jsonl` + `test.jsonl` records `{image, question}`,
        image manifests (`{image_id, image_path, split}`) and a provenance
        `dataset_manifest.json`, all under `src/vvqg/data/sft/`.

Design:
- Splits are **image-disjoint** (an image's questions never cross splits).
- train/dev/test are drawn from **train2014** only; the whole `val2014` pool and
  the frozen `val_debug` benchmark are reserved for evaluation (no leakage).
- Fixed seed; deterministic given the same inputs.

No reward labels, no preference pairs, no API calls, no training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

TRAIN_QUESTIONS = "data/vqav2/raw/questions/v2_OpenEnded_mscoco_train2014_questions.json"
VAL_QUESTIONS = "data/vqav2/raw/questions/v2_OpenEnded_mscoco_val2014_questions.json"
TRAIN_IMAGE_DIR = "data/vqav2/raw/images/train2014"
VAL_IMAGE_DIR = "data/vqav2/raw/images/val2014"
EVAL_MANIFEST = "data/train_resources/image_split_manifest.json"
HOLDOUT = "data/vqav2/splits/val_debug.jsonl"
SCHEMA_VERSION = "vvqg-sft-dataset-manifest-1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_image_path(image_id: str) -> str:
    return f"{TRAIN_IMAGE_DIR}/COCO_train2014_{int(image_id):012d}.jpg"


def load_questions(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, str]] = []
    for item in payload["questions"]:
        records.append(
            {
                "image_id": str(item["image_id"]),
                "question_id": str(item["question_id"]),
                "question": str(item["question"]).strip(),
            }
        )
    return records


def group_by_image(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for record in records:
        grouped.setdefault(record["image_id"], []).append(record)
    for image_id in grouped:
        grouped[image_id].sort(key=lambda r: r["question_id"])
    return grouped


def load_eval_image_ids(manifest_path: Path, holdout_path: Path) -> set[str]:
    eval_ids: set[str] = set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vqav2 = manifest.get("datasets", {}).get("vqav2", {})
    for name in vqav2.get("eval_splits", []):
        spec = vqav2.get("splits", {}).get(name, {})
        eval_ids.update(str(i) for i in spec.get("image_ids", []))
    if holdout_path.is_file():
        with holdout_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    eval_ids.add(str(json.loads(line)["image_id"]))
    return eval_ids


def take_split(
    grouped: dict[str, list[dict[str, str]]],
    image_ids: list[str],
    target: int,
) -> tuple[list[dict[str, str]], list[str]]:
    samples: list[dict[str, str]] = []
    used: list[str] = []
    for image_id in image_ids:
        if len(samples) >= target:
            break
        remaining = target - len(samples)
        for record in grouped[image_id][:remaining]:
            samples.append(record)
        used.append(image_id)
    return samples, used


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the VQAv2 SFT dataset.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "src" / "vvqg" / "data" / "sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--eval-manifest", type=Path, default=REPO_ROOT / EVAL_MANIFEST)
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / HOLDOUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_questions_path = REPO_ROOT / TRAIN_QUESTIONS
    val_questions_path = REPO_ROOT / VAL_QUESTIONS

    records = load_questions(train_questions_path)
    grouped = group_by_image(records)
    eval_ids = load_eval_image_ids(args.eval_manifest, args.holdout)

    image_ids = sorted(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(image_ids)

    safe_ids = [i for i in image_ids if i not in eval_ids]
    excluded_overlap = len(image_ids) - len(safe_ids)

    train_samples, train_images = take_split(grouped, safe_ids, args.train_size)
    remaining = [i for i in safe_ids if i not in set(train_images)]
    dev_samples, dev_images = take_split(grouped, remaining, args.dev_size)
    remaining = [i for i in remaining if i not in set(dev_images)]
    test_samples, test_images = take_split(grouped, remaining, args.test_size)

    splits = {
        "train": (train_samples, train_images),
        "dev": (dev_samples, dev_images),
        "test": (test_samples, test_images),
    }

    split_meta: dict[str, Any] = {}
    for name, (samples, images) in splits.items():
        dataset_rows = [{"image": train_image_path(r["image_id"]), "question": r["question"]} for r in samples]
        manifest_rows = [
            {"image_id": image_id, "image_path": train_image_path(image_id), "split": name}
            for image_id in images
        ]
        dataset_path = out_dir / f"{name}.jsonl"
        manifest_path = out_dir / f"{name}_manifest.jsonl"
        write_jsonl(dataset_path, dataset_rows)
        write_jsonl(manifest_path, manifest_rows)
        split_meta[name] = {
            "samples": len(dataset_rows),
            "images": len(manifest_rows),
            "dataset_file": dataset_path.name,
            "manifest_file": manifest_path.name,
            "dataset_sha256": sha256_file(dataset_path),
            "manifest_sha256": sha256_file(manifest_path),
        }

    provenance = {
        "schema": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Image -> Question",
        "seed": args.seed,
        "source": {
            "dataset": "VQAv2",
            "train_questions": {
                "path": TRAIN_QUESTIONS,
                "sha256": sha256_file(train_questions_path),
                "n_questions": len(records),
                "usage": "sampled for train/dev/test",
            },
            "val_questions": {
                "path": VAL_QUESTIONS,
                "sha256": sha256_file(val_questions_path),
                "usage": "reserved_for_evaluation (not used for SFT)",
            },
            "eval_image_manifest": {"path": EVAL_MANIFEST, "sha256": sha256_file(args.eval_manifest)},
            "holdout": {"path": HOLDOUT, "sha256": sha256_file(args.holdout)},
            "images_dir": TRAIN_IMAGE_DIR,
        },
        "splits": split_meta,
        "excluded": {
            "eval_images": len(eval_ids),
            "train2014_images_overlapping_eval": excluded_overlap,
        },
        "notes": [
            "Records contain only {image, question} (no answer/rationale/reward/evidence).",
            "Splits are image-disjoint; each image belongs to exactly one split.",
            "val2014 and the frozen val_debug benchmark are reserved for evaluation.",
        ],
    }
    provenance_path = out_dir / "dataset_manifest.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("== prepare_sft_data ==")
    print(f"seed={args.seed} eval_images={len(eval_ids)} overlap_excluded={excluded_overlap}")
    for name, meta in split_meta.items():
        print(f"{name:5s}: {meta['samples']:>6d} samples / {meta['images']:>5d} images -> {meta['dataset_file']}")
    print(f"provenance -> {provenance_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
