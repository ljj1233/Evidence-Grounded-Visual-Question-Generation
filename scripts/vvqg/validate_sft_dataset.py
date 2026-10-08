#!/usr/bin/env python3
"""Validate the Phase-1 VQAv2 SFT dataset and its image manifests.

Checks (per split):
- schema: every record matches `src/vvqg/data/sft/schema.json` (`{image, question}`);
- question non-empty;
- image file exists on disk;
- split correct: manifest `split` matches the file name and dataset/manifest agree;
- no eval leakage: train/dev/test image ids are disjoint from the reserved
  evaluation pool (vqav2 `val` + frozen `val_debug`), and splits are pairwise
  image-disjoint.

Read-only. Exit codes: 0 = all checks pass, 1 = validation errors, 2 = usage/IO.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_MANIFEST = "data/train_resources/image_split_manifest.json"
HOLDOUT = "data/vqav2/splits/val_debug.jsonl"
SPLITS = ("train", "dev", "test")
MAX_ERRORS = 50


def _load_schema_validator() -> Any:
    path = REPO_ROOT / "scripts" / "new_vqg" / "check_dataset.py"
    spec = importlib.util.spec_from_file_location("new_vqg_check_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eval_image_ids(manifest_path: Path, holdout_path: Path) -> set[str]:
    eval_ids: set[str] = set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vqav2 = manifest.get("datasets", {}).get("vqav2", {})
    for name in vqav2.get("eval_splits", []):
        spec = vqav2.get("splits", {}).get(name, {})
        eval_ids.update(str(i) for i in spec.get("image_ids", []))
    if holdout_path.is_file():
        for row in read_jsonl(holdout_path):
            eval_ids.add(str(row["image_id"]))
    return eval_ids


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the VQAv2 SFT dataset.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "src" / "vvqg" / "data" / "sft")
    parser.add_argument("--eval-manifest", type=Path, default=REPO_ROOT / EVAL_MANIFEST)
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / HOLDOUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir: Path = args.data_dir
    schema_path = data_dir / "schema.json"
    validator = _load_schema_validator()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    eval_ids = load_eval_image_ids(args.eval_manifest, args.holdout)

    errors: list[str] = []
    split_images: dict[str, set[str]] = {}
    split_image_ids: dict[str, set[str]] = {}
    summary: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        dataset_path = data_dir / f"{split}.jsonl"
        manifest_path = data_dir / f"{split}_manifest.jsonl"
        if not dataset_path.is_file() or not manifest_path.is_file():
            errors.append(f"{split}: missing {dataset_path.name} or {manifest_path.name}")
            continue

        records = read_jsonl(dataset_path)
        manifest = read_jsonl(manifest_path)
        images: set[str] = set()
        manifest_paths: set[str] = set()
        manifest_image_ids: set[str] = set()

        for lineno, record in enumerate(records, 1):
            for message in validator.validate_instance(record, schema):
                errors.append(f"{split}.jsonl:{lineno}: {message}")
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                errors.append(f"{split}.jsonl:{lineno}: empty question")
            image = record.get("image")
            if isinstance(image, str) and not (REPO_ROOT / image).is_file():
                errors.append(f"{split}.jsonl:{lineno}: image not found: {image}")
            if isinstance(image, str):
                images.add(image)

        for lineno, row in enumerate(manifest, 1):
            if row.get("split") != split:
                errors.append(f"{split}_manifest.jsonl:{lineno}: split={row.get('split')!r} != {split!r}")
            image_path = row.get("image_path")
            if not isinstance(image_path, str) or not (REPO_ROOT / image_path).is_file():
                errors.append(f"{split}_manifest.jsonl:{lineno}: image_path not found: {image_path}")
            manifest_paths.add(image_path)
            if "image_id" in row:
                manifest_image_ids.add(str(row["image_id"]))

        if images != manifest_paths:
            only_dataset = sorted(images - manifest_paths)[:3]
            only_manifest = sorted(manifest_paths - images)[:3]
            errors.append(
                f"{split}: dataset/manifest image mismatch "
                f"(dataset_only={only_dataset}, manifest_only={only_manifest})"
            )

        leaked = manifest_image_ids & eval_ids
        if leaked:
            errors.append(f"{split}: eval leakage on {len(leaked)} image(s), e.g. {sorted(leaked)[:3]}")

        split_images[split] = set(images)
        split_image_ids[split] = manifest_image_ids
        summary[split] = {"samples": len(records), "images": len(manifest)}

    for a in range(len(SPLITS)):
        for b in range(a + 1, len(SPLITS)):
            left, right = SPLITS[a], SPLITS[b]
            overlap = split_image_ids.get(left, set()) & split_image_ids.get(right, set())
            if overlap:
                errors.append(f"{left}/{right}: not image-disjoint ({len(overlap)} shared image(s))")

    print("== validate_sft_dataset ==")
    print(f"eval pool: {len(eval_ids)} image(s)")
    for split in SPLITS:
        if split in summary:
            print(f"{split:5s}: {summary[split]['samples']:>6d} samples / {summary[split]['images']:>5d} images")

    if errors:
        for message in errors[:MAX_ERRORS]:
            print(f"ERROR: {message}")
        if len(errors) > MAX_ERRORS:
            print(f"... {len(errors) - MAX_ERRORS} more error(s) suppressed")
        print(f"FAIL: {len(errors)} error(s)")
        return 1
    print("OK: schema, questions, images, splits, no eval leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
