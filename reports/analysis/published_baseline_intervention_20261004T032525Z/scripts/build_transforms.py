#!/usr/bin/env python3
"""Build image-intervention conditions + manifests for development1000.

Conditions: original, blurred, blank, shuffled. region_masked is skipped (no
project region/saliency artifact exists). All randomness is seed 42. Transformed
images are cached once and reused by every baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parents[4]
DEFAULT_DEV = REPO / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl"
BLANK_RGB = (128, 128, 128)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derangement(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fixed = np.flatnonzero(perm == np.arange(n))
    while fixed.size:
        perm[fixed] = np.roll(perm[fixed], 1)
        fixed = np.flatnonzero(perm == np.arange(n))
    return perm


def blur_radius(width: int, height: int) -> int:
    return max(4, round(0.03 * min(height, width)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-manifest", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out = args.out_dir.resolve()
    dev = read_jsonl(args.dev_manifest)
    if args.limit:
        dev = dev[: args.limit]
    for row in dev:
        row["image_id"] = str(row["image_id"])
        row["image_path"] = str(row.get("image_path") or row["image"])

    ids = [row["image_id"] for row in dev]
    paths = [row["image_path"] for row in dev]
    n = len(dev)
    perm = derangement(n, seed=42)
    shuffle_map = {ids[i]: ids[int(perm[i])] for i in range(n)}

    for cond in ("blurred", "blank", "shuffled"):
        (out / "transformed_images" / cond).mkdir(parents=True, exist_ok=True)
    (out / "condition_manifests").mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    blur_stats = []
    for index, (image_id, rel_path) in enumerate(zip(ids, paths)):
        src = REPO / rel_path
        original = Image.open(src).convert("RGB")
        width, height = original.size
        radius = blur_radius(width, height)
        blur_stats.append(radius)

        blurred = original.filter(ImageFilter.GaussianBlur(radius=radius))
        blank = Image.new("RGB", (width, height), BLANK_RGB)
        shuffled = Image.open(REPO / paths[int(perm[index])]).convert("RGB").resize((width, height))

        cols = {"original": rel_path}
        for name, image, suffix in (("blurred", blurred, ".jpg"), ("blank", blank, ".png"), ("shuffled", shuffled, ".jpg")):
            target = out / "transformed_images" / name / f"{image_id}{suffix}"
            if suffix == ".jpg":
                image.save(target, quality=95)
            else:
                image.save(target)
            cols[name] = str(target.relative_to(REPO))
        manifest_rows.append({
            "image_id": image_id,
            "original_path": rel_path,
            "original_sha256": sha256(src),
            "conditions": cols,
            "condition_sha256": {name: sha256(REPO / cols[name]) for name in ("blurred", "blank", "shuffled")},
            "size": [width, height],
            "blur_radius": radius,
        })

    (out / "image_manifest.json").write_text(json.dumps(manifest_rows, indent=2) + "\n", encoding="utf-8")
    (out / "shuffle_mapping.json").write_text(json.dumps(
        {"seed": 42, "derangement": shuffle_map}, indent=2) + "\n", encoding="utf-8")

    condition_rows = {c: [] for c in ("original", "blurred", "blank", "shuffled")}
    for row in manifest_rows:
        for cond in condition_rows:
            path = row["original_path"] if cond == "original" else row["conditions"][cond]
            condition_rows[cond].append({"image_id": row["image_id"], "image_path": path, "condition": cond, "seed": 42})
    for cond, rows in condition_rows.items():
        (out / "condition_manifests" / f"{cond}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    fixed_points = sum(1 for i in range(n) if perm[i] == i)
    print(json.dumps({
        "images": n, "blur_radius_min": min(blur_stats), "blur_radius_max": max(blur_stats),
        "shuffle_fixed_points": fixed_points, "seed": 42,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
