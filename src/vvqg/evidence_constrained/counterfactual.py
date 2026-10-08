"""Deterministic counterfactual image construction for CER.

Reuses the repository's existing derangement and Gaussian-blur implementation
verbatim (``reports/analysis/published_baseline_intervention_20261004T032525Z/scripts/
build_transforms.py``).  If that implementation is missing the import fails
loudly -- CER must not invent a different transform.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
_TRANSFORM_SRC = REPO_ROOT / "reports/analysis/published_baseline_intervention_20261004T032525Z/scripts/build_transforms.py"
FINAL_HOLDOUT = REPO_ROOT / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl"


def assert_not_final_holdout(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved == FINAL_HOLDOUT.resolve() or "final_holdout" in resolved.name.lower():
        raise ValueError(f"final_holdout must not be read: {resolved}")
    return resolved


def _load_existing_transforms():
    if not _TRANSFORM_SRC.is_file():
        raise FileNotFoundError(f"missing existing transform implementation: {_TRANSFORM_SRC}")
    spec = importlib.util.spec_from_file_location("_cer_existing_transforms", _TRANSFORM_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXISTING = _load_existing_transforms()
derangement = _EXISTING.derangement
blur_radius = _EXISTING.blur_radius


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_path_of(row: dict[str, Any]) -> str:
    return str(row.get("image_path") or row.get("image"))


def dedup_train(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per image_id, original order preserved, equal image weight."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id.setdefault(str(row["image_id"]), row)
    return sorted(by_id.values(), key=lambda row: str(row["image_id"]))


def build_counterfactuals(
    rows: Sequence[dict[str, Any]], out_dir: str | Path, seed: int = 42
) -> dict[str, Any]:
    """Materialise shuffled + blurred counterfactuals for each unique training image."""
    from PIL import Image, ImageFilter

    out = Path(out_dir).resolve()
    (out / "transformed_images" / "shuffled").mkdir(parents=True, exist_ok=True)
    (out / "transformed_images" / "blurred").mkdir(parents=True, exist_ok=True)

    ids = [str(row["image_id"]) for row in rows]
    paths = [image_path_of(row) for row in rows]
    n = len(rows)
    perm = derangement(n, seed=seed)
    shuffle_map = {ids[i]: ids[int(perm[i])] for i in range(n)}

    manifest, radii = [], []
    for index, (image_id, rel_path) in enumerate(zip(ids, paths)):
        src = REPO_ROOT / rel_path
        original = Image.open(src).convert("RGB")
        width, height = original.size
        radius = blur_radius(width, height)
        radii.append(radius)
        shuffled = Image.open(REPO_ROOT / paths[int(perm[index])]).convert("RGB").resize((width, height))
        blurred = original.filter(ImageFilter.GaussianBlur(radius=radius))
        stem = Path(image_id).stem
        shuffled_path = out / "transformed_images" / "shuffled" / f"{stem}.jpg"
        blurred_path = out / "transformed_images" / "blurred" / f"{stem}.jpg"
        shuffled.save(shuffled_path, quality=95)
        blurred.save(blurred_path, quality=95)
        manifest.append({
            "image_id": image_id,
            "original_path": rel_path,
            "shuffled_path": str(shuffled_path.relative_to(REPO_ROOT)),
            "blurred_path": str(blurred_path.relative_to(REPO_ROOT)),
            "original_sha256": sha256(src),
            "shuffled_sha256": sha256(shuffled_path),
            "blurred_sha256": sha256(blurred_path),
            "shuffle_source_image_id": shuffle_map[image_id],
            "size": [width, height],
            "blur_radius": radius,
        })

    write_jsonl(out / "counterfactual_manifest.jsonl", manifest)
    config = {
        "seed": seed,
        "n_images": n,
        "source_manifest_sha256": None,
        "derangement": "reports/analysis/published_baseline_intervention_20261004T032525Z/scripts/build_transforms.py:derangement",
        "blur": "PIL GaussianBlur radius=max(4,round(0.03*min(h,w)))",
        "shuffle_operation": "pixel content of image_id replaced by derangement partner, resized to original size, JPEG q95",
        "shuffle_fixed_points": int(sum(int(perm[i]) == i for i in range(n))),
    }
    (out / "transform_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    summary = {
        "n_images": n,
        "shuffle_fixed_points": config["shuffle_fixed_points"],
        "blur_radius_min": min(radii) if radii else None,
        "blur_radius_max": max(radii) if radii else None,
        "seed": seed,
        "shuffled_written": n,
        "blurred_written": n,
    }
    (out / "counterfactual_generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
