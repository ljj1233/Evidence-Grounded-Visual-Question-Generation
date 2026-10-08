"""Image-split integrity helpers (Phase-7, Agent 1).

Stdlib only; no repo imports, so this module can be used from scripts,
`vqg.image_splits` (via the ``src/vqg/image_splits.py`` symlink) and tests
without pulling in heavy dependencies.

It reads the frozen manifest written by
``scripts/build_image_split_manifest.py``
(``data/train_resources/image_split_manifest.json``), which lists the official
image ids for every dataset/split. The manifest is treated as immutable.

Public API
----------
``MANIFEST_PATH``
    Absolute path to the frozen manifest (resolved from the repo root).
``load_manifest(path=None)``
    Return the parsed manifest dict (cached per path).
``image_ids(dataset, split, path=None)``
    Set of official image ids for one split, e.g. ``image_ids("gqa", "val")``.
``train_image_ids(dataset, path=None)``
    Set of image ids for the dataset's ``train`` split.
``eval_image_ids(dataset, path=None)``
    Union of the held-out splits (``val``/``test``/``testdev`` where present,
    per the manifest's ``eval_splits``).
``assert_disjoint(a_ids, b_ids, label)``
    Raise ``ValueError`` if two id collections intersect (counts + <=5 examples).
``assert_no_eval_overlap(ids, dataset, label, path=None)``
    Raise ``ValueError`` if ``ids`` intersects the dataset's eval set.
``filter_train_only(rows, dataset, image_key="image_id",
                    split_key="source_split", require_split=True, path=None)``
    Return ``(kept_rows, audit)`` after dropping any row that is not train-only.
``summarize_rows(rows, dataset, ...)``
    Lightweight audit summary of a row list (rows / unique images / eval leak).

``filter_train_only`` semantics
-------------------------------
A row is dropped when:
  * its ``split_key`` is present and its value is not ``"train"``; or
  * its image is in the dataset's eval set.
When ``split_key`` is absent, the row is kept only if its image is a member of
the dataset's official train split; with ``require_split=True`` an unverifiable
row (no split, image not in train) is dropped as ``unknown_split``. Input rows
are never mutated and dropped/kept lists hold the original dict objects.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_HERE = Path(__file__).resolve()


def _find_repo_root() -> Path:
    for candidate in _HERE.parents:
        if (candidate / "data" / "train_resources" / "image_split_manifest.json").is_file():
            return candidate
    # shared/dataset/image_splits.py -> parents[0]=dataset, [1]=shared, [2]=repo
    return _HERE.parents[2]


REPO_ROOT = _find_repo_root()
MANIFEST_PATH = REPO_ROOT / "data" / "train_resources" / "image_split_manifest.json"

_MANIFEST_CACHE: dict[str, dict] = {}


def load_manifest(path: str | Path | None = None) -> dict:
    """Load (and cache) the frozen image-split manifest."""
    resolved = Path(path) if path is not None else MANIFEST_PATH
    key = str(resolved)
    if key not in _MANIFEST_CACHE:
        with resolved.open(encoding="utf-8") as handle:
            _MANIFEST_CACHE[key] = json.load(handle)
    return _MANIFEST_CACHE[key]


def _dataset_payload(dataset: str, path: str | Path | None = None) -> dict:
    manifest = load_manifest(path)
    try:
        return manifest["datasets"][dataset]
    except KeyError as exc:
        raise KeyError(
            f"unknown dataset {dataset!r}; known: {sorted(manifest['datasets'])}") from exc


def image_ids(dataset: str, split: str, path: str | Path | None = None) -> set[str]:
    """Set of image ids for one dataset/split from the manifest."""
    payload = _dataset_payload(dataset, path)
    try:
        return set(payload["splits"][split]["image_ids"])
    except KeyError as exc:
        raise KeyError(
            f"unknown split {split!r} for dataset {dataset!r}; "
            f"known: {sorted(payload['splits'])}") from exc


def train_image_ids(dataset: str, path: str | Path | None = None) -> set[str]:
    """Official training image ids for ``dataset``."""
    return image_ids(dataset, "train", path)


def eval_image_ids(dataset: str, path: str | Path | None = None) -> set[str]:
    """Union of the dataset's held-out (eval) split image ids."""
    payload = _dataset_payload(dataset, path)
    splits: Sequence[str] = payload.get("eval_splits") or [
        s for s in ("val", "test", "testdev") if s in payload["splits"]]
    out: set[str] = set()
    for split in splits:
        out |= set(payload["splits"][split]["image_ids"])
    return out


def _examples(values: Iterable[str], limit: int = 5) -> list[str]:
    return sorted(str(v) for v in values)[:limit]


def assert_disjoint(a_ids: Iterable, b_ids: Iterable, label: str) -> bool:
    """Raise ``ValueError`` if the two id collections intersect."""
    a = {str(v) for v in a_ids}
    b = {str(v) for v in b_ids}
    overlap = a & b
    if overlap:
        raise ValueError(
            f"{label}: {len(overlap)} overlapping image ids; "
            f"examples: {_examples(overlap)}")
    return True


def assert_no_eval_overlap(ids: Iterable, dataset: str, label: str,
                           path: str | Path | None = None) -> bool:
    """Raise ``ValueError`` if ``ids`` intersects ``dataset``'s eval set."""
    eval_ids = eval_image_ids(dataset, path)
    given = {str(v) for v in ids}
    overlap = given & eval_ids
    if overlap:
        raise ValueError(
            f"{label}: {len(overlap)} image ids overlap the {dataset} eval set "
            f"({len(eval_ids)} images); examples: {_examples(overlap)}")
    return True


def filter_train_only(rows: Sequence[Mapping], dataset: str, *,
                      image_key: str = "image_id", split_key: str = "source_split",
                      require_split: bool = True,
                      path: str | Path | None = None) -> tuple[list[Mapping], dict]:
    """Drop non-train rows; return ``(kept_rows, audit)``. Never mutates input."""
    train_ids = train_image_ids(dataset, path)
    eval_ids = eval_image_ids(dataset, path)

    kept: list[Mapping] = []
    dropped_by_split: Counter[str] = Counter()
    eval_overlap_before = 0
    missing_split_rows = 0

    for row in rows:
        raw_image = row.get(image_key)
        image = None if raw_image is None else str(raw_image)
        in_eval = image is not None and image in eval_ids
        if in_eval:
            eval_overlap_before += 1

        split = row[split_key] if split_key in row else None

        if split is not None and str(split) != "train":
            dropped_by_split[str(split)] += 1
            continue
        if in_eval:
            dropped_by_split["eval_overlap"] += 1
            continue
        if split is None:
            missing_split_rows += 1
            if require_split and (image is None or image not in train_ids):
                dropped_by_split["unknown_split"] += 1
                continue
        kept.append(row)

    eval_overlap_after = sum(
        1 for row in kept
        if row.get(image_key) is not None and str(row.get(image_key)) in eval_ids)

    dropped_rows = len(rows) - len(kept)
    audit = {
        "dataset": dataset,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": dropped_rows,
        "dropped_by_split": dict(sorted(dropped_by_split.items())),
        "eval_overlap_before": eval_overlap_before,
        "eval_overlap_after": eval_overlap_after,
        "missing_split_rows": missing_split_rows,
        "require_split": require_split,
    }
    return kept, audit


def summarize_rows(rows: Sequence[Mapping], dataset: str, *,
                   image_key: str = "image_id", split_key: str = "source_split",
                   path: str | Path | None = None) -> dict:
    """Read-only audit summary: rows, unique images, eval leak and split counts."""
    eval_ids = eval_image_ids(dataset, path)

    split_counts: Counter[str] = Counter()
    overlap_split_counts: Counter[str] = Counter()
    images: set[str] = set()
    eval_images: set[str] = set()
    eval_rows = 0

    for row in rows:
        raw_image = row.get(image_key)
        image = None if raw_image is None else str(raw_image)
        split = row.get(split_key)
        split_name = "<missing>" if split is None else str(split)
        split_counts[split_name] += 1
        if image is not None:
            images.add(image)
            if image in eval_ids:
                eval_images.add(image)
                eval_rows += 1
                overlap_split_counts[split_name] += 1

    return {
        "dataset": dataset,
        "input_rows": len(rows),
        "unique_images": len(images),
        "eval_overlap_images": len(eval_images),
        "eval_overlap_rows": eval_rows,
        "split_counts": dict(sorted(split_counts.items())),
        "eval_overlap_split_counts": dict(sorted(overlap_split_counts.items())),
    }
