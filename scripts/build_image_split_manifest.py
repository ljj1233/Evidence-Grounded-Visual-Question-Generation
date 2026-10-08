#!/usr/bin/env python3
"""Build the immutable, deterministic image-split manifest.

Phase-7 (Experimental Correctness Repair, Agent 1). This generator reads the
*official* raw split files (never any derived pool) and writes a single frozen
manifest describing, per dataset/split, the exact sorted list of image ids.

    python3 scripts/build_image_split_manifest.py            # write
    python3 scripts/build_image_split_manifest.py --check    # verify on-disk

Determinism / idempotence
-------------------------
The only field whose value could vary between runs is ``generated_utc``. It is
*derived* (not wall-clock): it is the maximum mtime of the source files, in UTC
with second resolution. Therefore two runs over unchanged inputs produce
byte-identical output; if a source file changes, both its sha256 and the derived
timestamp change. The image-id list is canonicalised as utf-8, one id per line,
sorted lexicographically, with a trailing newline; its sha256 is over those
bytes.

Nothing here mutates existing artifacts, pools, checkpoints or frozen results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "train_resources" / "image_split_manifest.json"

SCHEMA_VERSION = "image-split-manifest-1.0"
IMAGE_ID_LIST_ENCODING = "utf-8, one image id per line, lexicographically sorted, trailing newline"

# dataset -> split -> (official_split, relative source path, image-id field)
SOURCES: dict[str, dict[str, tuple[str, str, str]]] = {
    "gqa": {
        "train": ("train_balanced",
                  "data/raw/gqa/questions1.2/train_balanced_questions.json", "imageId"),
        "val": ("val_balanced",
                "data/raw/gqa/questions1.2/val_balanced_questions.json", "imageId"),
        "test": ("test_balanced",
                 "data/raw/gqa/questions1.2/test_balanced_questions.json", "imageId"),
        "testdev": ("testdev_balanced",
                    "data/raw/gqa/questions1.2/testdev_balanced_questions.json", "imageId"),
    },
    "vqav2": {
        "train": ("train",
                  "data/vqav2/raw/questions/v2_OpenEnded_mscoco_train2014_questions.json",
                  "image_id"),
        "val": ("val",
                "data/vqav2/raw/questions/v2_OpenEnded_mscoco_val2014_questions.json",
                "image_id"),
    },
}

# Splits that count as held-out evaluation for the aligned protocol.
EVAL_SPLITS: dict[str, list[str]] = {
    "gqa": ["val", "test", "testdev"],
    "vqav2": ["val"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_image_ids(path: Path, field: str) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    ids: set[str] = set()
    if field == "imageId":  # GQA: {question_id: {"imageId": "..."}} (balanced + all)
        for record in payload.values():
            value = record.get(field)
            if value is not None:
                ids.add(str(value))
    else:  # VQAv2: {"questions": [{"image_id": ...}, ...]}
        for record in payload.get("questions", []):
            value = record.get(field)
            if value is not None:
                ids.add(str(value))
    return ids


def _canonical_image_list(image_ids: set[str]) -> tuple[str, str]:
    text = "\n".join(sorted(image_ids))
    if text:
        text += "\n"
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest() -> dict:
    datasets: dict[str, dict] = {}
    source_mtimes: list[float] = []
    for dataset, splits in SOURCES.items():
        split_payload: dict[str, dict] = {}
        for split, (official, rel_path, field) in splits.items():
            path = ROOT / rel_path
            if not path.is_file():
                raise SystemExit(f"[{dataset}:{split}] missing source file: {path}")
            source_mtimes.append(path.stat().st_mtime)
            image_ids = _iter_image_ids(path, field)
            _, digest = _canonical_image_list(image_ids)
            split_payload[split] = {
                "official_split": official,
                "source_path": rel_path,
                "source_sha256": sha256_file(path),
                "image_id_field": field,
                "n_images": len(image_ids),
                "image_ids_sha256": digest,
                "image_ids": sorted(image_ids),
            }
        datasets[dataset] = {
            "eval_splits": list(EVAL_SPLITS[dataset]),
            "splits": split_payload,
        }
    # Derived, not wall-clock: max source mtime -> byte-identical re-runs.
    derived_utc = datetime.fromtimestamp(max(source_mtimes), tz=timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": derived_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_utc_derivation": "max mtime of the referenced source files (UTC, seconds)",
        "generator": "scripts/build_image_split_manifest.py",
        "image_id_list_encoding": IMAGE_ID_LIST_ENCODING,
        "datasets": datasets,
    }


def _serialize(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--check", action="store_true",
                    help="verify the on-disk manifest equals a fresh build; write nothing")
    args = ap.parse_args()

    text = _serialize(build_manifest())
    if args.check:
        if not args.out.is_file():
            print(f"MISSING manifest: {args.out}", file=sys.stderr)
            return 1
        existing = args.out.read_text(encoding="utf-8")
        if existing != text:
            print(f"STALE manifest: {args.out}", file=sys.stderr)
            return 1
        print(f"OK manifest up to date: {args.out}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    manifest = json.loads(text)
    for dataset, payload in manifest["datasets"].items():
        counts = {s: v["n_images"] for s, v in payload["splits"].items()}
        print(f"{dataset}: {counts}")
    print(f"wrote {args.out.relative_to(ROOT)} "
          f"({len(text.encode('utf-8'))} bytes, generated_utc={manifest['generated_utc']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
