#!/usr/bin/env python3
"""Support-Delta GRPO — data resolution and audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stem(value: str) -> str:
    name = Path(str(value)).stem
    return name.split("_")[-1].lstrip("0") or "0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--train-manifest", type=Path, default=ROOT / "outputs/phase4b_online_uvfc/data/train_10000.jsonl")
    ap.add_argument("--pilot-manifest", type=Path, default=ROOT / "reports/vc_qse_20261005T035840Z/dev256_manifest.jsonl")
    ap.add_argument("--dev-manifest", type=Path, default=ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl")
    args = ap.parse_args()

    final_holdout = ROOT / "src/vvqg/data/evaluation/final_holdout_manifest.jsonl"
    for path in (args.train_manifest, args.pilot_manifest, args.dev_manifest):
        if "final_holdout" in str(path).lower():
            raise ValueError(f"final_holdout must not be read: {path}")
    for path in (args.train_manifest, args.pilot_manifest, args.dev_manifest):
        if not path.is_file():
            print(json.dumps({"STATUS": "MISSING_REQUIRED_INPUT", "missing": str(path)}))
            return 1

    train_raw = read_jsonl(args.train_manifest)
    by_id: dict[str, dict] = {}
    for row in train_raw:
        image_id = str(row.get("image_id"))
        by_id.setdefault(image_id, {"image_id": image_id, "image": str(row.get("image") or row.get("image_path")), "seed": 42})
    train = sorted(by_id.values(), key=lambda row: row["image_id"])

    pilot = [{"image_id": str(row["image_id"]), "image_path": str(row.get("image_path") or row.get("image")), "seed": 42}
             for row in read_jsonl(args.pilot_manifest)]
    dev = read_jsonl(args.dev_manifest)

    train_stems = {stem(row["image_id"]) for row in train}
    pilot_stems = {stem(row["image_id"]) for row in pilot}
    dev_stems = {stem(row.get("image_path") or row["image_id"]) for row in dev}

    write_jsonl(args.report_dir / "resolved_train_manifest.jsonl", train)
    write_jsonl(args.report_dir / "resolved_pilot_manifest.jsonl", pilot)

    overlaps = {
        "train_images": len(train_stems),
        "dev1000_images": len(dev_stems),
        "pilot256_images": len(pilot_stems),
        "train_dev_overlap": len(train_stems & dev_stems),
        "train_pilot_overlap": len(train_stems & pilot_stems),
        "pilot_dev_overlap": len(pilot_stems & dev_stems),
    }
    with (args.report_dir / "train_dev_overlap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "overlap_count"])
        for key in ("train_dev_overlap", "train_pilot_overlap", "pilot_dev_overlap"):
            writer.writerow([key, overlaps[key]])

    audit = {
        **overlaps,
        "raw_train_records": len(train_raw),
        "deduped_train_images": len(train),
        "dropped_duplicate_records": len(train_raw) - len(train),
        "unique_image_ids_in_train": len({row["image_id"] for row in train}) == len(train),
        "pilot_manifest": str(args.pilot_manifest.relative_to(ROOT)),
        "pilot_manifest_sha256": sha256(args.pilot_manifest),
        "train_manifest": str(args.train_manifest.relative_to(ROOT)),
        "dev_manifest": str(args.dev_manifest.relative_to(ROOT)),
        "final_holdout_accessed": False,
        "pass": overlaps["train_dev_overlap"] == 0 and overlaps["train_pilot_overlap"] == 0,
    }
    (args.report_dir / "DATA_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
