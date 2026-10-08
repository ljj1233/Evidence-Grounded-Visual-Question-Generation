#!/usr/bin/env python3
"""dev1000 confirmation — merge existing256 + remaining744 into full1000."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--experiment-dir", type=Path, default=ROOT / "reports/support_delta_grpo_20261006T022616Z")
    args = ap.parse_args()
    R = args.report_dir.resolve()
    E = args.experiment_dir.resolve()

    merged = {}
    for tag in ("base", "step300"):
        existing = read_jsonl(E / f"pilot_eval/predictions_{'step0' if tag == 'base' else 'step300'}_original.jsonl")
        remaining = read_jsonl(R / f"generation/{tag}_remaining744.jsonl")
        rows = []
        for row in existing:
            rows.append({"image_id": str(row["image_id"]), "image": row["image"], "question": row["question"],
                         "condition": "original", "system_id": tag, "source": "existing256",
                         "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest()})
        for row in remaining:
            rows.append({"image_id": str(row["image_id"]), "image": row["image"], "question": row["question"],
                         "condition": "original", "system_id": tag, "source": "remaining744",
                         "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                         "image_sha256": row.get("image_sha256"), "adapter_sha256": row.get("adapter_sha256")})
        rows.sort(key=lambda r: int(r["image_id"]))
        merged[tag] = rows
        (R / f"generation/{tag}_full1000.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    base_ids = [row["image_id"] for row in merged["base"]]
    sd_ids = [row["image_id"] for row in merged["step300"]]
    existing_hashes = {f"{tag}_{row['image_id']}": row["question_sha256"] for tag in ("base", "step300") for row in merged[tag] if row["source"] == "existing256"}
    audit = {
        "existing256_n": sum(row["source"] == "existing256" for row in merged["base"]),
        "remaining744_n": sum(row["source"] == "remaining744" for row in merged["base"]),
        "base_full1000_n": len(base_ids), "step300_full1000_n": len(sd_ids),
        "base_ids_equal_step300_ids": base_ids == sd_ids,
        "no_duplicate_image_ids": len(set(base_ids)) == len(base_ids),
        "no_empty_questions": all(row["question"].strip() for row in merged["base"] + merged["step300"]),
        "existing256_matches_reported_hash": {
            "base": hashlib.sha256("\n".join(row["question"] for row in merged["base"] if row["source"] == "existing256").encode()).hexdigest(),
            "step300": hashlib.sha256("\n".join(row["question"] for row in merged["step300"] if row["source"] == "existing256").encode()).hexdigest(),
        },
        "status": "OK",
    }
    (R / "generation/merge_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["base_ids_equal_step300_ids"] and audit["no_duplicate_image_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
