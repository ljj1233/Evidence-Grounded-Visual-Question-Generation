#!/usr/bin/env python3
"""Offline DPO pilot — build Flat preference pairs from an existing aligned candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vvqg.training.fixed_group_relative import norm_question  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=ROOT / "outputs/ec_qse_aligned_direct4/dev/candidates_scored.jsonl")
    ap.add_argument("--pilot-n", type=int, default=128)
    args = ap.parse_args()
    R = args.report_dir.resolve()
    (R / "data").mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.pool.read_text().splitlines() if line.strip()]
    by_image = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(row)

    pairs, skipped = [], []
    for image_id, members in by_image.items():
        counts = Counter(norm_question(m["question"]) for m in members)
        valid = [m for m in members if m.get("overall_visual_support") in ("supported", "unsupported")
                 and m.get("flat_state") not in (None, "FAILED") and len(str(m["question"]).strip()) >= 2
                 and counts[norm_question(m["question"])] == 1]
        supported = [m for m in valid if m.get("flat_state") == "NO_CONFIRMED_VIOLATION"]
        unsupported = [m for m in valid if m.get("flat_state") == "CONFIRMED_VIOLATION"]
        if not supported or not unsupported:
            skipped.append({"image_id": image_id, "reason": "no_supported_unsupported_pair",
                            "n_supported": len(supported), "n_unsupported": len(unsupported)})
            continue
        chosen = max(supported, key=lambda m: float(m["UVFC"]))
        rejected = min(unsupported, key=lambda m: float(m["UVFC"]))
        pairs.append({
            "image_id": image_id, "image": chosen["image"],
            "chosen": str(chosen["question"]).strip(), "rejected": str(rejected["question"]).strip(),
            "chosen_support_floor": chosen.get("support_floor"), "rejected_support_floor": rejected.get("support_floor"),
            "chosen_UVFC": chosen.get("UVFC"), "rejected_UVFC": rejected.get("UVFC"),
            "source": str(args.pool.relative_to(ROOT)), "uvfc_used_for": "tie_break_only",
        })
    pairs.sort(key=lambda p: p["image_id"])
    pilot_ids = sorted((p["image_id"] for p in pairs), key=lambda i: hashlib.sha256(i.encode()).hexdigest())[: args.pilot_n]
    pilot_set = set(pilot_ids)
    train_pairs = [p for p in pairs if p["image_id"] not in pilot_set]
    pilot_pairs = [p for p in pairs if p["image_id"] in pilot_set]
    (R / "data/pairs_train.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in train_pairs), encoding="utf-8")
    (R / "data/pairs_pilot.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pilot_pairs), encoding="utf-8")
    (R / "data/skipped_images.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in skipped), encoding="utf-8")
    stats = {
        "source_pool": str(args.pool.relative_to(ROOT)), "instruction": "Generate a question about this image.",
        "pool_images": len(by_image), "pairs_total": len(pairs), "skipped_images": len(skipped),
        "pilot_n": len(pilot_pairs), "train_n": len(train_pairs),
        "pilot_train_image_overlap": len({p["image_id"] for p in pilot_pairs} & {p["image_id"] for p in train_pairs}),
        "chosen_flat_state": "NO_CONFIRMED_VIOLATION", "rejected_flat_state": "CONFIRMED_VIOLATION",
        "uvfc_role": "tie_break_only", "pool_provenance": "existing prompt-aligned Direct-4 (SFT) candidate pool",
        "note": "train2014 aligned pool had only 64 images (<256); dev aligned pool reused with image-level split",
    }
    (R / "data/DATA_STATS.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
