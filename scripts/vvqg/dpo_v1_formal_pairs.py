#!/usr/bin/env python3
"""Formal DPO-v1 preference pairs from an instruction-matched candidate pool.

Inherits the historical DPO-v1 pair rule verbatim
(`scripts/vvqg/dpo_pilot_pairs_from_pool.py`):

  chosen    = Flat ``NO_CONFIRMED_VIOLATION`` candidate with max UVFC
  rejected  = Flat ``CONFIRMED_VIOLATION`` candidate with min UVFC
  eligible  = overall_visual_support in {supported, unsupported}, flat_state not
              FAILED, question length >= 2, unique normalized question within image
  one pair per image; UVFC is tie-break only; identical chosen/rejected dropped.

No support_floor / UVFC median / U>X thresholds; no external labels.
"""

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
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source", default="outputs/offline_dpo_v1_formal/candidates_aligned")
    ap.add_argument("--prompt-hash", default="")
    ap.add_argument("--train-records", type=int, default=0)
    ap.add_argument("--train-images", type=int, default=0)
    args = ap.parse_args()
    R = args.out_dir.resolve()
    R.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.pool.read_text().splitlines() if line.strip()]
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(row)

    pairs, skipped = [], []
    parseable = 0
    for image_id, members in by_image.items():
        counts = Counter(norm_question(m["question"]) for m in members)
        parseable += sum(m.get("flat_state") not in (None, "FAILED") for m in members)
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
        if norm_question(chosen["question"]) == norm_question(rejected["question"]):
            skipped.append({"image_id": image_id, "reason": "identical_questions"})
            continue
        pairs.append({
            "pair_id": hashlib.sha256(f"{image_id}\x00{chosen['question']}\x00{rejected['question']}".encode()).hexdigest()[:16],
            "image_id": image_id, "image": chosen["image"],
            "chosen": str(chosen["question"]).strip(), "rejected": str(rejected["question"]).strip(),
            "chosen_flat_state": chosen["flat_state"], "rejected_flat_state": rejected["flat_state"],
            "chosen_support_floor": chosen.get("support_floor"), "rejected_support_floor": rejected.get("support_floor"),
            "chosen_UVFC": chosen.get("UVFC"), "rejected_UVFC": rejected.get("UVFC"),
            "source": args.source, "prompt_hash": args.prompt_hash, "uvfc_used_for": "tie_break_only",
        })
    pairs.sort(key=lambda p: p["image_id"])
    (R / "PAIR_MANIFEST.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    (R / "pairs_train.jsonl").write_text("".join(json.dumps({"image_id": p["image_id"], "image": p["image"],
        "chosen": p["chosen"], "rejected": p["rejected"]}, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    (R / "skipped_images.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in skipped), encoding="utf-8")
    images_both = len(by_image) - sum(s["reason"] == "no_supported_unsupported_pair" for s in skipped)
    stats = {
        "method_train_records": args.train_records, "method_train_unique_images": args.train_images,
        "raw_candidate_rows": len(rows), "candidate_images": len(by_image), "parseable_candidates": parseable,
        "images_with_supported_and_unsupported": images_both, "final_pairs": len(pairs),
        "skipped_images": len(skipped),
        "chosen_flat_state": "NO_CONFIRMED_VIOLATION", "rejected_flat_state": "CONFIRMED_VIOLATION",
        "uvfc_role": "tie_break_only", "prompt_hash": args.prompt_hash, "source": args.source,
        "pairs_unique_images": len({p["image_id"] for p in pairs}),
    }
    (R / "PAIR_FILTER_STATS.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
