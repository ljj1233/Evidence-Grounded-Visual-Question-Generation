#!/usr/bin/env python3
"""Score greedy generations with the frozen requirement-level Flat verifier."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path
from vvqg.rewards.support.structured_flat import StructuredFlatVerifier, support_summary  # noqa: E402

FLAT_MODEL = local_model_path('Qwen2.5-VL-7B-Instruct')


def read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--flat-model", default=FLAT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    import torch

    try:
        os.sched_setaffinity(0, set(range(8)))
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read(args.predictions)
    done = {str(r["image_id"]) for r in read(args.out)} if args.out.is_file() else set()
    pending = [r for r in rows if str(r["image_id"]) not in done]
    if pending:
        flat = StructuredFlatVerifier(args.flat_model, device, batch_size=args.batch_size)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), args.batch_size):
                chunk = pending[start : start + args.batch_size]
                for row, item in zip(chunk, flat.verify_structured(chunk)):
                    handle.write(json.dumps({"image_id": str(row["image_id"]), "question": row["question"],
                                             "overall_visual_support": item.get("overall_visual_support"),
                                             **support_summary(item)}, ensure_ascii=False) + "\n")
                handle.flush()
        del flat
        torch.cuda.empty_cache()
    print(json.dumps({"scored": len(read(args.out)), "total": len(rows), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
