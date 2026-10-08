#!/usr/bin/env python3
"""Generate B2 questions from the frozen caption cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

from vvqg.baselines.contrastive_grounding.inference import B2Generator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path(local_model_path('Qwen3-VL-8B-Instruct')).expanduser())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    captions = {}
    for line in args.captions.open(encoding="utf-8"):
        row = json.loads(line)
        captions[str(row["image_id"])] = row
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    rows = rows[: args.limit] if args.limit else rows
    items = []
    for row in rows:
        items.append((row.get("image_path", row.get("image")), captions[str(row["image_id"])] ["caption"]))
    questions = B2Generator(args.base_model, args.checkpoint, device=args.device, batch_size=args.batch_size).generate(items, num_samples=args.num_samples, temperature=args.temperature, top_p=args.top_p)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row, generated in zip(rows, questions):
            for generation_id, question in enumerate(generated):
                handle.write(json.dumps({"image_id": str(row["image_id"]), "image": row.get("image_path", row.get("image")), "question": question, "generation_id": generation_id, "source": "b2"}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
