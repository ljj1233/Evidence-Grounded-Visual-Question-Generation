#!/usr/bin/env python3
"""dev1000 confirmation — generate remaining744 fixed outputs for one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

from vvqg.generation.inference import generate_questions, model_prompt  # noqa: E402
from vvqg.training.online_uvfc_relative_grpo import build_frozen_model  # noqa: E402

INSTRUCTION = "Generate a question about this image."


def numeric_id(value: str) -> str:
    return str(int(Path(str(value)).stem.split("_")[-1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--dev-manifest", type=Path, required=True)
    ap.add_argument("--remaining-ids", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-model", default=local_model_path('Qwen3-VL-8B-Instruct'))
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import torch

    dev = {numeric_id(row.get("image_path") or row["image_id"]): row for row in
           (json.loads(line) for line in args.dev_manifest.read_text().splitlines() if line.strip())}
    ids = [line.strip() for line in args.remaining_ids.read_text().splitlines() if line.strip()]
    rows = [{"image_id": i, "image_path": dev[i]["image_path"]} for i in ids]
    model, processor = build_frozen_model(args.base_model, str(args.checkpoint), "sdpa")
    model.to("cuda").eval()
    prompt = model_prompt(processor, INSTRUCTION)
    started = time.perf_counter()
    generated = generate_questions(model, processor, [row["image_path"] for row in rows], prompt, num_samples=1,
                                   temperature=0.0, top_p=1.0, max_new_tokens=32, seed=42, batch_size=args.batch_size,
                                   desc=f"gen-{args.tag}")
    adapter_hash = hashlib.sha256((args.checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
    outputs, failures = [], []
    for row, group in zip(rows, generated):
        question = group[0].strip()
        record = {
            "image_id": row["image_id"], "image": row["image_path"], "question": question,
            "condition": "original", "system_id": args.tag, "checkpoint": str(args.checkpoint),
            "adapter_sha256": adapter_hash, "image_sha256": hashlib.sha256((ROOT / row["image_path"]).read_bytes()).hexdigest(),
            "seed": 42, "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 32},
        }
        if question == "" or question is None:
            failures.append({"image_id": row["image_id"], "error": "empty_question"})
        outputs.append(record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8")
    (args.report_dir / "generation" / f"generation_failures_{args.tag}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8")
    print(json.dumps({"tag": args.tag, "n": len(outputs), "failures": len(failures), "adapter_sha256": adapter_hash,
                      "wall_seconds": time.perf_counter() - started}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
