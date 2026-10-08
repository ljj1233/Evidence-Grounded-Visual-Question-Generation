#!/usr/bin/env python3
"""Phase 5B-A: Direct-4 candidate pool (matched control for Explore-4).

Identical sampling protocol to Phase 5A Explore-4 (SFT checkpoint, N=4, T=0.8,
seed 42, max_new_tokens 64, sdpa) with the plain direct VQG prompt instead of the
exploration prompt.
"""

from __future__ import annotations

import argparse
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
from vvqg.training.online_uvfc_grpo import build_frozen_model  # noqa: E402

BASE = local_model_path('Qwen3-VL-8B-Instruct')
SFT_CHECKPOINT = ROOT / "outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final"
DEV = ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl"

DIRECT4 = (
    "Generate one natural question about this image.\n"
    "The question must be answerable from the image.\n"
    "Output only the question."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--manifest", type=Path, default=DEV)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--sft-checkpoint", type=Path, default=SFT_CHECKPOINT)
    args = parser.parse_args()

    import torch

    try:
        os.sched_setaffinity(0, set(range(8)))
    except Exception:
        pass
    args.outdir.mkdir(parents=True, exist_ok=True)
    dev = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        dev = dev[: args.limit]
    images = [row.get("image_path") or row["image"] for row in dev]
    model, processor = build_frozen_model(args.base, str(args.sft_checkpoint), "sdpa")
    model.to("cuda")
    prompt = model_prompt(processor, DIRECT4)
    started = time.perf_counter()
    generated = generate_questions(
        model, processor, images, prompt, num_samples=args.num_samples, temperature=args.temperature,
        max_new_tokens=args.max_new_tokens, seed=42, batch_size=args.batch_size, desc="direct4",
    )
    with (args.outdir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row, image, questions in zip(dev, images, generated):
            seen = []
            for question in questions:
                question = question.strip()
                if question and question not in seen:
                    seen.append(question)
            for generation_id, question in enumerate(seen[: args.num_samples]):
                handle.write(json.dumps({
                    "image_id": str(row["image_id"]), "image": image, "question": question,
                    "generation_id": generation_id, "source": "direct4",
                }, ensure_ascii=False) + "\n")
    with (args.outdir / "greedy1000.jsonl").open("w", encoding="utf-8") as handle:
        for row, image, questions in zip(dev, images, generated):
            handle.write(json.dumps({
                "image_id": str(row["image_id"]), "image": image,
                "question": questions[0].strip(), "source": "direct4_candidate0",
            }, ensure_ascii=False) + "\n")
    (args.outdir / "generation_runtime.json").write_text(json.dumps({
        "instruction": DIRECT4, "num_samples": args.num_samples, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens, "attn": "sdpa", "seed": 42,
        "wall_seconds": time.perf_counter() - started,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wall_seconds": time.perf_counter() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
