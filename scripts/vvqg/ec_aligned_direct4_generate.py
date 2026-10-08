#!/usr/bin/env python3
"""Evidence-Calibrated QSE: prompt-aligned Direct-4 candidate generation.

Identical sampling protocol to phase5b_direct4_generate.py (frozen Final-SFT,
K=4, T=0.8, top_p=0.9, max_new_tokens=64, seed 42, sdpa, batch 16) but with the
student/evaluation instruction instead of the DIRECT4 teacher instruction:

    Generate a question about this image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
ALIGNED = "Generate a question about this image."


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        os.sched_setaffinity(0, set(range(8)))
    except Exception:
        pass

    args.outdir.mkdir(parents=True, exist_ok=True)
    dev = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    images = [row.get("image_path") or row["image"] for row in dev]
    model, processor = build_frozen_model(BASE, str(SFT_CHECKPOINT), "sdpa")
    model.to("cuda")
    prompt = model_prompt(processor, ALIGNED)
    started = time.perf_counter()
    generated = generate_questions(
        model, processor, images, prompt, num_samples=args.num_samples, temperature=args.temperature,
        top_p=args.top_p, max_new_tokens=args.max_new_tokens, seed=args.seed,
        batch_size=args.batch_size, desc="ec_aligned_direct4",
    )
    errors = []
    candidates_path = args.outdir / "candidates.jsonl"
    with candidates_path.open("w", encoding="utf-8") as handle:
        for row, image, questions in zip(dev, images, generated):
            seen = []
            for question in questions:
                question = question.strip()
                if question and question not in seen:
                    seen.append(question)
            if not seen:
                errors.append({"image_id": str(row["image_id"]), "error": "empty_generation"})
            for generation_id, question in enumerate(seen[: args.num_samples]):
                handle.write(json.dumps({
                    "image_id": str(row["image_id"]), "image": image, "question": question,
                    "generation_id": generation_id, "source": "ec_aligned_direct4",
                }, ensure_ascii=False) + "\n")
    runtime = {
        "instruction": ALIGNED,
        "instruction_sha256": hashlib.sha256(ALIGNED.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "base_model": BASE, "sft_checkpoint": str(SFT_CHECKPOINT),
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "image_count": len(images), "num_samples": args.num_samples,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens, "attn": "sdpa", "seed": args.seed,
        "batch_size": args.batch_size, "candidates": sum(1 for _ in candidates_path.open()),
        "output_sha256": sha256(candidates_path),
        "wall_seconds": time.perf_counter() - started,
    }
    (args.outdir / "generation_manifest.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    (args.outdir / "error_log.json").write_text(json.dumps({"errors": errors}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runtime, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
