#!/usr/bin/env python3
"""Full Ours image-intervention generation (inference only).

Canonical config: G2 step_final adapter, instruction "Generate a question about
this image.", greedy, max_new_tokens=32, seed 42, sdpa, batch 8. Transformed
images are used only for generation; the evaluator image is always the original.
"""

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
from vvqg.training.online_uvfc_grpo import build_frozen_model  # noqa: E402

BASE = local_model_path('Qwen3-VL-8B-Instruct')
CHECKPOINT = ROOT / "outputs/phase5c_qse_relativeguard/checkpoints/step_final"
INSTRUCTION = "Generate a question about this image."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--original-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--base-model", default=BASE)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    original = {str(json.loads(l)["image_id"]): json.loads(l)["image_path"]
                for l in args.original_manifest.read_text(encoding="utf-8").splitlines() if l.strip()}
    gen_images = [r["image_path"] for r in rows]
    model, processor = build_frozen_model(args.base_model, str(args.checkpoint), "sdpa")
    model.to("cuda").eval()
    prompt = model_prompt(processor, INSTRUCTION)
    started = time.perf_counter()
    generated = generate_questions(
        model, processor, gen_images, prompt, num_samples=1, temperature=0.0, top_p=1.0,
        max_new_tokens=args.max_new_tokens, seed=args.seed, batch_size=args.batch_size, desc=args.condition,
    )
    wall = time.perf_counter() - started
    prompt_hash = hashlib.sha256(INSTRUCTION.encode()).hexdigest()
    with args.out.open("w", encoding="utf-8") as handle:
        for row, questions in zip(rows, generated):
            iid = str(row["image_id"])
            question = questions[0].strip()
            handle.write(json.dumps({
                "image_id": iid, "image": original[iid], "generation_image": row["image_path"],
                "condition": args.condition, "question": question, "generation_id": 0,
                "checkpoint": str(args.checkpoint), "seed": args.seed, "prompt_hash": prompt_hash,
                "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0,
                             "max_new_tokens": args.max_new_tokens, "attention": "sdpa", "batch_size": args.batch_size},
                "empty": question == "", "source": "full_ours",
            }, ensure_ascii=False) + "\n")
    runtime = {"condition": args.condition, "n": len(rows), "empty": sum(q[0].strip() == "" for q in generated),
               "wall_seconds": wall, "checkpoint": str(args.checkpoint), "checkpoint_adapter_sha256":
               hashlib.sha256((args.checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest(),
               "prompt_hash": prompt_hash, "instruction": INSTRUCTION}
    (args.out.parent / f"runtime_{args.condition}.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
