#!/usr/bin/env python3
"""Greedy development1000 generation for selected Phase-4B/5A checkpoints.

Mirrors the trainer's greedy path (frozen base + LoRA adapter, sdpa, PIL backend,
greedy decoding, batch 8) so every checkpoint is evaluated with one identical
image -> question protocol.  Writes ``<out>/<tag>/greedy1000.jsonl``.
"""

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

from vvqg.generation.inference import generate_questions, model_prompt  # noqa: E402
from vvqg.training.online_uvfc_grpo import build_frozen_model  # noqa: E402

BASE_MODEL = local_model_path('Qwen3-VL-8B-Instruct')
INSTRUCTION = "Generate a question about this image."
DEV_MANIFEST = ROOT / "outputs/same_backbone_baselines/manifests/development1000_images.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--dev-manifest", type=Path, default=DEV_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch

    try:
        os.sched_setaffinity(0, set(range(8)))
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = [json.loads(line) for line in args.dev_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    images = [row.get("image_path") or row["image"] for row in dev]

    for tag in args.tags:
        target = args.out / tag / "greedy1000.jsonl"
        if target.is_file():
            print(f"skip existing {target}", flush=True)
            continue
        checkpoint = args.run_dir / "checkpoints" / tag
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        model, processor = build_frozen_model(BASE_MODEL, str(checkpoint), "sdpa")
        model.to(device)
        prompt = model_prompt(processor, INSTRUCTION)
        greedy = generate_questions(
            model, processor, images, prompt, num_samples=1, temperature=0.0,
            max_new_tokens=32, seed=42, batch_size=args.batch_size, desc=f"{tag}",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row, image, questions in zip(dev, images, greedy):
                handle.write(json.dumps({
                    "image_id": str(row["image_id"]), "image": image,
                    "question": questions[0], "source": tag,
                }, ensure_ascii=False) + "\n")
        print(f"wrote {target}", flush=True)
        del model, processor
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
