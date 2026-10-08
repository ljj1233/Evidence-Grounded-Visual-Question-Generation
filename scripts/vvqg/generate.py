#!/usr/bin/env python3
"""Canonical question generation from the base model or final SFT checkpoint.

One entry point for greedy evaluation and N-sample candidate pools.

Examples
--------
  # greedy, one question per image
  .venv/bin/python scripts/vvqg/generate.py --data <manifest.jsonl> \
      --output <preds.jsonl> --schema greedy

  # 4 sampled questions per image
  .venv/bin/python scripts/vvqg/generate.py --data <manifest.jsonl> \
      --output <cands.jsonl> --schema sampled --num-samples 4

Uses the PIL image-processor backend (fast) and cross-image batching. Incremental
JSONL writing + resume. Does not modify the checkpoint or call any API.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from vvqg.paths import local_model_path

from vvqg.generation import GenerationConfig, VQGGenerator  # noqa: E402

DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "checkpoints" / "qwen3vl8b_lora_r16_vqav2_50k" / "checkpoint-final"
DEFAULT_BASE = str(Path(local_model_path('Qwen3-VL-8B-Instruct')).expanduser())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_manifest(path: Path) -> dict[str, str]:
    images: dict[str, str] = {}
    for row in read_jsonl(path):
        images[str(row["image_id"])] = row.get("image_path") or row.get("image")
    return images


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate questions with the base model or final SFT adapter.")
    parser.add_argument("--data", type=Path, required=True, help="image manifest JSONL (image_id, image_path)")
    parser.add_argument("--output", type=Path, required=True)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    model.add_argument("--zero-shot", action="store_true", help="use the base model without a LoRA adapter")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--schema", choices=["greedy", "sampled", "1x"], default="greedy")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backend", choices=["pil", "torchvision"], default="pil")
    parser.add_argument("--source", default=None)
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=147456)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def pending_batches(image_ids: list[str], done: set[tuple[str, int]], batch_size: int, num_samples: int):
    for start in range(0, len(image_ids), batch_size):
        chunk = image_ids[start:start + batch_size]
        if any((iid, g) not in done for iid in chunk for g in range(num_samples)):
            yield chunk


def record(schema: str, image_id: str, image: str, generation_id: int, question: str, args: argparse.Namespace) -> dict:
    if schema == "greedy":
        return {"image_id": image_id, "image": image, "question": question, "source": args.source}
    if schema == "1x":
        return {
            "image_id": image_id,
            "image": image,
            "question": question,
            "generation_id": 0,
            "decoding": f"sample_t{args.temperature}_p{args.top_p}",
        }
    return {
        "image_id": image_id,
        "image": image,
        "question": question,
        "generation_id": generation_id,
        "source": args.source,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
    }


def main(argv: Optional[list[str]] = None) -> int:
    import torch
    from tqdm import tqdm

    args = parse_args(argv)
    args.source = args.source or ("zero_shot" if args.zero_shot else "final_sft")
    if args.schema == "greedy":
        args.num_samples, args.temperature = 1, 0.0
    images = image_manifest(args.data)
    image_ids = sorted(images)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, int]] = set()
    if args.output.is_file():
        for row in read_jsonl(args.output):
            done.add((str(row["image_id"]), int(row.get("generation_id", 0))))

    pending = [iid for iid in image_ids if any((iid, g) not in done for g in range(args.num_samples))]
    pending_ids = set(pending)
    print(f"images total={len(image_ids)} pending={len(pending)} schema={args.schema} num_samples={args.num_samples}", flush=True)
    if not pending:
        return 0

    config = GenerationConfig(
        base_model=args.base_model,
        checkpoint=None if args.zero_shot else str(args.checkpoint),
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        backend=args.backend,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    load_start = time.time()
    generator = VQGGenerator(config)
    load_seconds = time.time() - load_start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    written = 0
    processed = 0
    gen_start = time.time()
    with args.output.open("a", encoding="utf-8") as handle:
        bar = None if args.no_progress else tqdm(total=len(pending), desc="generate", unit="img")
        for chunk in pending_batches(image_ids, done, args.batch_size, config.num_samples):
            outputs = generator.generate_many([(iid, images[iid]) for iid in chunk])
            processed += len(chunk)
            for image_id, questions in outputs:
                for generation_id, question in enumerate(questions):
                    if (image_id, generation_id) in done:
                        continue
                    handle.write(json.dumps(record(args.schema, image_id, images[image_id], generation_id, question, args), ensure_ascii=False) + "\n")
                    written += 1
            handle.flush()
            if bar is not None:
                bar.update(sum(iid in pending_ids for iid in chunk))
        if bar is not None:
            bar.close()
    gen_seconds = time.time() - gen_start

    stats = {
        "checkpoint": config.checkpoint,
        "base_model": args.base_model,
        "schema": args.schema,
        "num_samples": config.num_samples,
        "batch_size": args.batch_size,
        "backend": args.backend,
        "image_resolution": {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        "seed": args.seed,
        "images_processed": processed,
        "images_pending": len(pending),
        "oom_splits": generator.oom_splits,
        "records_written": written,
        "model_load_seconds": round(load_seconds, 2),
        "generation_seconds": round(gen_seconds, 2),
        "images_per_sec": round(processed / gen_seconds, 3) if gen_seconds else None,
        "questions_per_sec": round(written / gen_seconds, 3) if gen_seconds else None,
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0.0,
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
