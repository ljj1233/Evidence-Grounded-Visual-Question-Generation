#!/usr/bin/env python3
"""Phase 5C-9: matched 2-epoch distillation with per-epoch targets.

D1 and D2 share the identical implementation, model, LoRA, optimizer, cosine
schedule, batch size and step count (2 x N / (batch*accum)); the only difference
is the per-epoch target file (single best_question vs grounded positives).
"""

from __future__ import annotations

import argparse
import json
import math
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
sys.path.insert(0, str(ROOT / "src" / "vvqg" / "training"))
from vvqg.paths import local_model_path

from train_sft import ChatDataset, make_collator, prepare_rows  # noqa: E402
from vvqg.training.trainer_base import load_model_and_processor  # noqa: E402

BASE = local_model_path('Qwen3-VL-8B-Instruct')
INIT = ROOT / "outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final"
LORA = {"enabled": True, "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
INSTRUCTION = "Generate a question about this image."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--init", type=Path, default=INIT, help="init adapter (default: frozen Final SFT)")
    args = parser.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    try:
        os.sched_setaffinity(0, set(range(8)))
    except Exception:
        pass
    print(json.dumps({"cpu_affinity": sorted(os.sched_getaffinity(0)),
                      "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                      "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                      "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                      "num_workers": args.num_workers}), flush=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    targets = [json.loads(line) for line in args.targets.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        targets = targets[: args.limit]
    args.outdir.mkdir(parents=True, exist_ok=True)

    model, processor, _ = load_model_and_processor(Path(args.base), LORA, "flash_attention_2", init_adapter=str(args.init))
    model.to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(targets) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    collate = make_collator(processor)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step = 0
    losses = []
    for epoch in range(args.epochs):
        key = "q_epoch1" if epoch == 0 else "q_epoch2"
        wkey = "w_epoch1" if epoch == 0 else "w_epoch2"
        records = [{"image": row["image"], "question": row[key]} for row in targets]
        weights = [float(row.get(wkey, 1.0)) for row in targets]
        prepare_rows(processor, records, INSTRUCTION)
        dataset = ChatDataset(processor, records, int(getattr(processor.image_processor, "merge_size", 2)), 4096, 147456)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            batch = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in batch.items()}
            chunk = weights[index * args.batch_size : (index + 1) * args.batch_size]
            batch_weight = sum(chunk) / len(chunk)
            loss = model(**batch).loss * batch_weight / args.grad_accum
            loss.backward()
            losses.append(float(loss.detach()) * args.grad_accum)
            if (index + 1) % args.grad_accum == 0 or (index + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {sum(losses[-50:])/min(50,len(losses)):.4f}", flush=True)
        del loader, dataset

    checkpoint = args.outdir / "checkpoint-final"
    model.save_pretrained(str(checkpoint))
    processor.save_pretrained(str(checkpoint))
    summary = {
        "targets": str(args.targets), "init_adapter": str(args.init), "epochs": args.epochs,
        "records": len(targets), "total_steps": step, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr, "warmup_steps": warmup, "seed": args.seed,
        "loss_first": losses[0], "loss_last": sum(losses[-50:]) / min(50, len(losses)),
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
        "checkpoint": str(checkpoint),
    }
    (args.outdir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
