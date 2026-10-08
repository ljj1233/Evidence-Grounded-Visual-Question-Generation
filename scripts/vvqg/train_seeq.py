#!/usr/bin/env python3
"""Train one SeeQ QG/QA round from exactly 20,000 formatted records."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path
from vvqg.baselines.common import read_jsonl
from vvqg.baselines.seeq_style.training import SeeQDataset, make_collator
from vvqg.training.trainer_base import PlainSFTTrainer, make_training_arguments

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--formatted", type=Path, required=True); p.add_argument("--base-model", type=Path, default=Path(local_model_path('Qwen3-VL-8B-Instruct')).expanduser()); p.add_argument("--start-adapter", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--max-steps", type=int, default=-1); p.add_argument("--attn-implementation", default="flash_attention_2"); p.add_argument("--dataloader-num-workers", type=int, default=8); p.add_argument("--per-device-batch-size", type=int, default=2); p.add_argument("--gradient-accumulation-steps", type=int, default=1); p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--learning-rate", type=float, default=2e-5); args = p.parse_args()
    records = read_jsonl(args.formatted)
    if len(records) != 20_000: raise AssertionError(f"expected 20,000 formatted records, got {len(records)}")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(str(args.base_model), local_files_only=True, use_fast=False)
    backbone = AutoModelForImageTextToText.from_pretrained(str(args.base_model), dtype=torch.bfloat16, local_files_only=True, attn_implementation=args.attn_implementation)
    actual_attn = getattr(backbone.config, "_attn_implementation", None)
    print(json.dumps({"requested_attn_implementation": args.attn_implementation, "actual_attn_implementation": actual_attn}, sort_keys=True))
    if actual_attn != args.attn_implementation:
        raise RuntimeError(f"attention backend mismatch: requested {args.attn_implementation}, loaded {actual_attn}")
    model = PeftModel.from_pretrained(backbone, str(args.start_adapter), is_trainable=True)
    dataset = SeeQDataset(processor, records, int(getattr(processor.image_processor, "merge_size", 2)))
    steps_per_epoch = (len(records) + args.per_device_batch_size * args.gradient_accumulation_steps - 1) // (args.per_device_batch_size * args.gradient_accumulation_steps)
    warmup_steps = 0 if args.max_steps > 0 else int(steps_per_epoch * 0.1)
    train_args = make_training_arguments(args.output, epochs=1, per_device_batch_size=args.per_device_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, gradient_checkpointing=args.gradient_checkpointing, learning_rate=args.learning_rate, warmup_ratio=0.1, warmup_steps=warmup_steps, max_steps=args.max_steps, dataloader_num_workers=args.dataloader_num_workers)
    trainer = PlainSFTTrainer(model=model, args=train_args, train_dataset=dataset, data_collator=make_collator(processor), processing_class=processor)
    result = trainer.train(); trainer.save_model()
    summary = {"records": len(records), "steps": result.global_step, "output": str(args.output), "start_adapter": str(args.start_adapter)}
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary)); return 0
if __name__ == "__main__": raise SystemExit(main())
