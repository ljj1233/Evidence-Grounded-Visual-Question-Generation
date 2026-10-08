#!/usr/bin/env python3
"""Train the implicit-guided Qwen baseline from frozen detector features."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path
from vvqg.baselines.common import read_jsonl
from vvqg.baselines.implicit_guided import GuidedQwenDataset, make_guided_collator
from vvqg.baselines.implicit_guided.data import load_detector_cache
from vvqg.baselines.implicit_guided.trainer import B3Trainer
from vvqg.baselines.implicit_guided_model import GuidedQwenAdapter, register_guide_tokens
from vvqg.training.trainer_base import load_baseline_model_and_processor, make_training_arguments

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, default=ROOT / "outputs/same_backbone_baselines/b3_implicit/cache/train_targets.jsonl")
    p.add_argument("--detector-cache", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path(local_model_path('Qwen3-VL-8B-Instruct')).expanduser())
    p.add_argument("--output", type=Path, default=ROOT / "outputs/same_backbone_baselines/b3_implicit/checkpoints")
    p.add_argument("--limit", type=int); p.add_argument("--max-steps", type=int, default=-1); p.add_argument("--attn-implementation", default="flash_attention_2"); p.add_argument("--processor-backend", choices=("pil", "torchvision"), default="pil"); p.add_argument("--dataloader-num-workers", type=int, default=8); p.add_argument("--per-device-batch-size", type=int, default=2); p.add_argument("--gradient-accumulation-steps", type=int, default=8); p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--min-pixels", type=int, default=4096); p.add_argument("--max-pixels", type=int, default=147456)
    args = p.parse_args()
    records = read_jsonl(args.records); records = records[:args.limit] if args.limit else records
    detector = load_detector_cache(args.detector_cache)
    records = [row for row in records if str(row["image_id"]) in detector]
    backbone, processor, _ = load_baseline_model_and_processor(args.model, {"enabled": True, "r": 16, "lora_alpha": 32, "lora_dropout": 0.05}, args.attn_implementation, args.processor_backend)
    guide_ids = register_guide_tokens(processor.tokenizer, backbone)
    hidden = int(backbone.config.text_config.hidden_size)
    object_dim = len(detector[str(records[0]["image_id"])] ["features"][0]) + 4
    model = GuidedQwenAdapter(backbone, object_dim, hidden)
    dataset = GuidedQwenDataset(processor, records, detector, int(getattr(processor.image_processor, "merge_size", 2)), min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    args_train = make_training_arguments(args.output, epochs=2, per_device_batch_size=args.per_device_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, gradient_checkpointing=args.gradient_checkpointing, max_steps=args.max_steps, dataloader_num_workers=args.dataloader_num_workers)
    trainer = B3Trainer(model=model, args=args_train, train_dataset=dataset, data_collator=make_guided_collator(processor, guide_ids), processing_class=processor)
    result = trainer.train(); trainer.save_model()
    print(json.dumps({"records": len(records), "steps": result.global_step, "output": str(args.output)}))
    return 0

if __name__ == "__main__": raise SystemExit(main())
