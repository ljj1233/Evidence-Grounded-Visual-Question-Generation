#!/usr/bin/env python3
"""Generate B3 questions with detector regions and predicted internal guides."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vvqg.paths import local_model_path

import torch
from vvqg.baselines.implicit_guided.data import inference_regions
from vvqg.baselines.implicit_guided.data import GUIDE_TEXT
from vvqg.baselines.implicit_guided.data import load_detector_cache
from vvqg.baselines.implicit_guided_model import GuidedQwenAdapter, register_guide_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detector-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path(local_model_path('Qwen3-VL-8B-Instruct')).expanduser())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    detector = load_detector_cache(args.detector_cache)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(args.base_model), local_files_only=True, use_fast=False)
    processor.tokenizer.padding_side = "left"
    backbone = AutoModelForImageTextToText.from_pretrained(str(args.base_model), dtype=torch.bfloat16, local_files_only=True, attn_implementation=args.attn_implementation)
    adapter = args.checkpoint / "backbone"
    if (adapter / "adapter_config.json").is_file():
        from peft import PeftModel
        backbone = PeftModel.from_pretrained(backbone, str(adapter))
    guide_ids = register_guide_tokens(processor.tokenizer, backbone)
    object_dim = len(next(iter(detector.values()))["features"][0]) + 4
    model = GuidedQwenAdapter(backbone, object_dim, int(backbone.config.text_config.hidden_size))
    state = torch.load(args.checkpoint / "guided_head.pt", map_location="cpu", weights_only=True)
    model.guidance.load_state_dict(state["guidance"])
    model.guide_projection.load_state_dict(state["guide_projection"])
    model.eval().to(args.device)
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    rows = [row for row in rows if str(row["image_id"]) in detector]
    rows = rows[: args.limit] if args.limit else rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            from PIL import Image

            expanded = [row for row in chunk for _ in range(args.num_samples)]
            images = [Image.open(ROOT / row.get("image_path", row.get("image"))).convert("RGB") for row in expanded]
            messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": GUIDE_TEXT}]}] for _ in expanded]
            prompts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
            batch = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(args.device)
            regions = [inference_regions(row["image_id"], detector) for row in expanded]
            objects = torch.cat([r["objects"] for r in regions]).to(args.device)
            masks = torch.cat([r["object_mask"] for r in regions]).to(args.device)
            ids = model.generate_with_guides(objects, batch["input_ids"], guide_ids, object_mask=masks, **{k: v for k, v in batch.items() if k != "input_ids"}, max_new_tokens=32, do_sample=args.num_samples > 1, temperature=args.temperature, top_p=args.top_p)
            width = batch["input_ids"].shape[1]
            for row, output_index in zip(expanded, range(len(ids))):
                generation_id = output_index % args.num_samples
                handle.write(json.dumps({"image_id": str(row["image_id"]), "image": row.get("image_path", row.get("image")), "question": processor.tokenizer.decode(ids[output_index][width:], skip_special_tokens=True).strip(), "generation_id": generation_id, "source": "b3"}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
