#!/usr/bin/env python3
"""Smoke test for image-conditioned SFT (Qwen3-VL-8B-Instruct + LoRA).

Verifies, on a small subset (default 1000 samples), that:
1. the image processor is loaded;
2. forward inputs contain image tensors (`pixel_values`, `image_grid_thw`,
   `input_ids`), and the loss changes when the image is zeroed (the model is not
   ignoring the image);
3. the training loss can decrease;
4. generation works: image-only input -> a generated question.

This is a smoke test, not the full 50k run. No reward/GRPO/evaluation changes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vvqg.training.train_sft import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_CONFIG,
    DEFAULT_DATA,
    DEFAULT_INSTRUCTION,
    ChatDataset,
    _load_yaml,
    _resolve_path,
    _resolve_train_file,
    load_model_and_processor,
    load_records,
    make_collator,
    prepare_rows,
)


def generate(model, processor, image_path: Path, instruction: str, max_new_tokens: int = 32) -> str:
    import torch
    from PIL import Image

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-conditioned SFT smoke test.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--attn-implementation", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    args = parse_args(argv)
    cfg = _load_yaml(args.config)
    smoke_cfg = cfg.get("smoke", {})
    training = cfg.get("training", {})
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    samples = args.samples if args.samples is not None else int(smoke_cfg.get("num_samples", 1000))
    max_steps = args.max_steps if args.max_steps is not None else int(smoke_cfg.get("max_steps", 20))
    logging_steps = int(smoke_cfg.get("logging_steps", 1))
    output_dir = args.output_dir or _resolve_path(
        smoke_cfg.get("output_dir", ""), REPO_ROOT / "checkpoint" / "sft_smoke"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = _resolve_path(cfg.get("base_model", ""), DEFAULT_BASE_MODEL)
    train_file = _resolve_train_file(cfg.get("train_data", ""), DEFAULT_DATA)
    instruction = cfg.get("instruction", DEFAULT_INSTRUCTION)
    attn = args.attn_implementation or training.get("attn_implementation", "sdpa")

    if not base_model.exists():
        raise FileNotFoundError(f"base model not found: {base_model}")

    all_records = load_records(train_file)
    records = random.Random(seed).sample(all_records, min(samples, len(all_records)))
    set_seed(seed)

    model, processor, targeted = load_model_and_processor(base_model, attn, cfg.get("lora") or {})
    if torch.cuda.is_available():
        model.to("cuda")
    prepare_rows(processor, records, instruction)
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    image_cfg = cfg.get("image") or {}
    dataset = ChatDataset(
        processor, records, merge_size, image_cfg.get("min_pixels"), image_cfg.get("max_pixels")
    )
    collate = make_collator(processor)

    image_processor_loaded = hasattr(processor, "image_processor") and processor.image_processor is not None

    batch = collate([dataset[0], dataset[1]])
    batch_keys = sorted(batch)
    pixel_values = batch["pixel_values"]
    image_grid_thw = batch["image_grid_thw"]
    image_tensors = {
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values_numel": int(pixel_values.numel()),
        "pixel_values_nonzero": bool(torch.count_nonzero(pixel_values).item() > 0),
        "image_grid_thw_shape": list(image_grid_thw.shape),
        "image_grid_thw": image_grid_thw.tolist(),
        "has_mm_token_type_ids": "mm_token_type_ids" in batch,
        "labels_masked_tokens": int((batch["labels"] == -100).sum().item()),
        "input_ids_shape": list(batch["input_ids"].shape),
    }

    device = next(model.parameters()).device
    batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        loss_with_image = float(model(**batch).loss.item())
        zeroed = {key: value.clone() for key, value in batch.items()}
        zeroed["pixel_values"] = torch.zeros_like(zeroed["pixel_values"])
        loss_zeroed_image = float(model(**zeroed).loss.item())
    image_dependence = {
        "loss_with_image": loss_with_image,
        "loss_zeroed_image": loss_zeroed_image,
        "abs_delta": abs(loss_with_image - loss_zeroed_image),
        "model_consumes_image": abs(loss_with_image - loss_zeroed_image) > 1e-4,
    }

    model.train()
    warmup_steps = int(max_steps * float(training.get("warmup_ratio", 0.03)))
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        learning_rate=float(training.get("learning_rate", 2.0e-4)),
        warmup_steps=warmup_steps,
        bf16=bool(cfg.get("precision", {}).get("bf16", True)),
        logging_steps=logging_steps,
        max_steps=max_steps,
        save_strategy="no",
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=collate,
        processing_class=processor,
    )
    trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    loss_behavior = {
        "steps_logged": len(losses),
        "first": losses[0] if losses else None,
        "last": losses[-1] if losses else None,
        "min": min(losses) if losses else None,
        "decreased": bool(losses and losses[-1] < losses[0]),
        "history": losses,
    }

    model.eval()
    generations = []
    for row in records[:2]:
        generations.append(
            {"image": row["image"], "question": generate(model, processor, REPO_ROOT / row["image"], instruction)}
        )
    generation_differs = len(generations) == 2 and generations[0]["question"] != generations[1]["question"]

    gpu = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
        "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3) if torch.cuda.is_available() else None,
    }

    status = "PASS" if (
        image_processor_loaded
        and image_tensors["pixel_values_numel"] > 0
        and image_tensors["pixel_values_nonzero"]
        and image_dependence["model_consumes_image"]
        and loss_behavior["decreased"]
        and all(item["question"] for item in generations)
    ) else "FAIL"

    report = {
        "status": status,
        "model_path": str(base_model),
        "attn_implementation": attn,
        "processor_class": type(processor).__name__,
        "image_processor_loaded": image_processor_loaded,
        "lora_matched_modules": len(list(targeted)),
        "samples": len(records),
        "max_steps": max_steps,
        "gpu": gpu,
        "batch_keys": batch_keys,
        "image_tensors": image_tensors,
        "image_dependence": image_dependence,
        "loss_behavior": loss_behavior,
        "generations": generations,
        "generation_differs_across_images": generation_differs,
    }
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("== smoke_sft ==")
    print(f"status            : {status}")
    print(f"model             : {base_model}")
    print(f"processor         : {report['processor_class']} (image_processor_loaded={image_processor_loaded})")
    print(f"lora modules      : {report['lora_matched_modules']}")
    print(f"samples/steps     : {len(records)} / {max_steps}")
    print(f"batch keys        : {batch_keys}")
    print(f"pixel_values      : shape={image_tensors['pixel_values_shape']} nonzero={image_tensors['pixel_values_nonzero']}")
    print(f"image_grid_thw    : shape={image_tensors['image_grid_thw_shape']} value={image_tensors['image_grid_thw']}")
    print(f"image dependence  : with={loss_with_image:.4f} zeroed={loss_zeroed_image:.4f} delta={image_dependence['abs_delta']:.4f} consumes={image_dependence['model_consumes_image']}")
    print(f"loss              : first={loss_behavior['first']} last={loss_behavior['last']} decreased={loss_behavior['decreased']}")
    print(f"gpu               : {gpu}")
    for item in generations:
        print(f"gen               : [{Path(item['image']).name}] -> {item['question']!r}")
    print(f"report            : {output_dir / 'smoke_report.json'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
