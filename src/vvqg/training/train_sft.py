#!/usr/bin/env python3
"""Image-conditioned SFT entry point (Stage 1) for the reward-driven VQG pipeline.

Frozen task: Image -> Question. Base model: Qwen3-VL-8B-Instruct
(`~/models/Qwen3-VL-8B-Instruct`). LoRA adaptation, stock causal-LM loss from
`transformers` -- no custom loss and no model-architecture modification.

Dataset: JSONL `{image, question}` from `src/vvqg/data/sft/train.jsonl`, validated
against `src/vvqg/data/sft/schema.json`. The image is fed to the model through the
Qwen3-VL processor (`pixel_values`, `image_grid_thw`, `mm_token_type_ids`); labels
are masked to the assistant answer only.

Default mode is `--dry-run` (no model load). `--train` runs full training; a
separate smoke entry lives at `scripts/vvqg/smoke_sft.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vvqg.paths import local_model_path

from vvqg.training.trainer_base import REQUIRED_LORA_TARGETS

DATA_ROOT = SRC_ROOT / "vvqg" / "data"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "new_vqg" / "sft.yaml"
DEFAULT_DATA = DATA_ROOT / "sft"
DEFAULT_OUTPUT = REPO_ROOT / "checkpoint" / "sft"
DEFAULT_BASE_MODEL = Path(local_model_path('Qwen3-VL-8B-Instruct')).expanduser()
DEFAULT_INSTRUCTION = "Generate a question about this image."
KEEP_KEYS = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _resolve_path(value: str, default: Path) -> Path:
    if not value:
        return default
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def _resolve_train_file(value: str, default: Path) -> Path:
    path = _resolve_path(value, default)
    if path.is_dir():
        path = path / "train.jsonl"
    return path


def _load_validator() -> Any:
    path = REPO_ROOT / "scripts" / "new_vqg" / "check_dataset.py"
    spec = importlib.util.spec_from_file_location("new_vqg_check_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in tqdm(handle, desc=f"load {path.name}", unit="rec"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit is not None:
        records = records[:limit]
    return records


def resolve_image(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def build_messages(instruction: str, question: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
        {"role": "assistant", "content": question},
    ]


def prepare_rows(processor: Any, records: list[dict[str, Any]], instruction: str) -> None:
    for row in tqdm(records, desc="tokenize", unit="rec"):
        messages = build_messages(instruction, row["question"])
        row["_full_text"] = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt = processor.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )
        row["_prompt_token_ids"] = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]


def mask_labels(input_ids: Any, attention_mask: Any, prompt_lens: list[int], ignore_index: int = -100) -> Any:
    labels = input_ids.clone()
    labels[attention_mask == 0] = ignore_index
    for index, prompt_len in enumerate(prompt_lens):
        labels[index, : int(prompt_len)] = ignore_index
    return labels


def make_collator(processor: Any) -> Any:
    import torch

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        padded = processor.tokenizer.pad(
            {
                "input_ids": [item["input_ids"] for item in batch],
                "attention_mask": [item["attention_mask"] for item in batch],
            },
            padding=True,
            return_tensors="pt",
        )
        labels = mask_labels(
            padded["input_ids"], padded["attention_mask"], [item["prompt_len"] for item in batch]
        )
        collated = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": labels,
            "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
            "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch], dim=0),
        }
        if any(item["has_mm_token_type_ids"] for item in batch):
            width = padded["input_ids"].shape[1]
            mm = torch.zeros(len(batch), width, dtype=batch[0]["mm_token_type_ids"].dtype)
            for index, item in enumerate(batch):
                tokens = item["mm_token_type_ids"]
                mm[index, : tokens.shape[0]] = tokens
            collated["mm_token_type_ids"] = mm
        return collated

    return collate


class ChatDataset:
    def __init__(
        self,
        processor: Any,
        records: list[dict[str, Any]],
        merge_size: int,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        self.processor = processor
        self.records = records
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        row = self.records[index]
        image_kwargs: dict[str, int] = {}
        if self.min_pixels is not None:
            image_kwargs["min_pixels"] = int(self.min_pixels)
        if self.max_pixels is not None:
            image_kwargs["max_pixels"] = int(self.max_pixels)
        with Image.open(resolve_image(row["image"])) as source:
            image = source.convert("RGB")
            encoded = self.processor(
                text=row["_full_text"], images=[image], return_tensors="pt", **image_kwargs
            )
        item: dict[str, Any] = {}
        for key, value in encoded.items():
            if key not in KEEP_KEYS:
                continue
            item[key] = value if key == "image_grid_thw" else value.squeeze(0)
        item["has_mm_token_type_ids"] = "mm_token_type_ids" in item
        image_expanded = int(encoded["image_grid_thw"].prod().item()) // (self.merge_size * self.merge_size)
        item["prompt_len"] = len(row["_prompt_token_ids"]) + image_expanded - 1
        return item


def load_model_and_processor(model_path: Path, attn_implementation: str, lora_cfg: dict[str, Any], init_adapter: Path | None = None) -> tuple[Any, Any, Any]:
    from vvqg.training.trainer_base import load_model_and_processor as _load
    return _load(model_path, lora_cfg, attn_implementation, init_adapter=str(init_adapter) if init_adapter else None)


def lora_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    lora = cfg.get("lora") or {}
    if not lora.get("enabled", False):
        return {"enabled": False}
    return {
        "enabled": True,
        "r": int(lora.get("r", 16)),
        "lora_alpha": float(lora.get("lora_alpha", 32)),
        "lora_dropout": float(lora.get("lora_dropout", 0.05)),
        "target_modules": list(lora.get("target_modules", REQUIRED_LORA_TARGETS)),
    }


def dry_run(cfg: dict[str, Any], data_dir: Path, output_dir: Path, train_file: Path) -> int:
    base_model = _resolve_path(cfg.get("base_model", ""), DEFAULT_BASE_MODEL)
    schema_path = data_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.is_file() else {}
    validator = _load_validator()
    records = load_records(train_file) if train_file.is_file() else []
    errors = [
        f"{train_file.name}:{line}: {message}"
        for line, message in (validator.validate_file(train_file, schema) if train_file.is_file() else (0, []))[1]
    ]

    print("== train_sft dry-run ==")
    print(f"base_model   : {base_model} ({'found' if base_model.exists() else 'MISSING'})")
    print(f"train_file   : {train_file} ({len(records)} record(s))")
    print(f"output_dir   : {output_dir}")
    print(f"lora         : {lora_plan(cfg)}")
    print(f"bf16         : {cfg.get('precision', {}).get('bf16', False)}")
    if errors:
        for message in errors[:20]:
            print(f"schema-error : {message}")
        print(f"schema       : FAIL ({len(errors)} error(s))")
        return 1
    print("schema       : OK")
    print("DRY RUN — no model loaded, no training performed.")
    return 0


def require_rft_sft_checkpoint(cfg: dict[str, Any], init_adapter: Path | None) -> None:
    """RFT must continue from Final SFT; refuse a silent fresh random LoRA."""
    if cfg.get("stage") == "rft" and init_adapter is None:
        raise SystemExit(
            "RFT training requires an explicit sft_checkpoint (Final SFT continuation). "
            "Refusing to create a fresh random LoRA. Set `sft_checkpoint`, or use `stage: sft` "
            "for fresh-adapter training."
        )


def train(
    cfg: dict[str, Any],
    train_file: Path,
    output_dir: Path,
    max_steps: int | None = None,
    dev_file: Path | None = None,
) -> int:
    import torch
    from transformers import set_seed
    from vvqg.training.trainer_base import PlainSFTTrainer, make_training_arguments

    base_model = _resolve_path(cfg.get("base_model", ""), DEFAULT_BASE_MODEL)
    if not base_model.exists():
        raise FileNotFoundError(f"base model not found: {base_model}")
    if not train_file.is_file():
        raise FileNotFoundError(f"training data not found: {train_file}")

    training = cfg.get("training", {})
    instruction = cfg.get("instruction", DEFAULT_INSTRUCTION)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    records = load_records(train_file)
    init_adapter = _resolve_path(cfg.get("sft_checkpoint", ""), Path("")) if cfg.get("sft_checkpoint") else None
    require_rft_sft_checkpoint(cfg, init_adapter)
    model, processor, targeted = load_model_and_processor(
        base_model, training.get("attn_implementation", "sdpa"), cfg.get("lora") or {}, init_adapter
    )
    prepare_rows(processor, records, instruction)
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    image_cfg = cfg.get("image") or {}
    dataset = ChatDataset(
        processor, records, merge_size, image_cfg.get("min_pixels"), image_cfg.get("max_pixels")
    )
    collate = make_collator(processor)

    batch_size = int(training.get("per_device_train_batch_size", 2))
    grad_accum = int(training.get("gradient_accumulation_steps", 8))
    epochs = float(training.get("num_train_epochs", 2))
    if max_steps is not None:
        total_steps = max_steps
    else:
        total_steps = math.ceil(math.ceil(len(dataset) / (batch_size * grad_accum)) * epochs)
    warmup_steps = int(total_steps * float(training.get("warmup_ratio", 0.03)))

    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = make_training_arguments(
        output_dir,
        epochs=epochs,
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(training.get("learning_rate", 2.0e-4)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        bf16=bool(cfg.get("precision", {}).get("bf16", True)),
        logging_steps=int(training.get("logging_steps", 10)),
        save_steps=int(training.get("save_steps", 500)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        seed=seed,
        max_steps=max_steps if max_steps is not None else -1,
    )
    arguments.warmup_steps = warmup_steps
    arguments.save_total_limit = int(training.get("save_total_limit", 3))
    arguments.dataloader_num_workers = int(training.get("dataloader_num_workers", 8))
    arguments.report_to = "none"
    arguments.disable_tqdm = False
    if max_steps is not None:
        arguments.save_strategy = "no"

    trainer = PlainSFTTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=collate,
        processing_class=processor,
    )
    trainer.train()
    history = trainer.state.log_history
    (output_dir / "trainer_log_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )

    summary: dict[str, Any] = {
        "base_model": str(base_model),
        "init_adapter": str(init_adapter) if init_adapter else None,
        "train_file": str(train_file),
        "train_records": len(records),
        "lora": lora_plan(cfg),
        "lora_matched_modules": len(list(targeted)),
        "max_steps": max_steps,
        "total_steps": int(trainer.state.global_step),
        "num_train_epochs": epochs,
        "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)
        if torch.cuda.is_available()
        else None,
        "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3)
        if torch.cuda.is_available()
        else None,
    }
    train_entries = [entry for entry in history if "loss" in entry]
    if train_entries:
        summary["train_loss_first"] = train_entries[0]["loss"]
        summary["train_loss_last"] = train_entries[-1]["loss"]

    if max_steps is None:
        checkpoint = output_dir / "checkpoint-final"
        trainer.save_model(str(checkpoint))
        processor.save_pretrained(str(checkpoint))
        summary["checkpoint"] = str(checkpoint)
        print(f"saved LoRA checkpoint -> {checkpoint}")

        if dev_file is not None and dev_file.is_file():
            dev_records = load_records(dev_file)
            prepare_rows(processor, dev_records, instruction)
            dev_dataset = ChatDataset(
                processor, dev_records, merge_size, image_cfg.get("min_pixels"), image_cfg.get("max_pixels")
            )
            metrics = trainer.evaluate(eval_dataset=dev_dataset)
            summary["dev_file"] = str(dev_file)
            summary["dev_records"] = len(dev_records)
            summary["dev_loss"] = metrics.get("eval_loss")
            (output_dir / "dev_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-conditioned SFT entry point (dry-run by default).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate only (default)")
    mode.add_argument("--train", action="store_true", help="run real training")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _load_yaml(args.config)
    data_dir = args.data_dir or _resolve_path(cfg.get("train_data", ""), DEFAULT_DATA)
    output_dir = args.output_dir or _resolve_path(cfg.get("output_dir", ""), DEFAULT_OUTPUT)
    train_file = _resolve_train_file(cfg.get("train_data", ""), DEFAULT_DATA)

    if args.train:
        return train(cfg, train_file, output_dir, args.max_steps, train_file.parent / "dev.jsonl")
    return dry_run(cfg, data_dir, output_dir, train_file)


if __name__ == "__main__":
    raise SystemExit(main())
