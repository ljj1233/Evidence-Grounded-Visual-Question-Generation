"""Shared model loading and optimization hooks for baseline trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import torch

REQUIRED_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def target_token_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return the exact assistant-target mask used by Plain SFT."""
    return labels.ne(-100)


def load_model_and_processor(model_path: Path, lora_cfg: dict[str, Any], attn_implementation: str = "sdpa", processor_backend: str = "torchvision", init_adapter: str | None = None):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor_kwargs = {"local_files_only": True}
    if processor_backend == "pil":
        processor_kwargs["use_fast"] = False
    processor = AutoProcessor.from_pretrained(str(model_path), **processor_kwargs)
    kwargs = {"dtype": torch.bfloat16, "local_files_only": True, "attn_implementation": attn_implementation}
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **kwargs)
    except (ImportError, ValueError) as error:
        if attn_implementation == "sdpa":
            raise error
        kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **kwargs)
    actual_attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
    print(json.dumps({"requested_attn_implementation": attn_implementation, "actual_attn_implementation": actual_attn}, sort_keys=True))
    if actual_attn != attn_implementation:
        raise RuntimeError(f"attention backend mismatch: requested {attn_implementation}, loaded {actual_attn}")
    if lora_cfg.get("enabled", True):
        if init_adapter:
            # Continue from an existing LoRA adapter (e.g. Final SFT) instead of a fresh one.
            model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=int(lora_cfg.get("r", 16)), lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
                lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
                target_modules=list(lora_cfg.get("target_modules", REQUIRED_LORA_TARGETS)),
                task_type=TaskType.CAUSAL_LM,
            ), autocast_adapter_dtype=False)
    return model, processor, getattr(model, "targeted_module_names", ())


def load_baseline_model_and_processor(model_path: Path, lora_cfg: dict[str, Any], attn_implementation: str = "sdpa", processor_backend: str = "torchvision"):
    return load_model_and_processor(model_path, lora_cfg, attn_implementation, processor_backend)


def make_training_arguments(output_dir: Path, *, learning_rate: float = 2e-5,
                            epochs: float = 2, per_device_batch_size: int = 2,
                            gradient_accumulation_steps: int = 8,
                            warmup_ratio: float = 0.1, logging_steps: int = 10,
                            save_steps: int = 500, max_steps: int = -1,
                            bf16: bool | None = None, gradient_checkpointing: bool = True,
                            seed: int = 42, warmup_steps: int = 0,
                            dataloader_num_workers: int = 0):
    from transformers import TrainingArguments
    if warmup_steps == 0 and max_steps > 0:
        warmup_steps = int(max_steps * warmup_ratio)

    if bf16 is None:
        bf16 = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate, lr_scheduler_type="cosine",
        warmup_steps=warmup_steps, logging_steps=logging_steps, save_steps=save_steps,
        save_strategy="steps", bf16=bf16, remove_unused_columns=False,
        report_to=[], max_steps=max_steps, gradient_checkpointing=gradient_checkpointing,
        seed=seed, dataloader_num_workers=dataloader_num_workers,
    )


class BaselineTrainer(__import__("transformers").Trainer):
    """Shared optimizer, scheduler, accumulation, logging and checkpoint path."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Keep auxiliary baseline tensors out of generated prediction labels while
        # retaining them for the model's custom loss during training.
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model = self.model
        hook = getattr(model, "save_baseline_pretrained", None)
        if hook is not None:
            hook(output)
        elif hasattr(model, "save_pretrained"):
            model.save_pretrained(str(output), state_dict=state_dict)
        else:
            torch.save(state_dict or model.state_dict(), output / "pytorch_model.bin")
        processor = getattr(self, "processing_class", None)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(str(output))
        if hasattr(self.args, "to_dict"):
            (output / "training_args.json").write_text(
                json.dumps(self.args.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
            )


class PlainSFTTrainer(BaselineTrainer):
    pass


def save_wrapped_model(model: torch.nn.Module, output: Path) -> None:
    """Save a baseline wrapper and its adapter without requiring a HF subclass."""
    output.mkdir(parents=True, exist_ok=True)
    hook = getattr(model, "save_baseline_pretrained", None)
    if hook is not None:
        hook(output)
        return
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output))
        return
    torch.save(model.state_dict(), output / "pytorch_model.bin")
