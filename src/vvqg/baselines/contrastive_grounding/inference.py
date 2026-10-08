"""Image-plus-caption inference for the B2 checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from .model import make_b2_messages


class B2Generator:
    """Load one B2 adapter and reuse it for batched caption-conditioned generation."""

    def __init__(self, base_model: str | Path, checkpoint: str | Path, *, device: str = "cuda", batch_size: int = 8, attn_implementation: str = "flash_attention_2"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device, self.batch_size = device, batch_size
        self.processor = AutoProcessor.from_pretrained(str(base_model), local_files_only=True, use_fast=False)
        self.processor.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(
            str(base_model), dtype=torch.bfloat16, local_files_only=True, attn_implementation=attn_implementation
        )
        adapter = Path(checkpoint) / "backbone"
        if (adapter / "adapter_config.json").is_file():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter))
        self.model = model.eval().to(device)

    @staticmethod
    def _image(path: str | Path):
        from PIL import Image

        with Image.open(path) as image:
            return image.convert("RGB")

    def generate(self, items: Sequence[tuple[str | Path, str]], *, max_new_tokens: int = 32, num_samples: int = 1, temperature: float = 0.8, top_p: float = 0.9) -> list[list[str]]:
        outputs: list[str] = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            expanded = [item for item in chunk for _ in range(num_samples)]
            messages = [make_b2_messages("", caption)[:1] for _, caption in expanded]
            prompts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
            images = [self._image(path) for path, _ in expanded]
            batch = self.processor(text=prompts, images=images, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                ids = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=num_samples > 1, temperature=temperature, top_p=top_p)
            width = batch["input_ids"].shape[1]
            outputs.extend(self.processor.tokenizer.decode(row[width:], skip_special_tokens=True).strip() for row in ids)
        return [outputs[i : i + num_samples] for i in range(0, len(outputs), num_samples)]


__all__ = ["B2Generator"]
