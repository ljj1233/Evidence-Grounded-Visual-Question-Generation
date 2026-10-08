"""Canonical image-conditioned generation from the base model or SFT adapter.

Key efficiency points (measured):
- the Qwen image processor is ~100x faster with the **PIL backend** than the
  torchvision backend (0.01 s/img vs 0.96 s/img); we default to ``backend="pil"``;
- images are processed in **cross-image batches** (``batch_size``) so the GPU is
  fed large batches instead of one image at a time;
- N samples per image are produced in a single batched call (each prompt
  repeated N times), so sampling costs almost the same as greedy;
- the model is loaded once and reused for many batches.

Determinism: a stable seed is set per batch chunk, derived from the base seed and
the chunk's image ids. Re-running with the same ``batch_size`` reproduces outputs.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEFAULT_INSTRUCTION = "Generate a question about this image."


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16) % (2**31 - 1)


@dataclass
class GenerationConfig:
    base_model: str
    checkpoint: Optional[str] = None
    instruction: str = DEFAULT_INSTRUCTION
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    backend: str = "pil"  # "pil" (fast) or "torchvision"
    attn_implementation: str = "flash_attention_2"
    device: str = "cuda"
    batch_size: int = 16
    num_samples: int = 1
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 32
    seed: int = 42
    dtype: str = "bfloat16"


class VQGGenerator:
    def __init__(self, config: GenerationConfig) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.config = config
        self.torch = torch
        # `use_fast=False` selects the PIL image processor (~100x faster here than
        # the torchvision backend). The `backend=` argument is not accepted for
        # this processor in the installed transformers version.
        self.processor = AutoProcessor.from_pretrained(
            config.base_model, local_files_only=True, use_fast=(config.backend != "pil")
        )
        self.processor.tokenizer.padding_side = "left"
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            if config.min_pixels is not None and hasattr(image_processor, "min_pixels"):
                image_processor.min_pixels = config.min_pixels
            if config.max_pixels is not None and hasattr(image_processor, "max_pixels"):
                image_processor.max_pixels = config.max_pixels
        base = AutoModelForImageTextToText.from_pretrained(
            config.base_model,
            dtype=getattr(torch, config.dtype),
            local_files_only=True,
            attn_implementation=config.attn_implementation,
        )
        if config.checkpoint is not None:
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, config.checkpoint)
        self.model = base
        self.oom_splits = 0
        self.model.eval()
        if config.device.startswith("cuda") and torch.cuda.is_available():
            self.model.to(config.device)
        self.prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": config.instruction}]}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------ helpers
    def _image_kwargs(self) -> dict:
        kwargs = {}
        if self.config.min_pixels is not None:
            kwargs["min_pixels"] = self.config.min_pixels
        if self.config.max_pixels is not None:
            kwargs["max_pixels"] = self.config.max_pixels
        return kwargs

    def _load_image(self, image_path: str):
        from PIL import Image

        with Image.open(image_path) as source:
            return source.convert("RGB")

    def generate_batch(self, image_paths: Sequence[str], seed_key: str) -> list[list[str]]:
        """Return ``len(image_paths)`` lists, each with ``num_samples`` questions."""
        n = self.config.num_samples
        greedy = n == 1 and self.config.temperature <= 0
        images = [self._load_image(path) for path in image_paths]
        texts = [self.prompt] * len(images) * n
        image_inputs = [im for im in images for _ in range(n)]
        inputs = self.processor(
            text=texts, images=image_inputs, padding=True, return_tensors="pt", **self._image_kwargs()
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        torch = self.torch
        torch.manual_seed(stable_seed(self.config.seed, seed_key))
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=not greedy,
                temperature=None if greedy else self.config.temperature,
                top_p=None if greedy else self.config.top_p,
            )
        texts_out = [
            self.processor.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip() for row in output
        ]
        return [texts_out[i * n : (i + 1) * n] for i in range(len(images))]

    def generate_many(self, items: Sequence[tuple[str, str]], progress=None) -> list[tuple[str, list[str]]]:
        """``items`` is a list of ``(image_id, image_path)``; returns the same order."""
        results: list[tuple[str, list[str]]] = []
        batch = self.config.batch_size
        for start in range(0, len(items), batch):
            chunk = items[start : start + batch]
            results.extend(self._generate_chunk(chunk))
            if progress is not None:
                progress(min(start + batch, len(items)), len(items))
        return results

    def _generate_chunk(self, chunk: Sequence[tuple[str, str]]) -> list[tuple[str, list[str]]]:
        try:
            outputs = self.generate_batch([path for _, path in chunk], ",".join(iid for iid, _ in chunk))
            return [(iid, questions) for (iid, _), questions in zip(chunk, outputs)]
        except self.torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            self.oom_splits += 1
        self.torch.cuda.empty_cache()
        half = len(chunk) // 2
        return self._generate_chunk(chunk[:half]) + self._generate_chunk(chunk[half:])
