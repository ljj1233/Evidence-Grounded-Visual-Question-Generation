"""Batched generation against an already-loaded model.

Mirrors ``VQGGenerator`` (PIL backend, cross-image batching, stable per-chunk
seed) so generation protocol stays identical while reusing a model held in
memory by a trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from vvqg.generation.generator import DEFAULT_INSTRUCTION, stable_seed

IMAGE_KWARGS = {"min_pixels": 4096, "max_pixels": 147456}
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_image(path: str):
    from PIL import Image

    candidate = Path(path)
    with Image.open(candidate if candidate.is_absolute() else REPO_ROOT / candidate) as source:
        return source.convert("RGB")


def model_prompt(processor: Any, instruction: str = DEFAULT_INSTRUCTION) -> str:
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_questions(
    model: Any,
    processor: Any,
    image_paths: Sequence[str],
    prompt: str,
    *,
    num_samples: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int = 32,
    seed: int = 42,
    batch_size: int = 16,
    desc: str | None = None,
) -> list[list[str]]:
    import torch

    greedy = num_samples == 1 and temperature <= 0
    was_training = model.training
    model.eval()
    previous_padding = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    results: list[list[str]] = []
    bar = tqdm(total=len(image_paths), desc=desc, unit="img", dynamic_ncols=True) if desc else None
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            chunk = list(image_paths[start : start + batch_size])
            images = [load_image(path) for path in chunk]
            texts = [prompt] * len(images) * num_samples
            image_inputs = [image for image in images for _ in range(num_samples)]
            inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt", **IMAGE_KWARGS).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            torch.manual_seed(stable_seed(seed, ",".join(chunk)))
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=None if greedy else temperature,
                top_p=None if greedy else top_p,
            )
            decoded = [
                processor.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
                for row in output
            ]
            results.extend(decoded[index * num_samples : (index + 1) * num_samples] for index in range(len(images)))
            if bar:
                bar.update(len(chunk))
    processor.tokenizer.padding_side = previous_padding
    if bar:
        bar.close()
    if was_training:
        model.train()
    return results
