"""Exact generation-prompt reconstruction and completion-token log-probabilities.

The behavior policy is the frozen Final SFT policy; the completion distribution
is recovered from the same prompt the rollouts used
(``VQGGenerator`` / ``generate.py``: image + "Generate a question about this
image." with the Qwen3-VL chat template).  Only completion tokens are scored;
prompt and padding tokens are masked out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from vvqg.generation.generator import DEFAULT_INSTRUCTION

IMAGE_KWARGS = {"min_pixels": 4096, "max_pixels": 147456}
KEEP_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _load_image(path: str):
    from PIL import Image

    with Image.open(_resolve(path)) as source:
        return source.convert("RGB")


def prompt_text(processor: Any, instruction: str = DEFAULT_INSTRUCTION) -> str:
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def full_text(processor: Any, question: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    return processor.apply_chat_template(
        [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
            {"role": "assistant", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


_IMAGE_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def _cached_image(processor: Any, image_path: str, instruction: str) -> dict[str, Any]:
    """Image-processor output cached per image (resize/normalize is the expensive part)."""
    key = (image_path, IMAGE_KWARGS["min_pixels"], IMAGE_KWARGS["max_pixels"], instruction)
    if key not in _IMAGE_CACHE:
        image = _load_image(image_path)
        processed = processor.image_processor(images=[image], return_tensors="pt", **IMAGE_KWARGS)
        replacement = processor.replace_image_token(processed, 0)
        prompt_replaced = prompt_text(processor, instruction).replace(processor.image_token, replacement)
        _IMAGE_CACHE[key] = {
            "pixel_values": processed["pixel_values"],
            "image_grid_thw": processed["image_grid_thw"],
            "replacement": replacement,
            "prompt_ids": processor.tokenizer(prompt_replaced, add_special_tokens=False)["input_ids"],
        }
    return _IMAGE_CACHE[key]


def encode_items(processor: Any, items: Sequence[tuple[str, str]], instruction: str = DEFAULT_INSTRUCTION) -> dict[str, Any]:
    """Encode a batch of ``(image_path, question)`` items.

    ``completion_start[i]`` is the index of the first completion token;
    ``completion_token_ids[i]`` is the exact question token sequence, verified to
    equal the tail of the full sequence.  The image processor runs once per
    unique image; the placeholder expansion mirrors ``ProcessorMixin.__call__``.
    """
    import torch

    probes = [_cached_image(processor, image_path, instruction) for image_path, _ in items]
    texts = [full_text(processor, question, instruction) for _, question in items]
    replaced, _ = processor.get_text_with_replacements(texts, [probe["replacement"] for probe in probes])
    tokenized = processor.tokenizer(replaced, padding=True)
    full_ids = tokenized["input_ids"]
    inputs = {
        "input_ids": torch.tensor(full_ids),
        "attention_mask": torch.tensor(tokenized["attention_mask"]),
        "pixel_values": torch.cat([probe["pixel_values"] for probe in probes], dim=0),
        "image_grid_thw": torch.cat([probe["image_grid_thw"] for probe in probes], dim=0),
        "mm_token_type_ids": torch.tensor(processor.create_mm_token_type_ids(full_ids)),
    }
    starts, token_ids = [], []
    for row, probe, (_, question) in zip(full_ids, probes, items):
        prompt_ids = probe["prompt_ids"]
        if row[: len(prompt_ids)] != prompt_ids:
            raise ValueError("prompt is not a prefix of the full sequence")
        question_ids = processor.tokenizer(question, add_special_tokens=False)["input_ids"]
        if row[len(prompt_ids) : len(prompt_ids) + len(question_ids)] != question_ids:
            raise ValueError("question tokens are not contiguous after the prompt")
        starts.append(len(prompt_ids))
        token_ids.append(question_ids)
    return {"inputs": inputs, "completion_start": starts, "completion_token_ids": token_ids}


def encode_group(processor: Any, image_path: str, questions: Sequence[str], instruction: str = DEFAULT_INSTRUCTION) -> dict[str, Any]:
    return encode_items(processor, [(image_path, question) for question in questions], instruction)


def gather_token_logprobs(logits: Any, input_ids: Any, starts: Sequence[int], counts: Sequence[int]) -> list[Any]:
    """Per-response completion-token log-probs of ``input_ids`` under ``logits``."""
    import torch

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for index, (start, count) in enumerate(zip(starts, counts)):
        positions = torch.arange(start - 1, start + count - 1, device=logits.device)
        tokens = input_ids[index, start : start + count]
        out.append(log_probs[index, positions, tokens])
    return out


def to_device(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def verify_encoding(processor: Any, items: Sequence[tuple[str, str]], instruction: str = DEFAULT_INSTRUCTION) -> None:
    """Assert the cached manual encoder reproduces ``ProcessorMixin.__call__`` exactly."""
    import torch

    manual = encode_items(processor, items, instruction)["inputs"]
    reference = processor(
        text=[full_text(processor, question, instruction) for _, question in items],
        images=[_load_image(image_path) for image_path, _ in items],
        padding=True,
        return_tensors="pt",
        **IMAGE_KWARGS,
    )
    for key in KEEP_KEYS:
        if key not in reference:
            raise AssertionError(f"reference missing {key}")
        if manual[key].shape != reference[key].shape or not torch.equal(manual[key], reference[key]):
            raise AssertionError(f"encoder mismatch on {key}: {tuple(manual[key].shape)} vs {tuple(reference[key].shape)}")
