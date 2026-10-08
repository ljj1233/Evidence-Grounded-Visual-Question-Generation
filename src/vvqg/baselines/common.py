"""Small shared contracts for the same-backbone baseline implementations."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SEED = 42
BACKBONE = "Qwen3-VL-8B-Instruct"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def seed_everything(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_generator(seed: int = SEED):
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def stable_cache_key(image_id: str | int, model_hash: str, prompt_hash: str, **config: object) -> str:
    return stable_hash({"image_id": str(image_id), "model_hash": model_hash, "prompt_hash": prompt_hash, **config})


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_ids(rows: Iterable[dict]) -> set[str]:
    return {str(row["image_id"]) for row in rows}


def assert_disjoint(named_sets: dict[str, set[str]]) -> None:
    names = list(named_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = named_sets[left] & named_sets[right]
            if overlap:
                raise AssertionError(f"image-id overlap {left}/{right}: {sorted(overlap)[:5]}")


def lora_kwargs() -> dict[str, object]:
    return {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": list(LORA_TARGETS)}


def masked_mean(hidden_states, mask):
    """Mean-pool only target tokens; ``mask`` is broadcast over hidden size."""
    import torch

    weights = mask.to(hidden_states.dtype).unsqueeze(-1)
    return (hidden_states * weights).sum(1) / weights.sum(1).clamp_min(1)


def normalize(x):
    import torch.nn.functional as F

    return F.normalize(x, dim=-1)


def cosine_distance(a, b):
    import torch

    return 1.0 - (a * b).sum(-1)


def assert_exact_count(rows: Sequence[object], expected: int, label: str) -> None:
    if len(rows) != expected:
        raise AssertionError(f"{label}: expected {expected}, got {len(rows)}")
