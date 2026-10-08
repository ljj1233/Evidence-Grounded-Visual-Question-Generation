"""B2 cache records and immutable embedding lookup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def _stable_text_hash(text: str) -> str:
    return hashlib.sha256(json.dumps(text, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def load_embedding_cache(path: Path) -> tuple[dict[str, torch.Tensor], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dim = int(payload["embedding_dim"])
    vectors = {key: torch.tensor(value, dtype=torch.float32) for key, value in payload["vectors"].items()}
    if any(vector.numel() != dim for vector in vectors.values()):
        raise AssertionError("embedding cache dimension mismatch")
    for vector in vectors.values():
        vector.requires_grad_(False)
    return vectors, dim


def load_image_caches(path: Path) -> dict[str, dict[str, str]]:
    """Join the three frozen image-level caches and fail on incomplete rows."""
    caches: dict[str, dict[str, str]] = {}
    qtxt_by_caption: dict[str, str] = {}
    qtxt_by_image: dict[str, str] = {}
    for line in (path / "q_txt.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        qtxt_by_caption[str(row["caption_hash"])] = str(row["question"])
        qtxt_by_image[str(row["image_id"])] = str(row["question"])
    for name, field in (("captions", "caption"), ("q_img", "q_img")):
        for line in (path / f"{name}.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            value = row.get("caption" if name == "captions" else "question")
            caches.setdefault(str(row["image_id"]), {})[field] = str(value)
    for image_id, row in caches.items():
        key = text_hash(row["caption"])
        q_txt = qtxt_by_caption.get(key)
        if q_txt is None:
            q_txt = qtxt_by_caption.get(_stable_text_hash(row["caption"]))
        if q_txt is None:
            q_txt = qtxt_by_image.get(str(image_id))
        if q_txt is None:
            raise ValueError(f"missing q_txt cache for image {image_id}")
        row["q_txt"] = q_txt
    missing = [image_id for image_id, row in caches.items() if set(row) != {"caption", "q_img", "q_txt"}]
    if missing:
        raise ValueError(f"incomplete B2 image cache rows: {missing[:5]}")
    return caches


__all__ = ["text_hash", "load_embedding_cache", "load_image_caches"]
