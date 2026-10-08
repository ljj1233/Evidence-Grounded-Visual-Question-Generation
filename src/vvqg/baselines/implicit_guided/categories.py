"""Official VQAv2 answer-to-category mapping used by B3."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

DEFAULT_MAPPING = Path(__file__).resolve().parents[2] / "data" / "baselines" / "iq_answer_categories.json"


def load_answer_categories(path: Path = DEFAULT_MAPPING) -> dict[str, tuple[str, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(category): tuple(str(answer).strip().lower() for answer in answers) for category, answers in raw.items()}


def normalize_answer(answer: object) -> str:
    value = str(answer).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .!?;,:\"'")


def answer_category(answer: object, mapping: dict[str, Iterable[str]] | None = None) -> int | None:
    """Map an answer to the fixed 16-way category, or return ``None``."""
    mapping = load_answer_categories() if mapping is None else mapping
    key = normalize_answer(answer)
    for category_id, values in enumerate(mapping.values()):
        if key in {normalize_answer(item) for item in values}:
            return category_id
    return None


def map_category_records(records: Iterable[dict], mapping: dict[str, Iterable[str]] | None = None, answer_key: str = "answer") -> tuple[list[dict], dict]:
    """Attach ``category_id`` while retaining explicit unmapped records."""
    mapping = load_answer_categories() if mapping is None else mapping
    mapped = []
    histogram = Counter()
    for record in records:
        row = dict(record)
        category = answer_category(row.get(answer_key, ""), mapping)
        row["category_id"] = category
        row["category_mapped"] = category is not None
        if category is not None:
            histogram[category] += 1
        mapped.append(row)
    report = {"mapped_count": sum(histogram.values()), "unmapped_count": len(mapped) - sum(histogram.values()), "category_histogram": dict(sorted(histogram.items()))}
    return mapped, report


__all__ = ["DEFAULT_MAPPING", "answer_category", "load_answer_categories", "map_category_records", "normalize_answer"]
