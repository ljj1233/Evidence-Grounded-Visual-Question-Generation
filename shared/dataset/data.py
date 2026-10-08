"""Dataset-neutral VQG schema, auditing, and deterministic image-level subsets."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError


REQUIRED_FIELDS = {
    "sample_id", "dataset", "split", "image_id", "image_path", "question_id", "question",
    "answers", "answer_type", "question_type", "metadata",
}
QUESTION_TYPES = {"yes_no", "count", "color", "object", "attribute", "action", "spatial", "relation", "who", "where", "why_how", "other"}


def coarse_question_type(question: str) -> str:
    value = question.strip().lower()
    if re.match(r"^(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", value):
        return "yes_no"
    if re.match(r"^how (many|much)\b", value):
        return "count"
    if re.match(r"^what colou?r\b", value):
        return "color"
    if re.match(r"^who\b", value):
        return "who"
    if re.match(r"^where\b", value):
        return "where"
    if re.match(r"^(why|how)\b", value):
        return "why_how"
    if any(word in value for word in ("left of", "right of", "behind", "in front of", "above", "below", "under", "between")):
        return "spatial"
    if any(word in value for word in ("next to", "holding", "wearing", "riding", "standing", "sitting", "doing")):
        return "relation" if "next to" in value else "action"
    if value.startswith(("what", "which")):
        return "object"
    return "other"


def validate_record(record: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"missing required VQG schema fields: {', '.join(missing)}")
    for field in ("sample_id", "dataset", "split", "image_id", "image_path", "question_id", "question", "answer_type", "question_type"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if not isinstance(record["answers"], list) or not all(isinstance(answer, str) for answer in record["answers"]):
        raise ValueError("answers must be a list of strings")
    if record["question_type"] not in QUESTION_TYPES:
        raise ValueError("question_type is not in the controlled VQG taxonomy")
    if not isinstance(record["metadata"], dict):
        raise ValueError("metadata must be an object")


@lru_cache(maxsize=200000)
def image_is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def canonical_record(
    *, dataset: str, split: str, image_id: str, image_path: Path, question_id: str,
    question: str, answers: list[str], answer_type: str | None, metadata: dict[str, Any],
) -> dict[str, Any]:
    question = " ".join(question.split())
    return {
        "sample_id": f"{dataset}:{question_id}", "dataset": dataset, "split": split,
        "image_id": str(image_id), "image_path": str(image_path), "question_id": str(question_id),
        "question": question, "answers": [str(answer) for answer in answers if str(answer).strip()],
        "answer_type": answer_type or "unknown", "question_type": coarse_question_type(question),
        "metadata": metadata,
    }


def filter_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Apply only task-independent, deterministic validity checks."""
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        dataset = str(record.get("dataset", ""))
        original_split = str(record.get("split", ""))
        reason: str | None = None
        if not isinstance(record.get("question"), str) or not record["question"].strip():
            reason = "empty_question"
        else:
            try:
                validate_record(record)
            except ValueError:
                reason = "malformed_annotation"
        if reason is None and not image_is_valid(Path(record["image_path"])):
            reason = "missing_or_invalid_image"
        pair = (str(record.get("image_id", "")), str(record.get("question", "")).strip().casefold())
        if reason is None and pair in seen_pairs:
            reason = "exact_duplicate_image_question"
        if reason is not None:
            filtered.append({"sample_id": sample_id, "dataset": dataset, "original_split": original_split, "reason": reason})
        else:
            seen_pairs.add(pair)
            kept.append(record)
    return kept, filtered


def build_image_level_subsets(records: Iterable[dict[str, Any]], sizes: dict[str, int], *, seed: int) -> dict[str, list[dict[str, Any]]]:
    """Allocate every image to at most one deterministic named subset."""
    rows = list(records)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        validate_record(row)
        by_image[row["image_id"]].append(row)
    images = sorted(by_image)
    rng = random.Random(seed)
    rng.shuffle(images)
    result: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for name, requested in sizes.items():
        if requested < 0:
            raise ValueError("subset sizes must be non-negative")
        selection = images[offset: offset + requested]
        offset += requested
        result[name] = [row for image_id in sorted(selection) for row in by_image[image_id]]
    return result


def assert_no_image_overlap(splits: dict[str, Iterable[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for split, records in splits.items():
        for row in records:
            image_id = str(row["image_id"])
            previous = seen.setdefault(image_id, split)
            if previous != split:
                raise ValueError(f"image-level leakage: {image_id} occurs in {previous} and {split}")


def composition_balanced_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic sample balanced across ``composition_bin`` with shortfall redistribution.

    Bins are naturally skewed; balancing makes the rare high-composition buckets
    measurable. The same seed and rows always yield the same selection.
    """
    by_bin: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        plan = row.get("gold_evidence_plan") or {}
        by_bin.setdefault(str(plan.get("composition_bin", "unknown")), []).append(row)
    bins = sorted(by_bin)
    if not bins:
        return []
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {}
    for name in bins:
        pool = sorted(by_bin[name], key=lambda row: row["sample_id"])
        rng.shuffle(pool)
        pools[name] = pool
    target_per = max(1, size // len(bins))
    taken = {name: 0 for name in bins}
    selected: list[dict[str, Any]] = []
    for name in bins:
        take = min(target_per, len(pools[name]))
        selected.extend(pools[name][:take])
        taken[name] = take
    remaining = size - len(selected)
    while remaining > 0:
        progressed = False
        for name in bins:
            if remaining <= 0:
                break
            if taken[name] < len(pools[name]):
                selected.append(pools[name][taken[name]])
                taken[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    selected.sort(key=lambda row: row["sample_id"])
    return selected[:size]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, returning no rows when the file does not exist."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Durably append complete records so an interrupted long run can resume."""
    rows = list(records)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    counts = Counter(row["question_type"] for row in rows)
    by_image = Counter(str(row["image_id"]) for row in rows)
    duplicate_count = len(rows) - len({(row["image_id"], row["question"].casefold()) for row in rows})
    lengths = [len(row["question"].split()) for row in rows]
    return {
        "images": len(by_image), "questions": len(rows), "questions_per_image": (len(rows) / len(by_image)) if by_image else 0.0,
        "question_length": {"min": min(lengths, default=0), "max": max(lengths, default=0), "mean": (sum(lengths) / len(lengths)) if lengths else 0.0},
        "duplicate_count": duplicate_count, "invalid_count": 0,
        "answer_type_distribution": dict(sorted(Counter(row["answer_type"] for row in rows).items())),
        "question_type_distribution": dict(sorted(counts.items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
    }
