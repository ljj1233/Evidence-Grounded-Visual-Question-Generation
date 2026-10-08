"""CER diagnostics: question length, duplicate and generic rates."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from vvqg.training.fixed_group_relative import norm_question
from vvqg.training.g3_reward import generic_flags


def within_image_duplicate_rate(questions: Sequence[str], image_ids: Sequence[str]) -> float:
    """Duplicate rate computed only among questions sharing the same image."""
    groups: dict[str, list[str]] = defaultdict(list)
    for question, image_id in zip(questions, image_ids):
        groups[str(image_id)].append(norm_question(question))
    total = len(questions)
    duplicates = sum(sum(count - 1 for count in Counter(qs).values() if count > 1) for qs in groups.values())
    return duplicates / total if total else 0.0


def global_duplicate_rate(questions: Sequence[str]) -> float:
    total = len(questions)
    duplicates = sum(count - 1 for count in Counter(norm_question(q) for q in questions).values() if count > 1)
    return duplicates / total if total else 0.0


def generic_rate(questions: Sequence[str]) -> float:
    if not questions:
        return 0.0
    return sum(generic_flags(questions)) / len(questions)


def template_dominance(questions: Sequence[str]) -> float:
    if not questions:
        return 0.0
    counts = Counter(norm_question(q) for q in questions)
    return max(counts.values()) / len(questions)


def question_stats(processor: Any, questions: Sequence[str], image_ids: Sequence[str]) -> dict[str, Any]:
    lengths = [len(processor.tokenizer(q, add_special_tokens=False)["input_ids"]) for q in questions]
    total = len(questions)
    return {
        "n": total,
        "mean_token_length": sum(lengths) / total if total else 0.0,
        "max_token_length": max(lengths) if lengths else 0,
        "duplicate_rate_within_image": within_image_duplicate_rate(questions, image_ids),
        "duplicate_rate_global": global_duplicate_rate(questions),
        "generic_rate": generic_rate(questions),
        "template_dominance": template_dominance(questions),
    }
