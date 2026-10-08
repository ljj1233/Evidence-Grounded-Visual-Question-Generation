"""Diversity metrics (canonical active metrics).

Distinct-2 (corpus-level)
    total unique word bigrams / total word bigrams, counted over the whole
    candidate list. Bigrams never cross question boundaries. Tokenization is
    lowercase ``[a-z0-9']+``. Questions with fewer than two tokens contribute 0
    to the denominator. Questions are NOT deduplicated before counting.

Normalized excess duplicate rate
    ``1 - unique(normalize(q)) / N`` where ``normalize`` is ``strip().lower()``.
    For ``[A, A, B]`` this is ``1/3`` (excess duplicates), not ``2/3`` (the
    repeated-group membership rate). Punctuation is NOT stripped, so this is a
    case/whitespace-normalized rate, not an exact-duplicate rate.

    ``duplicate_rate`` is kept as a backwards-compatible alias of
    ``normalized_excess_duplicate_rate``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Sequence

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def distinct_2(questions: Sequence[str]) -> float:
    all_bigrams = 0
    unique_bigrams: set = set()
    for question in questions:
        words = tokenize(question)
        for i in range(len(words) - 1):
            unique_bigrams.add((words[i], words[i + 1]))
            all_bigrams += 1
    return round(len(unique_bigrams) / all_bigrams, 6) if all_bigrams else 0.0


def normalized_excess_duplicate_rate(questions: Sequence[str]) -> float:
    if not questions:
        return 0.0
    normalized = [q.strip().lower() for q in questions]
    return round(1 - len(set(normalized)) / len(normalized), 6)


# Backwards-compatible alias; active reports should prefer the explicit name.
duplicate_rate = normalized_excess_duplicate_rate


def repeated_group_membership_rate(questions: Sequence[str]) -> float:
    """Diagnostic alternative: fraction of records whose question occurs >= 2 times."""
    if not questions:
        return 0.0
    normalized = [q.strip().lower() for q in questions]
    counts = Counter(normalized)
    return round(sum(1 for q in normalized if counts[q] >= 2) / len(normalized), 6)


def most_frequent(questions: Sequence[str], top: int = 10) -> list[dict[str, Any]]:
    counts = Counter(q.strip() for q in questions)
    return [{"question": text, "count": count} for text, count in counts.most_common(top)]


def diversity_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Corpus surface diversity and separately exact/normalized within-image counts."""
    import numpy as np
    groups: dict[str, list[str]] = {}
    questions = [r["question"] for r in records]
    for r in records:
        groups.setdefault(str(r["image_id"]), []).append(r["question"])
    if not groups:
        return {"records": 0, "images": 0, "distinct_2": 0.0, "normalized_excess_duplicate_rate": 0.0,
                "unique_question_rate": 0.0, "per_image_exact_unique_question_count": None,
                "per_image_normalized_unique_question_count": None, "rollout_unique_rate": 0.0,
                "normalized_rollout_unique_rate": 0.0, "p_at_least2_normalized_unique": 0.0,
                "rollout_unique_rate_definition": "mean over images of exact unique question count / samples per image; for sampled_4x denominator is 4",
                "normalized_rollout_unique_rate_definition": "mean over images of unique(strip().lower(question)) / samples per image"}
    exact = np.array([len(set(q)) for q in groups.values()])
    normalized = np.array([len({x.strip().lower() for x in q}) for q in groups.values()])
    sizes = np.array([len(q) for q in groups.values()])
    def counts(values):
        return {"mean": float(values.mean()), "median": float(np.median(values)),
                "min": int(values.min()), "max": int(values.max()),
                "distribution": {str(k): v for k, v in sorted(Counter(values.tolist()).items())}}
    return {
        "records": len(questions), "images": len(groups), "distinct_2": distinct_2(questions),
        "normalized_excess_duplicate_rate": normalized_excess_duplicate_rate(questions),
        "unique_question_rate": len({q.strip().lower() for q in questions}) / len(questions),
        "per_image_exact_unique_question_count": counts(exact),
        "per_image_normalized_unique_question_count": counts(normalized),
        "rollout_unique_rate": float((exact / sizes).mean()),
        "normalized_rollout_unique_rate": float((normalized / sizes).mean()),
        "p_at_least2_normalized_unique": float((normalized >= 2).mean()),
        "rollout_unique_rate_definition": "mean over images of exact unique question count / samples per image; for sampled_4x denominator is 4",
        "normalized_rollout_unique_rate_definition": "mean over images of unique(strip().lower(question)) / samples per image",
    }
