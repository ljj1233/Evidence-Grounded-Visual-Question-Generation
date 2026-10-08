"""Broad question-type taxonomy shared by the overnight diagnostics.

One canonical implementation: yes/no, count, color, what, other.  `other`
absorbs where/who/when/why/how/which and anything unfitting.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

TYPES = ("yes/no", "count", "color", "what", "other")
_YES_NO_PREFIX = (
    "is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ", "could ",
    "will ", "would ", "has ", "have ", "had ", "should ", "shall ", "may ",
    "might ", "is there", "are there", "do you", "does this", "does the",
    "is this", "are these", "is that", "is it", "are they", "is he", "is she",
    "are we", "do they", "can you", "could you", "will there",
)


def broad_type(question: str) -> str:
    q = question.strip().lower()
    if q.startswith("how many") or q.startswith("how much"):
        return "count"
    if q.startswith("what color") or q.startswith("which color"):
        return "color"
    if q.startswith(_YES_NO_PREFIX):
        return "yes/no"
    if q.startswith("what"):
        return "what"
    return "other"


def type_counts(questions: Iterable[str]) -> dict[str, int]:
    counter = Counter(broad_type(q) for q in questions)
    return {t: counter.get(t, 0) for t in TYPES}


def type_entropy(questions: Sequence[str], base: float = 2.0) -> float:
    counts = Counter(broad_type(q) for q in questions)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total, base) for c in counts.values())


def tokens(question: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", question.lower())


def distinct_n(questions: Sequence[str], n: int) -> float:
    grams = [tuple(tokens(q)[i : i + n]) for q in questions for i in range(max(0, len(tokens(q)) - n + 1))]
    return len(set(grams)) / len(grams) if grams else 0.0
