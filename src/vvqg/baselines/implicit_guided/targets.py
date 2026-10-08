"""Training-only object-selection targets for cached detector regions."""

from __future__ import annotations

import re
from collections.abc import Sequence


def _words(value: object) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def object_target(labels: Sequence[str], question: object, answer: object) -> list[float]:
    """Mark detector labels aligned with target question/answer tokens.

    This function is only for training target construction.  Generation does
    not call it and therefore cannot access the gold answer.
    """
    target = _words(question) | _words(answer)
    return [float(bool(_words(label) & target)) for label in labels]


def object_targets(records: Sequence[dict], labels_key: str = "class_labels", question_key: str = "question", answer_key: str = "answer") -> list[list[float]]:
    return [object_target(row.get(labels_key, ()), row.get(question_key, ""), row.get(answer_key, "")) for row in records]


__all__ = ["object_target", "object_targets"]
