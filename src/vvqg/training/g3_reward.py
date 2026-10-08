"""G3 conservative support-preserving reward (local, deterministic, no new API).

R = 1.00*support_continuous + 0.20*coverage_anchor - 0.25*duplicate_penalty
    - 0.15*generic_penalty
"""

from __future__ import annotations

import re

VIOLATION = "CONFIRMED_VIOLATION"
W_SUPPORT, W_ANCHOR, W_DUP, W_GENERIC = 1.00, 0.20, 0.25, 0.15

_GENERIC_PATTERNS = (
    re.compile(r"^what (is|are) (in|on) (the|this) (image|picture|photo)\??$"),
    re.compile(r"^what is (this|that|it)\??$"),
    re.compile(r"^is this (a|an) (picture|photo|image)\??$"),
    re.compile(r"^what (is|are) (this|these|the image|the picture)\??$"),
    re.compile(r"^what (do|can) you see\??$"),
    re.compile(r"^describe (the|this) (image|picture|photo)\??$"),
    re.compile(r"^what color is the (image|picture|photo)\??$"),
)
_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "do", "does", "did",
    "can", "could", "will", "would", "there", "here", "this", "that", "these", "those",
    "it", "its", "in", "on", "of", "to", "at", "with", "and", "or", "how", "many",
    "much", "what", "which", "who", "where", "when", "why", "any", "some",
}
_GENERIC_CONTENT = {
    "image", "picture", "photo", "thing", "things", "object", "objects", "scene",
    "color", "colors", "colour", "colours", "see", "shown", "visible", "appear",
    "looks", "look", "something", "anything",
}


def _has_concrete_entity(text: str) -> bool:
    content = [w.strip("?.!,") for w in text.split()]
    content = [w for w in content if w and w not in _STOPWORDS]
    return any(w not in _GENERIC_CONTENT for w in content)


def norm_question(q: str) -> str:
    return " ".join((q or "").strip().lower().split())


def support_continuous(support_floor: float | None, flat_state: str | None, malformed: bool) -> float:
    if malformed or support_floor is None or flat_state in (None, "FAILED"):
        return -1.0
    value = max(-1.0, min(1.0, 2.0 * (float(support_floor) - 0.5)))
    if flat_state == VIOLATION:
        value -= 0.5
    return value


def rank_anchor(uvfc: list[float]) -> list[float]:
    """Within-group rank normalisation to [-1, 1]."""
    n = len(uvfc)
    if n <= 1:
        return [0.0] * n
    order = sorted(range(n), key=lambda i: uvfc[i])
    out = [0.0] * n
    for rank, i in enumerate(order):
        out[i] = 2.0 * (rank / (n - 1)) - 1.0
    return out


def duplicate_flags(questions: list[str]) -> list[float]:
    counts: dict[str, int] = {}
    for q in questions:
        counts[norm_question(q)] = counts.get(norm_question(q), 0) + 1
    return [1.0 if counts[norm_question(q)] > 1 else 0.0 for q in questions]


def generic_flags(questions: list[str]) -> list[float]:
    flags = []
    for q in questions:
        text = norm_question(q)
        words = text.split()
        short = len(words) <= 2
        template = any(p.match(text) for p in _GENERIC_PATTERNS)
        flags.append(1.0 if (short or template or not _has_concrete_entity(text)) else 0.0)
    return flags


def group_rewards(members: list[dict]) -> list[dict]:
    """members: dicts with question, support_floor, flat_state, UVFC."""
    questions = [m["question"] for m in members]
    anchors = rank_anchor([float(m.get("UVFC", 0.0)) for m in members])
    dups = duplicate_flags(questions)
    generics = generic_flags(questions)
    out = []
    for m, anchor, dup, generic in zip(members, anchors, dups, generics):
        malformed = len(str(m["question"]).strip()) < 2
        support = support_continuous(m.get("support_floor"), m.get("flat_state"), malformed)
        total = W_SUPPORT * support + W_ANCHOR * anchor - W_DUP * dup - W_GENERIC * generic
        out.append({**m, "support_continuous": support, "coverage_anchor": anchor,
                    "duplicate_penalty": dup, "generic_penalty": generic, "g3_reward": total})
    return out
