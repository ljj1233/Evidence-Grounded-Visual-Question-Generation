"""SeeQ model-based visual-validity + difficulty-improvement filtering.

Implements PAPER 3.3 ("retains candidates that are visually valid and improve
over their proposals in either perception or reasoning difficulty") using the
executable scoring rules of OFFICIAL_CODE utils/gen/selfjudge_utils.py and
utils/gen/retrieval_utils.py.  Parser failure is never accepted.
"""
from __future__ import annotations

import re

from ..seeq_style.pipeline import _json_output

# OFFICIAL_CODE selfjudge_utils._is_compound_question
def is_compound_question(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if text.count("?") >= 2:
        return True
    return bool(re.search(r",\s+and\s+(what|how|why|which|where|when)\b", text, re.IGNORECASE))


# OFFICIAL_CODE retrieval_utils.lexical_overlap_ratio
def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower())) if isinstance(text, str) else set()


def lexical_overlap_ratio(a: str, b: str) -> float:
    ta = {t for t in _tokenize(a) if len(t) > 2}
    tb = {t for t in _tokenize(b) if len(t) > 2}
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def hard_constraints_valid(judge) -> tuple[bool, str]:
    """OFFICIAL_CODE is_valid_hard_constraints: Factuality=0, Ambiguity=0, answerable."""
    if not isinstance(judge, dict):
        return False, "invalid_judge"
    hard = judge.get("hard_constraints")
    if not isinstance(hard, dict):
        return False, "missing_hard_constraints"
    factuality = hard.get("Factuality")
    ambiguity = hard.get("Ambiguity")
    validity = hard.get("Validity")
    factuality = factuality.get("count") if isinstance(factuality, dict) else factuality
    ambiguity = ambiguity.get("count") if isinstance(ambiguity, dict) else ambiguity
    answerable = validity.get("answerable") if isinstance(validity, dict) else validity
    if not isinstance(factuality, (int, float)) or not isinstance(ambiguity, (int, float)):
        return False, "invalid_hard_constraint_counts"
    if isinstance(answerable, (int, float)) and not isinstance(answerable, bool):
        answerable = bool(answerable)
    if not isinstance(answerable, bool):
        return False, "invalid_answerability"
    if factuality or ambiguity or not answerable:
        return False, "hard_constraint_failed"
    return True, ""


def perception_score(judge) -> float | None:
    """OFFICIAL_CODE: count = hard_count*2 + easy_count*1 (visual grounding load)."""
    if not isinstance(judge, dict):
        return None
    image = (judge.get("soft_constraints", {}) or {}).get("perception_difficulty", {}).get("image")
    if not isinstance(image, dict):
        return None
    easy, hard = image.get("easy_count"), image.get("hard_count")
    if not isinstance(easy, (int, float)) or not isinstance(hard, (int, float)):
        return None
    return float(hard) * 2.0 + float(easy) * 1.0


def reasoning_score(judge) -> float | None:
    """OFFICIAL_CODE reasoning_difficulty.count."""
    if not isinstance(judge, dict):
        return None
    value = (judge.get("soft_constraints", {}) or {}).get("reasoning_difficulty", {}).get("count")
    return float(value) if isinstance(value, (int, float)) else None


def has_difficulty_improved(reasoning_delta: float, perception_delta: float) -> bool:
    """OFFICIAL_CODE selfjudge_utils.has_difficulty_improved thresholds."""
    if reasoning_delta >= 0.5:
        return True
    if reasoning_delta >= 0.0 and perception_delta >= 1.0:
        return True
    if reasoning_delta > 0.0 and perception_delta > 0.0:
        return True
    return False


def decide(*, hard_judge, perception_seed, perception_rew, reasoning_seed, reasoning_rew,
           seed_question: str, rewritten_question: str, seen_questions: set[str], ) -> dict:
    """Apply the SeeQ acceptance rule to one rewritten candidate.

    valid_visual AND (perception_improved OR reasoning_improved), plus the
    released non-duplicate / lexical-overlap / single-question gates.
    """
    valid_visual, validity_reason = hard_constraints_valid(hard_judge)
    parse_failed = (
        perception_rew is None or reasoning_rew is None
        or perception_seed is None or reasoning_seed is None
    )
    perception_delta = None if parse_failed else perception_rew - perception_seed
    reasoning_delta = None if parse_failed else reasoning_rew - reasoning_seed
    perception_improved = bool(perception_delta is not None and perception_delta >= 1.0)
    reasoning_improved = bool(reasoning_delta is not None and reasoning_delta >= 0.5)
    compound = is_compound_question(rewritten_question)
    overlap = lexical_overlap_ratio(rewritten_question, seed_question)
    duplicate = rewritten_question.strip().casefold() in seen_questions
    improved = False if parse_failed else has_difficulty_improved(reasoning_delta, perception_delta)

    if parse_failed:
        accepted, reason = False, "filter_parse_failed"
    elif compound:
        accepted, reason = False, "multi_question"
    elif overlap >= 0.82:
        accepted, reason = False, "lexical_overlap_too_high"
    elif duplicate:
        accepted, reason = False, "duplicate_simplified_question"
    elif not valid_visual:
        accepted, reason = False, validity_reason or "hard_constraint_failed"
    elif not improved:
        accepted, reason = False, "difficulty_not_improved_enough"
    else:
        accepted, reason = True, ""

    return {
        "valid_visual": bool(valid_visual),
        "perception_improved": perception_improved,
        "reasoning_improved": reasoning_improved,
        "improved": improved,
        "perception_delta": perception_delta,
        "reasoning_delta": reasoning_delta,
        "seed_overlap": round(overlap, 3),
        "accepted": accepted,
        "rejection_reason": reason,
    }


__all__ = [
    "is_compound_question", "lexical_overlap_ratio", "hard_constraints_valid",
    "perception_score", "reasoning_score", "has_difficulty_improved", "decide",
]
