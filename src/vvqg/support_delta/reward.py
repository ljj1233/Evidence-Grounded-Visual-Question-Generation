"""Support-Delta reward for the GRPO pilot.

R_i = delta_support_i - 0.50*coverage_guard_i - 0.25*duplicate_penalty_i - 0.50*malformed_penalty_i

delta_support_i = S(q_i) - S(q_0) is the primary term (relative to the frozen
Base anchor q0), UVFC enters only as a within-group coverage guard.
"""

from __future__ import annotations

from typing import Any, Sequence

VIOLATION = "CONFIRMED_VIOLATION"
EPSILON_COV = 0.10
W_COV = 0.50
W_DUP = 0.25
W_MALFORMED = 0.50


def support_score(support_floor: float | None, flat_state: str | None, malformed: bool) -> float:
    if malformed or support_floor is None or flat_state in (None, "FAILED"):
        return -1.0
    value = max(-1.0, min(1.0, 2.0 * float(support_floor) - 1.0))
    if flat_state == VIOLATION:
        value -= 0.5
    return value


def normalized_rank(values: Sequence[float], index: int) -> float:
    """Average-rank percentile in [0, 1]; ties share the mean rank."""
    n = len(values)
    if n <= 1:
        return 0.0
    value = values[index]
    less = sum(1 for other in values if other < value)
    equal = sum(1 for other in values if other == value)
    return (less + (equal - 1) / 2.0) / (n - 1)


def coverage_guard(uvfc_group: Sequence[float], q0_index: int, rollout_index: int, epsilon: float = EPSILON_COV) -> float:
    rank_q0 = normalized_rank(uvfc_group, q0_index)
    rank_rollout = normalized_rank(uvfc_group, rollout_index)
    return max(0.0, rank_q0 - rank_rollout - epsilon)


def support_delta_reward(
    support_floor: float | None,
    flat_state: str | None,
    malformed: bool,
    q0_score: float,
    uvfc_group: Sequence[float],
    q0_index: int,
    rollout_index: int,
    duplicate: bool,
    parser_failed: bool,
) -> dict[str, float]:
    score = support_score(support_floor, flat_state, malformed or parser_failed)
    delta = score - q0_score
    guard = coverage_guard(uvfc_group, q0_index, rollout_index)
    malformed_penalty = 1.0 if (malformed or parser_failed) else 0.0
    duplicate_penalty = 1.0 if duplicate else 0.0
    reward = delta - W_COV * guard - W_DUP * duplicate_penalty - W_MALFORMED * malformed_penalty
    return {
        "support_score": score,
        "delta_support": delta,
        "coverage_guard": guard,
        "duplicate_penalty": duplicate_penalty,
        "malformed_penalty": malformed_penalty,
        "reward": reward,
    }


def group_advantages(rewards: Sequence[float]) -> tuple[list[float], float, bool]:
    """GRPO group standardization; zero-variance groups yield zero advantage."""
    import numpy as np

    values = np.asarray(rewards, dtype=float)
    mean, std = float(values.mean()), float(values.std())
    if std <= 1e-8:
        return [0.0] * len(values), std, True
    return [float((value - mean) / std) for value in values], std, False


def duplicate_flags(questions: Sequence[str]) -> list[bool]:
    from vvqg.training.fixed_group_relative import norm_question

    counts: dict[str, int] = {}
    for question in questions:
        counts[norm_question(question)] = counts.get(norm_question(question), 0) + 1
    return [counts[norm_question(question)] > 1 for question in questions]
