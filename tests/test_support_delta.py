from __future__ import annotations

import math

import pytest

from vvqg.support_delta.reward import (
    coverage_guard,
    group_advantages,
    normalized_rank,
    support_delta_reward,
    support_score,
)


def test_support_score():
    assert support_score(1.0, "NO_CONFIRMED_VIOLATION", False) == 1.0
    assert support_score(0.5, "UNCERTAIN", False) == 0.0
    assert support_score(0.0, "UNCERTAIN", False) == -1.0
    assert support_score(1.0, "CONFIRMED_VIOLATION", False) == 0.5
    assert support_score(None, None, True) == -1.0
    assert support_score(1.0, "FAILED", False) == -1.0


def test_normalized_rank_bounds():
    values = [0.0, 1.0, 2.0]
    assert normalized_rank(values, 0) == 0.0
    assert normalized_rank(values, 2) == 1.0
    assert normalized_rank(values, 1) == 0.5


def test_coverage_guard_triggers_when_rollout_drops():
    assert coverage_guard([1.0, 0.0], q0_index=0, rollout_index=1) == pytest.approx(0.9)
    assert coverage_guard([0.0, 1.0], q0_index=0, rollout_index=1) == 0.0


def test_support_delta_reward_formula():
    result = support_delta_reward(
        support_floor=1.0, flat_state="NO_CONFIRMED_VIOLATION", malformed=False,
        q0_score=0.0, uvfc_group=[1.0, 0.0], q0_index=0, rollout_index=1,
        duplicate=True, parser_failed=False,
    )
    assert result["delta_support"] == pytest.approx(1.0)
    assert result["coverage_guard"] == pytest.approx(0.9)
    assert result["reward"] == pytest.approx(1.0 - 0.5 * 0.9 - 0.25 * 1.0)


def test_malformed_penalty():
    result = support_delta_reward(
        support_floor=None, flat_state="FAILED", malformed=True,
        q0_score=0.0, uvfc_group=[1.0, 0.0], q0_index=0, rollout_index=1,
        duplicate=False, parser_failed=True,
    )
    assert result["support_score"] == -1.0
    assert result["malformed_penalty"] == 1.0
    assert result["reward"] == pytest.approx(-1.0 - 0.5 * 0.9 - 0.5 * 1.0)


def test_group_advantages_zero_variance():
    advantages, std, zero = group_advantages([1.0, 1.0, 1.0])
    assert zero and std == 0.0 and advantages == [0.0, 0.0, 0.0]
    advantages, std, zero = group_advantages([0.0, 2.0])
    assert not zero
    assert math.isclose(advantages[0], -1.0) and math.isclose(advantages[1], 1.0)
