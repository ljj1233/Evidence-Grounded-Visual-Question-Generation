"""Geometry statistics on known configurations (no GPU)."""
import math

import pytest

from egvqg.geometry import effective_rank, pairwise_cosine_mean


def test_identical_directions_are_fully_anisotropic_and_orthogonal_ones_are_not():
    n = 4
    assert pairwise_cosine_mean(n * n, n) == pytest.approx(1.0)       # all unit vectors equal: ||sum||^2 = n^2
    assert pairwise_cosine_mean(n, n) == pytest.approx(0.0)           # orthonormal: ||sum||^2 = n
    assert math.isnan(pairwise_cosine_mean(1, 1))


def test_effective_rank_counts_equal_singular_values_and_ignores_a_dominant_one():
    assert effective_rank([1, 1, 1, 1]) == pytest.approx(4.0)
    assert effective_rank([1, 0, 0]) == pytest.approx(1.0)
    assert 1.0 < effective_rank([10, 1, 1]) < 2.0
    assert effective_rank([0, 0]) == 0.0
