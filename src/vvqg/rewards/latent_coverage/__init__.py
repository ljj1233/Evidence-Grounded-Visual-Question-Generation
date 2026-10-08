"""Frozen latent coverage reward (Phase 5A)."""

from vvqg.rewards.latent_coverage.coverage import (
    assign_clusters,
    coverage_scores,
    kmeans,
    load_coverage,
    visual_difference_repr,
)

__all__ = [
    "assign_clusters",
    "coverage_scores",
    "kmeans",
    "load_coverage",
    "visual_difference_repr",
]
