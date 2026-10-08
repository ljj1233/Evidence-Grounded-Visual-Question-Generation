"""Canonical bootstrap statistics for evaluation reports.

All system-level rates are reported with percentile bootstrap 95% CIs; paired
system comparisons resample by stable ``image_id`` (never by row order or by
copying a system-level number onto each sample).
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 42


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def bootstrap_ci(values: Sequence[float], resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED) -> dict[str, float]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    means.sort()
    return {
        "mean": round(sum(values) / n, 6),
        "lo": round(percentile(means, 0.025), 6),
        "hi": round(percentile(means, 0.975), 6),
        "n": n,
    }


def paired_bootstrap_diff(
    left: Sequence[float], right: Sequence[float], resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED
) -> dict[str, float]:
    """Paired bootstrap CI for mean(left - right); inputs aligned by stable key."""
    pairs = [(float(a), float(b)) for a, b in zip(left, right) if a is not None and b is not None]
    if not pairs:
        return {"mean_diff": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(a - b for a, b in sample) / n)
    diffs.sort()
    return {
        "mean_diff": round(sum(a - b for a, b in pairs) / n, 6),
        "lo": round(percentile(diffs, 0.025), 6),
        "hi": round(percentile(diffs, 0.975), 6),
        "n": n,
    }


def distribution_stats(values: Sequence[float]) -> dict[str, Optional[float]]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    variance = sum((v - mean) ** 2 for v in ordered) / len(ordered)
    return {
        "n": len(ordered),
        "mean": round(mean, 6),
        "std": round(variance ** 0.5, 6),
        "p5": round(percentile(ordered, 0.05), 6),
        "p50": round(percentile(ordered, 0.5), 6),
        "p95": round(percentile(ordered, 0.95), 6),
    }


def paired_corpus_bootstrap(left, right, score, resamples: int = 5000, seed: int = 42) -> dict:
    """Resample aligned image-level sufficient statistics and score corpus totals."""
    import numpy as np
    left, right = np.asarray(left), np.asarray(right)
    rng = np.random.default_rng(seed)
    n = len(left)
    draws = np.empty((2, resamples))
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        indices = rng.integers(0, n, (stop - start, n))
        draws[0, start:stop] = score(left[indices].sum(axis=1))
        draws[1, start:stop] = score(right[indices].sum(axis=1))
    estimates = [float(score(x.sum(axis=0))) for x in (left, right)]
    return {
        name: {"estimate": estimate, "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
               "resamples": resamples, "seed": seed, "n_images": n}
        for name, estimate, values in (
            ("left", estimates[0], draws[0]), ("right", estimates[1], draws[1]),
            ("delta", estimates[0] - estimates[1], draws[0] - draws[1]))
    }
