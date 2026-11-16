"""Shared statistics: paired bootstrap over parent materials, Wilson intervals, grouped AUC intervals.

All intervals in the VQAG mainline are computed here once. Groups are parent materials: resampling a parent
keeps all of its candidates together, so candidate rows never leak across bootstrap replicates.
"""
from __future__ import annotations

import numpy as np


def _group_index(groups):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    return uniq, members


def paired_bootstrap(diff_fn, groups, n=2000, seed=0, level=0.95):
    """Bootstrap `diff_fn(index_array)` by resampling parent groups.

    Returns point estimate, percentile interval and the replicate count. NaN replicates are dropped; if fewer
    than 20 remain the interval is [nan, nan] (never silently narrowed).
    """
    uniq, members = _group_index(groups)
    rng = np.random.default_rng(seed)
    point = float(diff_fn(np.arange(len(np.asarray(groups)))))
    samples = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        samples.append(diff_fn(idx))
    s = np.asarray(samples, float)
    s = s[~np.isnan(s)]
    if len(s) < 20:
        ci = [float('nan'), float('nan')]
    else:
        lo, hi = np.quantile(s, [(1 - level) / 2, 1 - (1 - level) / 2])
        ci = [float(lo), float(hi)]
    return {'point': point, 'ci': ci, 'lo': ci[0], 'hi': ci[1], 'n_boot': int(len(s)), 'n_groups': int(len(uniq))}


def wilson(k, n, z=1.959963984540054, level=0.95):
    """Wilson score interval for a proportion; returns point and both bounds."""
    if n <= 0:
        return {'k': int(k), 'n': int(n), 'p': None, 'lo': None, 'hi': None, 'level': level}
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return {'k': int(k), 'n': int(n), 'p': float(p), 'lo': float((centre - margin) / (1 + z * z / n)),
            'hi': float((centre + margin) / (1 + z * z / n)), 'level': level}


def _auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float('nan')
    if scores.ndim == 2:
        from sklearn.metrics import roc_auc_score
        present = sorted(np.unique(labels).tolist())
        if scores.shape[1] < len(present):
            return float('nan')
        return float(roc_auc_score(labels, scores, multi_class='ovr', average='macro'))
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def grouped_auc_ci(scores, labels, groups, n=2000, seed=0):
    """AUC (binary or multiclass score matrix) with a parent-group bootstrap interval."""
    scores, labels, groups = np.asarray(scores, float), np.asarray(labels), np.asarray(groups)
    point = _auc(scores, labels)
    boot = paired_bootstrap(lambda idx: _auc(scores[idx], labels[idx]), groups, n=n, seed=seed)
    return {'auc': point, 'ci': boot['ci'], 'n': int(len(labels)), 'n_groups': boot['n_groups'], 'n_boot': boot['n_boot']}


def spearman(x, y):
    """Spearman correlation and two-sided p; NaN with fewer than 3 finite pairs."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float('nan'), float('nan')
    r = spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue)

def bootstrap_groups(groups, metric_fns, B=1000, seed=0):
    """Paired bootstrap over parent groups. metric_fns: name -> f(index_array) -> float."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    out = {n: [] for n in metric_fns}
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        for n, f in metric_fns.items():
            out[n].append(f(idx))
    return {n: np.array(v) for n, v in out.items()}


def interval(samples, level=0.95):
    s = np.asarray(samples)[~np.isnan(np.asarray(samples))]
    if len(s) < 20:
        return [float('nan'), float('nan')]
    lo, hi = np.quantile(s, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


def wilson_lower(successes, n, z=1.96):
    if n == 0:
        return 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (centre - margin) / (1 + z * z / n)
