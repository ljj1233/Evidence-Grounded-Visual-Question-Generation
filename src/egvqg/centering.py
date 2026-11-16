"""Centering levels used by 1b/1c/2a: none, global, skill, material.

Means are computed from features only (never labels), so centering cannot leak label information. `pearson_rows`
is the PRISM-style row-internal centering kept as a contrast: each candidate's vector is centred across its own
coordinates and then compared by cosine.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def keys_for(meta, level):
    """Group key per row for a centering level. `meta` is a list of dicts or a dict of equal-length arrays."""
    n = len(meta['material_id']) if isinstance(meta, dict) else len(meta)
    get = (lambda i, k: meta[k][i]) if isinstance(meta, dict) else (lambda i, k: meta[i][k])
    if level == 'none':
        return None
    if level == 'global':
        return np.zeros(n, dtype=int)
    if level == 'skill':
        return np.array([get(i, 'skill') for i in range(n)])
    if level == 'material':
        return np.array([get(i, 'material_id') for i in range(n)])
    raise ValueError(f'unknown centering level: {level}')


def center(X, meta, level):
    """Subtract the global / same-skill / same-material mean. Material level implies removing the skill mean too."""
    X = np.asarray(X, float)
    keys = keys_for(meta, level)
    if keys is None:
        return X.copy()
    out = X.copy()
    for g in np.unique(keys):
        idx = np.where(keys == g)[0]
        out[idx] -= X[idx].mean(0)
    return out


def group_means(X, keys):
    """{key: mean vector} and the count per key."""
    X = np.asarray(X, float)
    out, counts = {}, {}
    for g in np.unique(keys):
        idx = np.where(keys == g)[0]
        out[g] = X[idx].mean(0)
        counts[g] = len(idx)
    return out, counts


def pearson_rows(X):
    """Pairwise cosine of row-internally centred vectors (PRISM-style), with the self-similarity set to NaN."""
    X = np.asarray(X, float)
    Y = X - X.mean(1, keepdims=True)
    norm = np.linalg.norm(Y, axis=1)
    Y = np.divide(Y, norm[:, None], out=np.zeros_like(Y), where=norm[:, None] > 1e-12)
    G = Y @ Y.T
    np.fill_diagonal(G, np.nan)
    return G


def within_material_cosine(X, material):
    """Mean pairwise cosine (self-pairs excluded) over all candidate pairs inside the same material."""
    X = np.asarray(X, float)
    norm = np.linalg.norm(X, axis=1)
    U = np.divide(X, norm[:, None], out=np.zeros_like(X), where=norm[:, None] > 1e-12)
    G = U @ U.T
    material = np.asarray(material)
    vals = [G[i, j] for i in range(len(X)) for j in range(i + 1, len(X)) if material[i] == material[j]]
    return float(np.mean(vals)) if vals else float('nan')


def projection(X, w):
    """Scalar projection of every row on a direction (no normalization of w)."""
    return np.asarray(X, float) @ np.asarray(w, float)


def balance_value(X, w, meta, mixed_materials):
    """b = mean projection on the mixed-material subset (materials with both label classes); global mean if empty."""
    proj = projection(X, w)
    material = np.asarray([m['material_id'] for m in meta] if isinstance(meta, list) else meta['material_id'])
    idx = np.isin(material, list(mixed_materials))
    return float(proj[idx].mean()) if idx.any() else float(proj.mean())
