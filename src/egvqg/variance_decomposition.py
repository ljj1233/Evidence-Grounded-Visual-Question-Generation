"""How much of the candidate-level label variance is decided by the context (skill, material) and how much is left inside one material?

Nested unbalanced ANOVA (skill > material > candidate), Searle moment estimators of the three variance components (negative
estimates clipped to 0), plus a leave-one-material-out check: does the mean of the *other* materials of the same skill predict a
material's mean better than the grand mean? Uncertainty: bootstrap over skills. CPU only, labels only.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .records import read_jsonl


def components(y, skills, materials):
    """Variance components {'skill','material','candidate'} and shares for a binary/real outcome with nested labels."""
    y = np.asarray(y, float)
    by = defaultdict(lambda: defaultdict(list))
    for v, s, m in zip(y, skills, materials):
        by[s][m].append(v)
    a = len(by)
    mats = [(s, m, np.array(v)) for s, ms in by.items() for m, v in ms.items()]
    M, N = len(mats), len(y)
    if a < 2 or M - a < 1 or N - M < 1:
        return None
    n_ij = np.array([len(v) for _, _, v in mats], float)
    mean_ij = np.array([v.mean() for _, _, v in mats])
    skill_of = [s for s, _, _ in mats]
    n_i = {s: sum(n for n, k in zip(n_ij, skill_of) if k == s) for s in by}
    mean_i = {s: sum(n * mu for n, mu, k in zip(n_ij, mean_ij, skill_of) if k == s) / n_i[s] for s in by}
    grand = y.mean()
    ss_w = sum(((v - v.mean()) ** 2).sum() for _, _, v in mats)
    ss_mat = sum(n * (mu - mean_i[k]) ** 2 for n, mu, k in zip(n_ij, mean_ij, skill_of))
    ss_skill = sum(n_i[s] * (mean_i[s] - grand) ** 2 for s in by)
    ms_w, ms_mat, ms_skill = ss_w / (N - M), ss_mat / (M - a), ss_skill / (a - 1)
    sq_over = {s: sum(n ** 2 for n, k in zip(n_ij, skill_of) if k == s) / n_i[s] for s in by}
    k1 = (N - sum(sq_over.values())) / (M - a)
    k2 = (sum(sq_over.values()) - (n_ij ** 2).sum() / N) / (a - 1)
    k3 = (N - sum(v ** 2 for v in n_i.values()) / N) / (a - 1)
    s_mat = max(0.0, (ms_mat - ms_w) / k1)
    s_skill = max(0.0, (ms_skill - ms_w - k2 * s_mat) / k3)
    parts = {'skill': s_skill, 'material': s_mat, 'candidate': ms_w}
    total = sum(parts.values())
    return {'variance': parts, 'share': {k: (v / total if total > 0 else float('nan')) for k, v in parts.items()},
            'n_skills': a, 'n_materials': M, 'n_candidates': N}


def loo_material_r2(y, skills, materials, min_candidates=4):
    """Out-of-sample R2 of 'skill mean of the other materials' vs 'grand mean of the other materials' for material means."""
    by = defaultdict(lambda: defaultdict(list))
    for v, s, m in zip(y, skills, materials):
        by[s][m].append(v)
    rows = [(s, m, float(np.mean(v))) for s, ms in by.items() for m, v in ms.items() if len(v) >= min_candidates]
    sse_skill = sse_grand = 0.0
    used = 0
    for s, m, mu in rows:
        peers = [x for s2, m2, x in rows if s2 == s and m2 != m]
        others = [x for s2, m2, x in rows if m2 != m]
        if not peers:
            continue
        sse_skill += (mu - np.mean(peers)) ** 2
        sse_grand += (mu - np.mean(others)) ** 2
        used += 1
    return {'n_materials_used': used, 'r2_oos': float('nan') if sse_grand == 0 else 1 - sse_skill / sse_grand}


def bootstrap_skills(y, skills, materials, fn, B=500, seed=0):
    rng = np.random.default_rng(seed)
    y, skills, materials = np.asarray(y, float), np.asarray(skills), np.asarray(materials)
    uniq = sorted(set(skills.tolist()))
    idx_of = {s: np.where(skills == s)[0] for s in uniq}
    out = []
    for b in range(B):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[uniq[i]] for i in pick])
        # a resampled skill appearing twice must stay a distinct group
        sk = np.concatenate([[f'{uniq[i]}#{n}'] * len(idx_of[uniq[i]]) for n, i in enumerate(pick)])
        mt = np.concatenate([[f'{uniq[i]}#{n}|{m}' for m in materials[idx_of[uniq[i]]]] for n, i in enumerate(pick)])
        v = fn(y[idx], sk, mt)
        if v is not None and not np.isnan(v):
            out.append(v)
    return out


def outcomes_from_labels(rows):
    """Binary candidate-level outcomes from probe_labels rows."""
    sets = {
        'redundant (valid, decided)': [(r, int(r['redundant'])) for r in rows if r['valid'] == 1 and r['redundant'] is not None],
        'image_not_text (redundant, one direction)': [(r, int(r['direction'] == 'image')) for r in rows if r['redundant'] == 1 and r['direction'] in ('image', 'text')],
        'valid (all candidates)': [(r, int(r['valid'] == 1)) for r in rows if r['valid'] is not None],
    }
    return sets


def summarize(label_path, B=500):
    rows = read_jsonl(label_path)
    out = {}
    for name, items in outcomes_from_labels(rows).items():
        y = np.array([v for _, v in items])
        sk = [r['skill'] for r, _ in items]
        mt = [r['material_id'] for r, _ in items]
        comp = components(y, sk, mt)
        if comp is None:
            out[name] = {'status': 'too_few'}
            continue
        boots = {k: [] for k in ('skill', 'material', 'candidate')}
        for part in boots:
            samples = bootstrap_skills(y, sk, mt, lambda yy, s, m, part=part: (components(yy, s, m) or {'share': {part: np.nan}})['share'][part], B=B)
            boots[part] = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))] if len(samples) > 20 else [float('nan')] * 2
        loo = loo_material_r2(y, sk, mt)
        loo_boot = bootstrap_skills(y, sk, mt, lambda yy, s, m: loo_material_r2(yy, s, m)['r2_oos'], B=B)
        out[name] = {**comp, 'share_ci': boots, 'prevalence': float(y.mean()), 'loo_skill_vs_grand': {**loo, 'ci': [float(np.quantile(loo_boot, .025)), float(np.quantile(loo_boot, .975))] if len(loo_boot) > 20 else None}}
    return out
