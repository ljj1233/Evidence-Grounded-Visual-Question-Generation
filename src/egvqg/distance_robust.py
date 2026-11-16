"""P5 robustness R1/R2 (pre-registered, plan 7.7): is the modality-shift signal more than template identity or input surface features?

R1  within skill: subtract the per-skill mean of the score, then AUC on the residuals (skills with both classes only).
R2  baseline-adjusted: logistic regression on [skill one-hot + input features A] vs the same plus the shift readouts, material-grouped CV,
    paired bootstrap over materials of the AUC difference.
Both are reported with the number of contributing skills / materials / positives; below the minimum they are 'not evaluable'.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .distance_analysis import PAIRS, feature, join, targets
from .probing import auc, bootstrap_groups, grouped_folds, input_features, interval
from .records import read_jsonl, write_json

MIN_SKILLS, MIN_POSITIVES = 3, 20


def residualize(x, skills):
    x = np.asarray(x, float)
    out = x.copy()
    for s in set(skills):
        idx = [i for i, k in enumerate(skills) if k == s]
        out[idx] = x[idx] - np.nanmean(x[idx])
    return out


def usable_skills(y, skills):
    """Skills whose candidates contain both classes."""
    by = defaultdict(set)
    for v, s in zip(y, skills):
        by[s].add(int(v))
    return {s for s, v in by.items() if len(v) == 2}


def r1_within_skill(x, y, skills, materials, B=1000):
    y, x = np.asarray(y), np.asarray(x, float)
    keep_skills = usable_skills(y, skills)
    keep = np.array([s in keep_skills for s in skills]) & ~np.isnan(x)
    n_pos = int(y[keep].sum()) if keep.any() else 0
    info = {'skills': len(keep_skills), 'materials': len(set(np.asarray(materials)[keep])), 'positives': n_pos, 'n': int(keep.sum())}
    if len(keep_skills) < MIN_SKILLS or n_pos < MIN_POSITIVES:
        return {**info, 'status': 'not_evaluable'}
    sk = [s for s, k in zip(skills, keep) if k]
    res = residualize(x[keep], sk)
    yy, mm = y[keep], np.asarray(materials)[keep]
    boot = bootstrap_groups(mm, {'a': lambda idx: auc(yy[idx], res[idx])}, B=B)
    return {**info, 'status': 'ok', 'auc': auc(yy, res), 'ci': interval(boot['a'])}


def _design(skill_levels, skills, A, shifts=None):
    onehot = np.array([[float(s == k) for k in skill_levels] for s in skills])
    mu, sd = A.mean(0), A.std(0)
    sd[sd < 1e-9] = 1.0
    cols = [onehot, (A - mu) / sd]
    if shifts is not None:
        m2, s2 = shifts.mean(0), shifts.std(0)
        s2[s2 < 1e-9] = 1.0
        cols.append((shifts - m2) / s2)
    return np.hstack(cols)


def r2_adjusted(shifts, A, y, skills, materials, B=1000, seeds=(0, 1, 2)):
    """AUC gain of adding `shifts` [n,k] to [skill one-hot + A] under material-grouped 5-fold CV."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y)
    ok = ~np.isnan(shifts).any(1)
    y, shifts, A = y[ok], shifts[ok], A[ok]
    skills, materials = [s for s, k in zip(skills, ok) if k], [m for m, k in zip(materials, ok) if k]
    info = {'n': int(len(y)), 'positives': int(y.sum()), 'materials': len(set(materials)), 'skills': len(set(skills))}
    if info['positives'] < MIN_POSITIVES or len(set(y.tolist())) < 2:
        return {**info, 'status': 'not_evaluable'}
    levels = sorted(set(skills))
    X0, X1 = _design(levels, skills, A), _design(levels, skills, A, shifts)
    preds = {0: [], 1: []}
    for seed in seeds:
        folds = grouped_folds(materials, 5, seed)
        for key, X in ((0, X0), (1, X1)):
            out = np.full(len(y), np.nan)
            for f in range(5):
                te = folds == f
                if te.sum() == 0 or len(set(y[~te].tolist())) < 2:
                    continue
                out[te] = LogisticRegression(C=1.0, max_iter=500).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
            preds[key].append(out)
    p0, p1 = np.nanmean(np.stack(preds[0]), 0), np.nanmean(np.stack(preds[1]), 0)
    boot = bootstrap_groups(np.array(materials), {'d': lambda idx: auc(y[idx], p1[idx]) - auc(y[idx], p0[idx])}, B=B)
    return {**info, 'status': 'ok', 'auc_base': auc(y, p0), 'auc_with_shift': auc(y, p1), 'gain': auc(y, p1) - auc(y, p0), 'gain_ci': interval(boot['d'])}


def attach_capture_fields(label_rows, capture_rows):
    cap = {c['candidate_id']: c for c in capture_rows}
    out = []
    for r in label_rows:
        c = cap[r['candidate_id']]
        out.append({**r, 'text_chars': c['text_chars'], 'n_choices': c.get('n_choices'), 'image': c['image'], 'zh_share': c['zh_share']})
    return out


def run_robust(*, run_id, sets, layers, project_root='.', output_root='outputs', B=1000):
    """sets: [(capture_run, label_file)]; layers: {'target/position/metric': layer} chosen on discovery (from p5_readout.json)."""
    base = Path(project_root).resolve() / output_root
    pairs = []
    for run, label_file in sets:
        labels = attach_capture_fields(read_jsonl(base / run / label_file), read_jsonl(base / run / 'captures.jsonl'))
        pairs += join(labels, read_jsonl(base / run / 'distances.jsonl'))
    result = {}
    for tname, (items, y_of) in targets(pairs).items():
        y = np.array([y_of(r) for r, _ in items])
        skills = [r['skill'] for r, _ in items]
        mats = [r['material_id'] for r, _ in items]
        A = input_features([r for r, _ in items], with_language=True)
        for position in ('prompt_end', 'question_end'):
            for metric in ('asymmetry', 'min_shift'):
                key = f'{tname}/{position}/{metric}'
                if key not in layers or layers[key] is None:
                    continue
                layer = layers[key]
                entry = {'layer': layer, 'R1': {}, 'R2': None}
                for pair in PAIRS:
                    x = np.array([feature(d, pair, position, metric, layer) for _, d in items])
                    entry['R1'][f'{pair[0]}|{pair[1]}'] = r1_within_skill(x, y, skills, mats, B)
                shifts = np.array([[feature(d, p, position, metric, layer) for p in PAIRS] for _, d in items])
                entry['R2'] = r2_adjusted(shifts, A, y, skills, mats, B)
                result[key] = entry
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'p5_robust.json', result)
    return {k: {'R1_ok': [v['status'] == 'ok' and v['ci'][0] > 0.5 or v.get('status') == 'ok' and v['ci'][1] < 0.5 for v in e['R1'].values()],
                'R2_gain_ci': e['R2'].get('gain_ci') if e['R2'] else None} for k, e in result.items()}
