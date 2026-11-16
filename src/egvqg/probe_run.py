"""P3 protocol: evaluate one probe task against baselines and a control, and run discovery (layer choice) -> confirmation (G2).

Pre-registered in docs/implementation_plan.md 7.7: layers are chosen on the pilot only; G2 is judged on the confirmation
materials with that layer fixed; the reference baseline is the stronger of the input-feature and lexical-surface baselines.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .probing import (bootstrap_groups, control_labels, direction_codes, input_features, interval, lexical_features, load_candidates,
                      metric_for, oof, score, to_control_binary)
from .records import write_json

TASK_POSITIONS = {'redundancy': ('early', 'question_end'), 'direction': ('early', 'question_end'), 'propensity': ('prompt_end',)}


def evaluate_task(X, y, skills, materials, kind, baselines, seeds=(0, 1, 2), B=1000):
    """Probe vs baselines vs control for one (task, position, layer). `baselines`: name -> feature matrix. Intervals bootstrap materials."""
    y = np.asarray(y)
    features = {'probe': X, **baselines}
    per_seed = {n: [] for n in features}
    ctrl_y, ctrl_pred = [], []
    for s in seeds:
        for n, M in features.items():
            per_seed[n].append(oof(M, y, skills, kind, s))
        cy = control_labels(materials, y, s) if kind == 'reg' else to_control_binary(materials, y, s)
        ctrl_y.append(cy)
        ctrl_pred.append(oof(X, cy, skills, kind, s))
    ens = {n: np.nanmean(np.stack(v), 0) for n, v in per_seed.items()}
    seed_scores = {n: [float(score(y, p, kind)) for p in v] for n, v in per_seed.items()}
    ctrl_scores = [float(score(cy, cp, kind)) for cy, cp in zip(ctrl_y, ctrl_pred)]
    names = list(baselines)
    point = {n: float(score(y, ens[n], kind)) for n in ens}
    strongest = max(names, key=lambda n: point[n]) if names else None
    metric = metric_for(kind)

    def sub(pred, yy, idx):
        p = pred[idx]
        ok = ~np.isnan(p).any(axis=tuple(range(1, p.ndim))) if p.ndim > 1 else ~np.isnan(p)
        return metric(yy[idx][ok], p[ok])

    fns = {n: (lambda idx, n=n: sub(ens[n], y, idx)) for n in ens}
    fns['control'] = lambda idx: float(np.nanmean([sub(cp, cy, idx) for cp, cy in zip(ctrl_pred, ctrl_y)]))
    boot = bootstrap_groups(materials, fns, B=B)
    selectivity = boot['probe'] - boot['control']
    diff = boot['probe'] - boot[strongest] if strongest else None
    return {'n': int(len(y)), 'n_groups': len(set(materials)), 'kind': kind, 'point': point, 'control_point': float(np.mean(ctrl_scores)),
            'ci': {n: interval(boot[n]) for n in boot}, 'strongest_baseline': strongest,
            'diff_vs_strongest_baseline': {'point': point['probe'] - point[strongest] if strongest else None, 'ci': interval(diff) if strongest else None},
            'selectivity': {'point': point['probe'] - float(np.mean(ctrl_scores)), 'ci': interval(selectivity)},
            'seed_scores': seed_scores, 'seed_control_scores': ctrl_scores,
            'seeds_consistent': bool(strongest) and all(p > b and p > c for p, b, c in zip(seed_scores['probe'], seed_scores[strongest], ctrl_scores))}


def g2_verdict(result):
    """(i) probe minus strongest baseline: CI excludes 0; (ii) selectivity CI excludes 0; (iii) every seed beats baseline and control."""
    d, s = result['diff_vs_strongest_baseline']['ci'], result['selectivity']['ci']
    i = bool(d) and not np.isnan(d[0]) and d[0] > 0
    ii = not np.isnan(s[0]) and s[0] > 0
    iii = bool(result['seeds_consistent'])
    return {'i_beats_baseline': bool(i), 'ii_selectivity_positive': bool(ii), 'iii_seeds_agree': iii, 'passed': bool(i and ii and iii)}


def build_tasks(rows, min_decided=4):
    """Rows, targets and kinds of the three tasks. Redundancy/direction are candidate-level; propensity is material-level."""
    decided = [i for i, r in enumerate(rows) if r['valid'] == 1 and r['redundant'] is not None]
    dirs = direction_codes(rows)
    d_idx = [i for i in decided if dirs[i] >= 0 and rows[i]['redundant'] == 1]
    by = {}
    for i in decided:
        by.setdefault(rows[i]['material_id'], []).append(i)
    mats = [m for m, ix in sorted(by.items()) if len(ix) >= min_decided]
    first = [by[m][0] for m in mats]
    return {'redundancy': {'idx': decided, 'y': np.array([rows[i]['redundant'] for i in decided]), 'kind': 'binary'},
            'direction': {'idx': d_idx, 'y': dirs[d_idx], 'kind': 'multiclass'},
            'propensity': {'idx': first, 'y': np.array([np.mean([rows[i]['redundant'] for i in by[m]]) for m in mats]), 'kind': 'reg'}}


def task_baselines(task, position, rows, idx):
    sub = [rows[i] for i in idx]
    out = {'input_features': input_features(sub, with_language=task != 'propensity')}
    if task != 'propensity':
        out['lexical_surface'] = lexical_features(sub, n_tokens=4 if position == 'early' else None)
    return out


def curve(spec, rows, X_all, seeds=(0, 1, 2)):
    """Layer curve: mean metric over seeds, skills held out. X_all is [n, layers, H]."""
    idx = spec['idx']
    X = X_all[idx]
    skills = [rows[i]['skill'] for i in idx]
    return [float(np.mean([score(spec['y'], oof(X[:, layer], spec['y'], skills, spec['kind'], s), spec['kind']) for s in seeds]))
            for layer in range(X.shape[1])]


def _usable(spec):
    return len(spec['idx']) >= 20 and len(set(spec['y'].tolist())) >= 2


def run_probes(*, run_id, discovery, confirmation, project_root='.', output_root='outputs', B=1000):
    """discovery / confirmation: (capture_runs, label_files). Layers are chosen on discovery only; G2 is judged on confirmation."""
    out = Path(project_root).resolve() / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    arrays = ('prompt_end', 'early', 'question_end')
    d_rows, *d_arr = load_candidates(*discovery, project_root=project_root, output_root=output_root)
    d_arr = dict(zip(arrays, d_arr))
    chosen, curves = {}, {}
    for task, spec in build_tasks(d_rows).items():
        for pos in TASK_POSITIONS[task]:
            if _usable(spec):
                curves[f'{task}/{pos}'] = curve(spec, d_rows, d_arr[pos])
                chosen[f'{task}/{pos}'] = int(np.nanargmax(curves[f'{task}/{pos}']))
    write_json(out / 'discovery.json', {'curves': curves, 'chosen_layer': chosen, 'n_candidates': len(d_rows)})
    c_rows, *c_arr = load_candidates(*confirmation, project_root=project_root, output_root=output_root)
    c_arr = dict(zip(arrays, c_arr))
    tasks, results = build_tasks(c_rows), {}
    for key, layer in chosen.items():
        task, pos = key.split('/')
        spec = tasks[task]
        if not _usable(spec):
            results[key] = {'layer': layer, 'status': 'insufficient_confirmation_data', 'n': len(spec['idx'])}
            continue
        idx = spec['idx']
        res = evaluate_task(c_arr[pos][idx][:, layer], spec['y'], [c_rows[i]['skill'] for i in idx], [c_rows[i]['material_id'] for i in idx],
                            spec['kind'], task_baselines(task, pos, c_rows, idx), B=B)
        results[key] = {'layer': layer, **res, 'g2': g2_verdict(res)}
    write_json(out / 'confirmation.json', results)
    return {'chosen_layer': chosen, 'confirmation': {k: v.get('g2') or v.get('status') for k, v in results.items()}}
