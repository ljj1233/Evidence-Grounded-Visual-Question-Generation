"""1c / H2–H3: re-train the P3 probes after each centering level (none, global, skill, material).

Reuses `probing.oof` / `score` (fixed hyperparameters, grouped CV) but groups by **parent material** here, and adds
the material-internal AUC (only materials with both classes, averaged). The direction probe at material level is
saved as `w_dir.npy`: the single text↔image axis used by 2a/2b. P5 scalar shifts are re-read from `distances.jsonl`
and centred at the scalar level (recentring the underlying hidden states would need new forward passes; flagged in
the output as `p5_centering`).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import candidate_table as CT
from .centering import center, projection
from .probing import auc, fit_predict, oof, score
from .records import read_jsonl, write_json
from .stats import grouped_auc_ci, paired_bootstrap

TARGETS = ('redundancy', 'direction')


def input_matrix(rows):
    out = []
    for r in rows:
        f = r['features']
        image = f.get('image') or {}
        out.append([f.get('text_chars') or 0, f.get('n_choices') or 0, image.get('width') or 0, image.get('height') or 0,
                    image.get('mean') or 0, image.get('std') or 0, f.get('zh_share') or 0])
    return np.array(out, float)


def lexical_matrix(rows, dim=512, n_tokens=None):
    X = np.zeros((len(rows), dim + 2))
    for i, r in enumerate(rows):
        ids = r['features'].get('generated_token_ids') or []
        ids = ids[:n_tokens] if n_tokens else ids
        for t in ids:
            X[i, t % dim] += 1
        X[i, dim], X[i, dim + 1] = len(ids), r['features'].get('zh_share') or 0
    return X


def stratified_group_folds(groups, y, n_folds, seed):
    """Grouped folds balanced by the group-level label (prevents pooled-AUC artefacts when a label is constant
    inside a material but its prevalence differs across folds)."""
    groups, y = np.asarray(groups), np.asarray(y)
    uniq = np.unique(groups)
    label = {g: float(np.mean(y[groups == g])) for g in uniq}
    rng = np.random.default_rng(seed)
    fold_of = {}
    for stratum in ([g for g in uniq if label[g] >= 0.5], [g for g in uniq if label[g] < 0.5]):
        order = sorted(stratum)
        rng.shuffle(order)
        for i, g in enumerate(order):
            fold_of[g] = i % n_folds
    return np.array([fold_of[g] for g in groups])


def oof_predict(X, y, groups, kind, seeds=(0,), folds=5):
    from .probing import oof
    preds = []
    for s in seeds:
        f = stratified_group_folds(groups, y, folds, s)
        p = oof(X, y, groups, kind, s, n_folds=folds, folds=f)
        if p is not None and not np.all(np.isnan(p)):
            preds.append(p)
    if not preds:
        return None
    stack = np.stack(preds)
    count = np.sum(~np.isnan(stack), axis=0)
    return np.where(count > 0, np.nansum(stack, axis=0) / np.maximum(count, 1), np.nan)


def _metric(y, pred, kind='binary'):
    return score(y, pred, kind)


def material_auc_arrays(preds, y, material):
    """Per-material AUC for each predictor (NaN when the material has one class)."""
    mats = np.unique(material)
    out = {k: np.full(len(mats), np.nan) for k in preds}
    for i, m in enumerate(mats):
        idx = np.where(material == m)[0]
        ok = idx[~np.isnan(preds['probe'][idx]).any(axis=1)] if preds['probe'].ndim > 1 else idx[~np.isnan(preds['probe'][idx])]
        if len(set(y[ok].tolist())) < 2:
            continue
        for k, p in preds.items():
            out[k][i] = auc(y[ok], p[ok])
    return mats, out


def material_bootstrap(mats, arrays, parents, n=2000):
    """Bootstrap over parents at the material level; returns point/CI for each array and paired differences."""
    group = np.array([parents[m] for m in mats])
    stats = {k: paired_bootstrap(lambda idx, a=v: float(np.nanmean(a[idx])), group, n=n) for k, v in arrays.items()}
    diffs = {}
    for k in arrays:
        if k == 'probe':
            continue
        d = arrays['probe'] - arrays[k]
        diffs[k] = paired_bootstrap(lambda idx, d=d: float(np.nanmean(d[idx])), group, n=n)
    return stats, diffs


def _paired_diff(pred_a, pred_b, y, groups, kind, n=2000):
    def diff(idx):
        return _metric(y[idx], pred_a[idx], kind) - _metric(y[idx], pred_b[idx], kind)
    return paired_bootstrap(diff, groups, n=n)



def p5_scalars(capture_run, rows, *, position, layer, project_root='.', output_root='outputs'):
    """Per-candidate asymmetry and min_shift from stored distances; None if the run has no distances.jsonl."""
    path = Path(project_root).resolve() / output_root / capture_run / 'distances.jsonl'
    if not path.exists():
        return None
    dist = {r['candidate_id']: r for r in read_jsonl(path)}
    from .modality_distance import modality_scores
    asym, mini = [], []
    for r in rows:
        d = dist.get(r['cand_id'])
        if (not d or 'image_removed' not in d['rel_l2'] or 'text_removed' not in d['rel_l2']
                or position not in d['rel_l2']['image_removed'] or position not in d['rel_l2']['text_removed']):
            asym.append(np.nan); mini.append(np.nan); continue
        s = modality_scores(d, scheme_pairs=(('image_removed', 'text_removed'),), position=position, layer=layer)
        v = s['image_removed|text_removed']
        asym.append(v['asymmetry']); mini.append(v['min_shift'])
    return {'p5_asymmetry': np.array(asym), 'p5_min_shift': np.array(mini)}


def _center_scalar(x, meta, level):
    keys = {'none': None, 'global': np.zeros(len(x), int), 'skill': meta['skill'], 'material': meta['material_id']}[level]
    if keys is None:
        return x
    out = x.copy()
    for g in np.unique(keys):
        idx = np.where(keys == g)[0]
        ok = ~np.isnan(x[idx])
        if ok.sum() > 1:
            out[idx] -= np.nanmean(x[idx])
    return out


def _direction_weights(Xc, y, material, rows):
    """Logistic direction on all material-level rows (standardised fit), returned in raw feature space."""
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xc.mean(0), Xc.std(0)
    sd[sd < 1e-6] = 1.0
    model = LogisticRegression(C=0.01, max_iter=300).fit((Xc - mu) / sd, y)
    w = model.coef_[0] / sd
    proj = projection(Xc, w)
    mixed = [m for m in np.unique(material) if len(set(y[material == m].tolist())) == 2]
    b = float(np.mean(proj[np.isin(material, mixed)])) if mixed else float(proj.mean())
    groups = {}
    for name, val in (('image', 1), ('text', 0)):
        sel = proj[y == val]
        groups[name] = {k: float(v) for k, v in zip(('mean', 'std', 'min', 'max'), (sel.mean(), sel.std(), sel.min(), sel.max()))}
    return w, {'orientation': 'higher projection = image', 'balance_b': b,
               'mixed_materials': len(mixed), 'projection': {'overall_mean': float(proj.mean()), 'overall_std': float(proj.std()),
                                                             'quantiles': [float(v) for v in np.quantile(proj, [0.05, 0.25, 0.5, 0.75, 0.95])],
                                                             'by_direction': groups}}


def direction_axis(rows, X, meta):
    """Material-centred text↔image direction fitted on direction labels; fallback when diag-curve was not run."""
    idx, y = CT.split_targets(rows, 'direction')
    if len(idx) < 20 or len(set(y.tolist())) < 2:
        raise ValueError('not enough direction-labelled candidates to fit the text-image axis')
    Xc = center(X, meta, 'material')[idx]
    w, _ = _direction_weights(Xc, y, meta['material_id'][idx], None)
    return w / (np.linalg.norm(w) or 1.0)


def run_diag_curve(*, capture_run, label_file, layers=(22,), positions=('early', 'question_end'), levels=('none', 'global', 'skill', 'material'),
                   materials=None, project_root='.', output_root='outputs', run_id=None, bootstrap=2000, seeds=(0,), dry_run=False):
    base = Path(project_root).resolve() / output_root
    rows = CT.build_table(capture_run=capture_run, label_file=label_file, project_root=project_root, output_root=output_root)
    if materials:
        ids = {line.strip() for line in Path(materials).read_text().splitlines() if line.strip()}
        rows = [r for r in rows if r['parent_id'] in ids or r['material_id'] in ids]
    if dry_run:
        layers, positions = tuple(layers)[:1], tuple(positions)[:1]
        bootstrap, seeds = min(bootstrap, 200), tuple(seeds)[:1]
        print(f'[dry-run] diag-curve: {len(rows)} candidates, layers={list(layers)}, positions={list(positions)}, '
              f'levels={list(levels)}, bootstrap={bootstrap}, seeds={list(seeds)}; no GPU/API used')
    meta = CT.meta(rows)
    feats = CT.load_feats(capture_run=capture_run, rows=rows, layers=layers, positions=positions,
                          project_root=project_root, output_root=output_root)
    targets = {t: CT.split_targets(rows, t) for t in TARGETS}
    base_input = input_matrix(rows)
    base_lex = lexical_matrix(rows)
    entries = []
    w_dir = None
    for pos in positions:
        for li, layer in enumerate(layers):
            p5 = p5_scalars(capture_run, rows, position=pos, layer=layer, project_root=project_root, output_root=output_root)
            for level in levels:
                Xc = center(feats[pos][:, li], meta, level)
                for target in TARGETS:
                    idx, y = targets[target]
                    if len(idx) < 20 or len(set(y.tolist())) < 2:
                        continue
                    X, yi, groups, material = Xc[idx], y, meta['parent_id'][idx], meta['material_id'][idx]
                    kind = 'binary'
                    pred = oof_predict(X, yi, groups, kind, seeds)
                    b_in = oof_predict(base_input[idx], yi, groups, kind, seeds)
                    b_lex = oof_predict(base_lex[idx], yi, groups, kind, seeds)
                    if pred is None:
                        continue
                    base_preds = {'input_features': b_in, 'lexical_surface': b_lex}
                    base_preds = {k: v for k, v in base_preds.items() if v is not None}
                    point = {k: float(_metric(yi, v, kind)) for k, v in {'probe': pred, **base_preds}.items()}
                    strongest = max(base_preds, key=lambda k: point[k])
                    diff = _paired_diff(pred, base_preds[strongest], yi, groups, kind, n=bootstrap)
                    per_base = {k: _paired_diff(pred, v, yi, groups, kind, n=bootstrap) for k, v in base_preds.items()}
                    ci = paired_bootstrap(lambda ii: _metric(yi[ii], pred[ii], kind), groups, n=bootstrap)
                    preds = {'probe': pred, **base_preds}
                    mats, arrays = material_auc_arrays(preds, yi, material)
                    parents = {m: groups[np.where(material == m)[0][0]] for m in mats}
                    mstats, mdiffs = material_bootstrap(mats, arrays, parents, n=bootstrap)
                    entry = {'position': pos, 'layer': layer, 'level': level, 'target': target, 'n': int(len(yi)),
                             'n_groups': int(len(set(groups.tolist()))), 'probe_auc': point['probe'], 'probe_ci': ci['ci'],
                             'baselines': {k: float(v) for k, v in point.items() if k != 'probe'},
                             'diff_vs_baselines': {k: {'point': v['point'], 'ci': v['ci']} for k, v in per_base.items()},
                             'strongest_baseline': strongest, 'diff_vs_baseline': diff['point'], 'diff_ci': diff['ci'],
                             'material_auc': mstats['probe']['point'], 'material_auc_ci': mstats['probe']['ci'],
                             'baselines_material_auc': {k: mstats[k]['point'] for k in base_preds},
                             'material_diff_vs_baselines': {k: {'point': v['point'], 'ci': v['ci']} for k, v in mdiffs.items()}}
                    if p5:
                        for name, arr in p5.items():
                            a = _center_scalar(arr[idx], {k: v[idx] for k, v in meta.items()}, level)
                            ok = ~np.isnan(a)
                            if ok.sum() > 20 and len(set(yi[ok].tolist())) == 2:
                                g = grouped_auc_ci(a[ok], yi[ok], groups[ok], n=bootstrap)
                                entry[f'{name}_auc'] = g['auc']; entry[f'{name}_ci'] = g['ci']
                    entries.append(entry)
                    print(f'[diag-curve] {pos} L{layer} {level:8s} {target:10s} probe={point["probe"]:.3f} '
                          f'{strongest}={point[strongest]:.3f} diff={diff["point"]:+.3f} '
                          f'[{diff["ci"][0]:+.3f},{diff["ci"][1]:+.3f}] matAUC={mstats["probe"]["point"]:.3f}')
                    if target == 'direction' and level == 'material' and w_dir is None:
                        w, info = _direction_weights(X, yi, material, rows)
                        w_dir = {'weights': w, 'info': {**info, 'layer': layer, 'position': pos}}
    if dry_run:
        return {'dry_run': True, 'entries': len(entries)}
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'diag_curve.json', {'capture_run': capture_run, 'entries': entries,
                                         'p5_centering': 'scalar readings centred by group; hidden states not re-forwarded'})
    fields = ['position', 'layer', 'level', 'target', 'n', 'n_groups', 'probe_auc', 'probe_ci_lo', 'probe_ci_hi',
              'strongest_baseline', 'baseline_auc', 'diff_vs_baseline', 'diff_ci_lo', 'diff_ci_hi',
              'material_auc', 'material_auc_ci_lo', 'material_auc_ci_hi', 'material_baseline_auc',
              'material_diff_vs_baseline', 'material_diff_ci_lo', 'material_diff_ci_hi',
              'p5_asymmetry_auc', 'p5_min_shift_auc']
    with (out / 'diag_curve.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(fields)
        for e in entries:
            strong = e['strongest_baseline']
            md = e['material_diff_vs_baselines'][strong]
            w.writerow([e['position'], e['layer'], e['level'], e['target'], e['n'], e['n_groups'], e['probe_auc'],
                        *e['probe_ci'], strong, e['baselines'][strong],
                        e['diff_vs_baseline'], *e['diff_ci'], e['material_auc'], *e['material_auc_ci'],
                        e['baselines_material_auc'][strong], md['point'], *md['ci'],
                        e.get('p5_asymmetry_auc', float('nan')), e.get('p5_min_shift_auc', float('nan'))])
    if w_dir is not None:
        np.save(out / 'w_dir.npy', w_dir['weights'])
        write_json(out / 'w_dir.json', {'layer': w_dir['info']['layer'], 'position': w_dir['info']['position'],
                                        'level': 'material', 'target': 'direction', **w_dir['info']})
    return {'run_id': run_id, 'entries': len(entries), 'w_dir': str(out / 'w_dir.npy') if w_dir is not None else None}
