"""2a / H2: within-material best-of-k selection using hidden-state geometry.

Selectors (higher score = preferred):
  outlier_centered  PRISM-style: after material centering, low mean cosine to the *other* candidates wins;
  outlier_raw       same without centering (the "mechanism removed" ablation);
  balance           closeness of the text↔image projection to the balance value b (mean projection on mixed materials);
  min_shift         the existing P5 intrinsic score (larger minimum modality shift wins);
  random            analytic within-material mean, no sampling.
Main outcome: selected candidate is valid AND label_new.verdict == dual_necessary (falls back to old
non-redundant labels only when no new labels exist; reported separately). Intervals: paired bootstrap over parents.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import candidate_table as CT
from .centering import center, projection
from .records import read_jsonl, write_json
from .stats import paired_bootstrap, wilson
from .selection_eval import min_shift_scores


def outcome(row):
    new = (row.get('label_new') or {}).get('verdict')
    if new:
        success = new == 'dual_necessary'
    else:
        success = row['label_old'].get('redundant') == 0
    valid = row.get('valid') == 1
    return (1.0 if valid else 0.0), (1.0 if valid and success else 0.0), bool(new)


def outlier_scores(rows, X, material, centered=True):
    """Negative mean cosine to the other candidates of the same material (self excluded)."""
    Y = center(X, {'material_id': material}, 'material') if centered else np.asarray(X, float)
    norm = np.linalg.norm(Y, axis=1)
    U = np.divide(Y, norm[:, None], out=np.zeros_like(Y), where=norm[:, None] > 1e-12)
    out = {}
    by = defaultdict(list)
    for i, m in enumerate(material):
        by[m].append(i)
    for m, idx in by.items():
        for i in idx:
            others = [j for j in idx if j != i]
            out[rows[i]['cand_id']] = float(-np.mean(U[i] @ U[others].T)) if others else 0.0
    return out


def balance_scores(rows, X, material, w, mixed_materials):
    proj = projection(X, w)
    b = float(np.mean(proj[np.isin(material, list(mixed_materials))])) if mixed_materials else float(proj.mean())
    return {r['cand_id']: float(-abs(proj[i] - b)) for i, r in enumerate(rows)}, b


def mixed(rows):
    """Materials with both success and non-success outcomes under the label in use."""
    by = defaultdict(list)
    for r in rows:
        if r.get('evaluated', True):
            by[r['material_id']].append(outcome(r)[1])
    return {m for m, v in by.items() if 0 < sum(v) < len(v)}


def compare(rows, scores, *, k=1, B=2000, subset='all'):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get('evaluated', True) and r['cand_id'] in scores and np.isfinite(scores[r['cand_id']]):
            by[r['material_id']].append(i)
    keep = mixed(rows) if subset == 'mixed' else set(by)
    records = []
    for m in sorted(by):
        if m not in keep:
            continue
        idx = by[m]
        s = np.array([scores[rows[i]['cand_id']] for i in idx], float)
        if k > 1:
            order = np.argsort(-s, kind='stable')[:k]
            picked = [idx[j] for j in order]
        else:
            picked = [idx[j] for j in np.where(s == s.max())[0]]
        sel = np.mean([outcome(rows[i])[:2] for i in picked], axis=0)
        rand = np.mean([outcome(rows[i])[:2] for i in idx], axis=0)
        records.append({'material_id': m, 'parent_id': rows[idx[0]].get('parent_id') or m,
                        'sel_valid': float(sel[0]), 'sel_success': float(sel[1]),
                        'rand_valid': float(rand[0]), 'rand_success': float(rand[1]), 'n': len(idx)})
    if len(records) < 5:
        return {'status': 'too_few_materials', 'n_materials': len(records)}
    groups = np.array([r['parent_id'] for r in records])
    d_success = np.array([r['sel_success'] - r['rand_success'] for r in records])
    d_valid = np.array([r['sel_valid'] - r['rand_valid'] for r in records])
    boot_s = paired_bootstrap(lambda idx: float(d_success[idx].mean()), groups, n=B)
    boot_v = paired_bootstrap(lambda idx: float(d_valid[idx].mean()), groups, n=B)
    return {'n_materials': len(records), 'success_selector': float(np.mean([r['sel_success'] for r in records])),
            'success_random': float(np.mean([r['rand_success'] for r in records])),
            'diff_success': boot_s['point'], 'diff_success_ci': boot_s['ci'],
            'valid_selector': float(np.mean([r['sel_valid'] for r in records])),
            'valid_random': float(np.mean([r['rand_valid'] for r in records])),
            'diff_valid': boot_v['point'], 'diff_valid_ci': boot_v['ci'],
            'wilson_success': wilson(round(sum(r['sel_success'] for r in records)), len(records)),
            'g4_pass': bool(boot_s['ci'][0] > 0 and boot_v['point'] >= -0.02)}


def run_select(*, capture_run, label_file, project_root='.', output_root='outputs', run_id=None, layer=22, position='early',
               materials=None, k=1, bootstrap=2000, label_new=None, w_dir=None, dry_run=False):
    root = Path(project_root).resolve()
    base = root / output_root
    rows = CT.build_table(capture_run=capture_run, label_file=label_file, project_root=project_root,
                          output_root=output_root, label_new=label_new or (base / run_id / 'labels_new.jsonl' if run_id and (base / run_id / 'labels_new.jsonl').exists() else None))
    if materials:
        ids = {line.split('#')[0].strip() for line in Path(materials).read_text().splitlines() if line.split('#')[0].strip()}
        rows = [r for r in rows if r['parent_id'] in ids or r['material_id'] in ids]
    if dry_run:
        print(f'[dry-run] select: {len(rows)} candidates, layer={layer}, position={position}, k={k}; no GPU/API used')
    meta = CT.meta(rows)
    X = CT.load_feats(capture_run=capture_run, rows=rows, layers=(layer,), positions=(position,),
                      project_root=project_root, output_root=output_root)[position][:, 0]
    material = meta['material_id']
    mx = mixed(rows)
    wpath = Path(w_dir) if w_dir else (base / run_id / 'w_dir.npy' if run_id else None)
    if wpath is None or not Path(wpath).exists():
        from .diag_curve import direction_axis
        w = direction_axis(rows, X, meta)
        print('[select] w_dir.npy not found; fitted the text-image axis from the current direction labels (fallback)')
    else:
        w = np.load(wpath)
    scores = {
        'outlier_centered': outlier_scores(rows, X, material, centered=True),
        'outlier_raw': outlier_scores(rows, X, material, centered=False),
        'balance': balance_scores(rows, X, material, w, mx)[0],
        'random': {r['cand_id']: 0.0 for r in rows},
    }
    b = balance_scores(rows, X, material, w, mx)[1]
    dist_path = base / capture_run / 'distances.jsonl'
    dist_rows = read_jsonl(dist_path) if dist_path.exists() else []
    if dist_rows and position in dist_rows[0].get('rel_l2', {}).get('image_removed', {}):
        scores['min_shift'] = min_shift_scores([{'candidate_id': r['cand_id']} for r in rows], dist_rows,
                                               layer=layer, position=position)
    results = {name: compare(rows, s, k=k, B=bootstrap) for name, s in scores.items()}
    results['outlier_centered_mixed'] = compare(rows, scores['outlier_centered'], k=k, B=bootstrap, subset='mixed')
    results['balance_mixed'] = compare(rows, scores['balance'], k=k, B=bootstrap, subset='mixed')
    results['random_mixed'] = compare(rows, scores['random'], k=k, B=bootstrap, subset='mixed')
    if dry_run:
        for name, v in results.items():
            if 'status' in v:
                print(f'[dry-run] {name}: {v["status"]}')
            else:
                print(f'[dry-run] {name:22s} success={v["success_selector"]:.3f} random={v["success_random"]:.3f} '
                      f'diff={v["diff_success"]:+.3f} [{v["diff_success_ci"][0]:+.3f},{v["diff_success_ci"][1]:+.3f}]')
        return {'dry_run': True, 'b': b, 'mixed_materials': len(mx), 'results': results}
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    payload = {'capture_run': capture_run, 'label_file': label_file, 'layer': layer, 'position': position, 'k': k,
               'b': b, 'mixed_materials': len(mx), 'results': results,
               'outcome': 'valid and label_new verdict == dual_necessary' if label_new or (out / 'labels_new.jsonl').exists()
               else 'valid and old non-redundant (new labels absent; pipeline check only)'}
    write_json(out / 'selection_results.json', payload)
    lines = ['# 2a selection (H2)', '', f'Candidates: {len(rows)}; layer {layer}, position {position}, k={k}; b={b:.4f}.',
             f'Outcome: {payload["outcome"]}.', '', '| selector | n materials | success | random | diff [95% CI] | valid | diff valid | G4 |', '|---|---|---|---|---|---|---|---|']
    for name, v in results.items():
        if 'status' in v:
            lines.append(f'| {name} | {v.get("n_materials")} | - | - | {v["status"]} | - | - | - |')
        else:
            lines.append(f'| {name} | {v["n_materials"]} | {v["success_selector"]:.3f} | {v["success_random"]:.3f} | '
                         f'{v["diff_success"]:+.3f} [{v["diff_success_ci"][0]:+.3f}, {v["diff_success_ci"][1]:+.3f}] | '
                         f'{v["valid_selector"]:.3f} | {v["diff_valid"]:+.3f} | {v["g4_pass"]} |')
    (out / 'selection_report.md').write_text('\n'.join(lines) + '\n')
    return {'run_id': run_id, 'b': b, 'results': results}
