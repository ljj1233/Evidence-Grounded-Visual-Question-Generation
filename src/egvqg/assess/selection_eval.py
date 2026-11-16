"""P6 best-of-k selection evaluated on already-labelled candidates (no new API calls).

Per material the selector picks one candidate among all of its candidates; the outcome is 'success' = valid and not redundant
(cascade-label scope, not "verified dual-necessary"). The random selector is the within-material mean outcome. Ties are averaged.
Differences are paired over materials with a bootstrap over materials (pre-registered G4: selector minus random, 95% interval
excludes 0, and validity of the picked candidate not more than 2 points below random).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..core.stats import bootstrap_groups, interval


def outcomes(row):
    valid = 1.0 if row['valid'] == 1 else 0.0
    success = 1.0 if row['valid'] == 1 and row['redundant'] == 0 else 0.0
    return valid, success


def per_material(rows, scores):
    """{material: (valid_pick, success_pick, valid_random, success_random)} for one selector; `scores[candidate_id]` higher = preferred."""
    by = defaultdict(list)
    for r in rows:
        if r['candidate_id'] in scores and r.get('evaluated', True):
            by[r['material_id']].append(r)
    out = {}
    for m, rs in by.items():
        s = np.array([scores[r['candidate_id']] for r in rs], float)
        if np.isnan(s).any():
            continue
        picked = [r for r, v in zip(rs, s) if v == s.max()]
        vp = float(np.mean([outcomes(r)[0] for r in picked]))
        sp = float(np.mean([outcomes(r)[1] for r in picked]))
        out[m] = (vp, sp, float(np.mean([outcomes(r)[0] for r in rs])), float(np.mean([outcomes(r)[1] for r in rs])))
    return out


def compare(rows, scores, *, B=1000, subset='all'):
    """Selector vs random on the same materials. subset 'mixed' keeps materials with both successes and non-successes."""
    pm = per_material(rows, scores)
    if subset == 'mixed':
        by = defaultdict(list)
        for r in rows:
            by[r['material_id']].append(outcomes(r)[1])
        pm = {m: v for m, v in pm.items() if 0 < sum(by[m]) < len(by[m])}
    mats = sorted(pm)
    if len(mats) < 5:
        return {'n_materials': len(mats), 'status': 'too_few_materials'}
    arr = np.array([pm[m] for m in mats])
    d_success, d_valid = arr[:, 1] - arr[:, 3], arr[:, 0] - arr[:, 2]
    boot = bootstrap_groups(np.array(mats), {'s': lambda idx: float(d_success[idx].mean()), 'v': lambda idx: float(d_valid[idx].mean())}, B=B)
    ci_s, ci_v = interval(boot['s']), interval(boot['v'])
    return {'n_materials': len(mats), 'success_selector': float(arr[:, 1].mean()), 'success_random': float(arr[:, 3].mean()),
            'diff_success': float(d_success.mean()), 'diff_success_ci': ci_s,
            'valid_selector': float(arr[:, 0].mean()), 'valid_random': float(arr[:, 2].mean()),
            'diff_valid': float(d_valid.mean()), 'diff_valid_ci': ci_v,
            'g4_pass': bool(ci_s[0] > 0 and d_valid.mean() >= -0.02)}


def min_shift_scores(rows, distances, *, layer, position='prompt_end'):
    """Intrinsic score for P6: the smaller of the text-removal and image-removal relative L2 shifts.
    Higher score = larger minimum shift; the direction of use is fixed on the pilot only."""
    by_id = {r['candidate_id']: r for r in distances}
    out = {}
    for r in rows:
        d = by_id.get(r['candidate_id'])
        if not d:
            continue
        text = d['rel_l2']['text_removed'][position][layer]
        image = d['rel_l2']['image_removed'][position][layer]
        out[r['candidate_id']] = float(min(text, image))
    return out


def run_selection(*, run_id, capture_run, label_file, project_root='.', output_root='outputs', layer, position='prompt_end', B=1000):
    """Evaluate the random baseline and the intrinsic min_shift selector on already-labelled candidates."""
    import json
    from pathlib import Path
    from ..core.records import read_jsonl, write_json
    base = Path(project_root).resolve() / output_root
    rows = read_jsonl(base / capture_run / label_file)
    distances = read_jsonl(base / capture_run / 'distances.jsonl')
    intrinsic = min_shift_scores(rows, distances, layer=layer, position=position)
    constant = {r['candidate_id']: 0.0 for r in rows}
    result = {'n_candidates': len(rows), 'n_with_intrinsic': len(intrinsic), 'layer': layer, 'position': position,
              'random': compare(rows, constant, B=B), 'intrinsic_min_shift': compare(rows, intrinsic, B=B),
              'intrinsic_min_shift_mixed': compare(rows, intrinsic, B=B, subset='mixed'),
              'random_mixed': compare(rows, constant, B=B, subset='mixed'),
              'outcome': 'valid and non-redundant (cascade labels); not verified dual-necessary',
              'note': 'random is the within-material mean; intrinsic layer/position were chosen on the pilot only'}
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'selection_results.json', result)
    lines = ['# P6 selection: random baseline and intrinsic min_shift', '',
             f'Run: {capture_run}; labels: {label_file}; min_shift at {position}, layer {layer} (chosen on the pilot).',
             'Outcome = valid and non-redundant, cascade labels (not verified dual-necessary).', '',
             '| Selector | n materials | success rate | random success | diff [95% CI] | valid rate | random valid | diff valid [95% CI] | G4 |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, key in (('random', 'random'), ('intrinsic min_shift (all)', 'intrinsic_min_shift'),
                      ('intrinsic min_shift (mixed materials)', 'intrinsic_min_shift_mixed')):
        v = result[key]
        if v.get('status'):
            lines.append(f'| {name} | {v.get("n_materials")} | - | - | - | - | - | - | {v["status"]} |')
            continue
        lines.append(f'| {name} | {v["n_materials"]} | {v["success_selector"]:.3f} | {v["success_random"]:.3f} | '
                     f'{v["diff_success"]:+.3f} [{v["diff_success_ci"][0]:+.3f}, {v["diff_success_ci"][1]:+.3f}] | '
                     f'{v["valid_selector"]:.3f} | {v["valid_random"]:.3f} | '
                     f'{v["diff_valid"]:+.3f} [{v["diff_valid_ci"][0]:+.3f}, {v["diff_valid_ci"][1]:+.3f}] | {v["g4_pass"]} |')
    (out / 'selection_report.md').write_text('\n'.join(lines) + '\n')
    return {'random': result['random'], 'intrinsic': result['intrinsic_min_shift'],
            'intrinsic_mixed': result['intrinsic_min_shift_mixed']}
