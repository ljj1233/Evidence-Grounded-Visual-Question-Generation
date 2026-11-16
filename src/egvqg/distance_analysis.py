"""P5 readout analysis: do per-layer modality shifts (modality_distance.py) predict redundancy / direction? (pre-registered, plan 7.7)

Layer and position are chosen on the discovery data (pilot) per metric, averaged over the three scheme pairs; the confirmation data
(expansion) is then scored at that layer. A conclusion is *adopted* only when at least two of the three scheme pairs give an AUC
interval on the same side of 0.5 that excludes 0.5 on the confirmation data. No gate; results are reported as they come out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .probing import auc, bootstrap_groups, interval
from .records import read_jsonl, write_json

PAIRS = (('image_removed', 'text_removed'), ('image_gray', 'text_removed'), ('image_swap', 'text_swap'))
POSITIONS = ('prompt_end', 'question_end')
METRICS = ('asymmetry', 'min_shift')


def join(label_rows, distance_rows):
    dist = {d['candidate_id']: d for d in distance_rows}
    return [(r, dist[r['candidate_id']]) for r in label_rows if r['candidate_id'] in dist]


def feature(d, pair, position, metric, layer, field='rel_l2'):
    img, txt = pair
    if img not in d[field] or txt not in d[field]:
        return float('nan')
    di, dt = d[field][img][position][layer], d[field][txt][position][layer]
    return di - dt if metric == 'asymmetry' else min(di, dt)


def targets(pairs):
    """Binary targets: nonredundant (1) vs redundant (0); image-sufficient (1) vs text-sufficient (0) among single-direction candidates."""
    red = [(r, d) for r, d in pairs if r['valid'] == 1 and r['redundant'] is not None]
    dirs = [(r, d) for r, d in pairs if r['redundant'] == 1 and r['direction'] in ('image', 'text')]
    return {'nonredundant': ([p for p in red], lambda r: int(r['redundant'] == 0)),
            'image_vs_text': ([p for p in dirs], lambda r: int(r['direction'] == 'image'))}


def curve(pairs, target, pair, position, metric):
    items, y_of = target
    y = np.array([y_of(r) for r, _ in items])
    out = []
    n_layers = len(items[0][1]['rel_l2']['image_removed'][position]) if items else 0
    for layer in range(n_layers):
        x = np.array([feature(d, pair, position, metric, layer) for _, d in items])
        ok = ~np.isnan(x)
        out.append(auc(y[ok], x[ok]) if ok.sum() > 10 else float('nan'))
    return out


def score_at(pairs, target, pair, position, metric, layer, B=1000):
    items, y_of = target
    y = np.array([y_of(r) for r, _ in items])
    x = np.array([feature(d, pair, position, metric, layer) for _, d in items])
    groups = np.array([r['material_id'] for r, _ in items])
    ok = ~np.isnan(x)
    y, x, groups = y[ok], x[ok], groups[ok]
    if len(set(y.tolist())) < 2:
        return {'n': int(len(y)), 'auc': float('nan'), 'ci': [float('nan')] * 2}
    boot = bootstrap_groups(groups, {'a': lambda idx: auc(y[idx], x[idx])}, B=B)
    return {'n': int(len(y)), 'auc': auc(y, x), 'ci': interval(boot['a'])}


def run_p5(*, run_id, discovery, confirmation=None, project_root='.', output_root='outputs', B=1000):
    """discovery/confirmation: list of (capture_run, label_file). Writes p5_readout.json under outputs/<run_id>/."""
    base = Path(project_root).resolve() / output_root

    def load(sets):
        out = []
        for run, label_file in sets:
            out += join(read_jsonl(base / run / label_file), read_jsonl(base / run / 'distances.jsonl'))
        return out
    d_pairs = load(discovery)
    c_pairs = load(confirmation) if confirmation else None
    result = {'n_discovery': len(d_pairs), 'n_confirmation': len(c_pairs) if c_pairs else 0, 'entries': {}}
    d_targets = targets(d_pairs)
    c_targets = targets(c_pairs) if c_pairs else None
    for tname, tgt in d_targets.items():
        for position in POSITIONS:
            for metric in METRICS:
                curves = {f'{p[0]}|{p[1]}': curve(d_pairs, tgt, p, position, metric) for p in PAIRS}
                mean_dev = np.nanmean([np.abs(np.array(c) - 0.5) for c in curves.values()], axis=0)
                layer = int(np.nanargmax(mean_dev)) if not np.all(np.isnan(mean_dev)) else None
                entry = {'discovery_curves': curves, 'chosen_layer': layer}
                if c_targets is not None and layer is not None:
                    conf = {f'{p[0]}|{p[1]}': score_at(c_pairs, c_targets[tname], p, position, metric, layer, B) for p in PAIRS}
                    sides = [1 if v['ci'][0] > 0.5 else -1 if v['ci'][1] < 0.5 else 0 for v in conf.values()]
                    entry['confirmation'] = conf
                    entry['adopted'] = bool(max(sides.count(1), sides.count(-1)) >= 2)
                result['entries'][f'{tname}/{position}/{metric}'] = entry
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'p5_readout.json', result)
    return {k: {'layer': v['chosen_layer'], 'adopted': v.get('adopted')} for k, v in result['entries'].items()}
