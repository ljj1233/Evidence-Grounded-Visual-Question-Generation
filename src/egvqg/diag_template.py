"""1b / H2: how much of the hidden-state variance is template (skill), material, and candidate residual?

Pure vector algebra (no labels), so it can run on a single position/layer and on synthetic data. Reports:
  - nested sum-of-squares shares SS_skill / SS_material / SS_candidate;
  - norm ratio ||mu_level|| / mean||x - mu_level|| for global / skill / material;
  - mean within-material pairwise cosine after none / global / skill / material centering;
  - Spearman correlation of per-candidate residual norms and material cosines with the question token count.
Descriptive only: 1b answers "how large is the template component", nothing causal.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import candidate_table as CT
from .centering import center, group_means, keys_for, within_material_cosine
from .records import write_json
from .stats import spearman


def decomposition(X, skill, material):
    """Nested SS decomposition; shares sum to 1 (before clipping)."""
    X = np.asarray(X, float)
    skill, material = np.asarray(skill), np.asarray(material)
    grand = X.mean(0)
    ss_cand = 0.0
    ss_mat = 0.0
    ss_skill = 0.0
    mat_mean, mat_skill, mat_n = {}, {}, {}
    for m in np.unique(material):
        idx = np.where(material == m)[0]
        mat_mean[m] = X[idx].mean(0)
        mat_skill[m] = skill[idx][0]
        mat_n[m] = len(idx)
        ss_cand += float(((X[idx] - mat_mean[m]) ** 2).sum())
    for s in np.unique(skill):
        mats = [m for m in mat_mean if mat_skill[m] == s]
        n_s = sum(mat_n[m] for m in mats)
        skill_mean = sum(mat_n[m] * mat_mean[m] for m in mats) / n_s
        ss_skill += n_s * float(((skill_mean - grand) ** 2).sum())
        for m in mats:
            ss_mat += mat_n[m] * float(((mat_mean[m] - skill_mean) ** 2).sum())
    total = ss_skill + ss_mat + ss_cand
    ss = {'skill': ss_skill, 'material': ss_mat, 'candidate': ss_cand}
    return {'ss': ss, 'share': {k: (v / total if total > 0 else float('nan')) for k, v in ss.items()},
            'total_ss': total, 'n_candidates': int(len(X)), 'n_materials': len(mat_n), 'n_skills': len(set(skill.tolist()))}


def norm_ratios(X, meta):
    out = {}
    for level in ('global', 'skill', 'material'):
        keys = keys_for(meta, level)
        means, counts = group_means(X, keys)
        row_mean = np.stack([means[keys[i]] for i in range(len(X))])
        row_res = X - row_mean
        mu_norm = float(np.mean([np.linalg.norm(v) for v in means.values()]))
        res_norm = float(np.linalg.norm(row_res, axis=1).mean())
        out[level] = {'mu_norm': mu_norm, 'residual_norm': res_norm, 'ratio': mu_norm / res_norm if res_norm > 0 else float('inf')}
    return out


def question_token_control(X, meta, tokens):
    """Spearman of per-row residual norm and material cosine with the question token count."""
    tokens = np.asarray(tokens, float)
    material = np.asarray(meta['material_id'])
    means, _ = group_means(X, material)
    res_norm = np.linalg.norm(X - np.stack([means[m] for m in material]), axis=1)
    cos = []
    for i, m in enumerate(material):
        a, b = X[i], means[m]
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    rho_res = spearman(res_norm, tokens)
    rho_cos = spearman(np.array(cos), tokens)
    mat_tokens = {m: float(np.nanmean(tokens[material == m])) for m in np.unique(material)}
    mat_norm = {m: float(np.linalg.norm(v)) for m, v in means.items()}
    rho_mat = spearman([mat_norm[m] for m in mat_tokens], [mat_tokens[m] for m in mat_tokens])
    return {'residual_norm_vs_tokens': rho_res, 'material_cosine_vs_tokens': rho_cos, 'material_mean_norm_vs_tokens': rho_mat}


def analyze(X, meta, tokens=None):
    """All 1b readings for one (position, layer)."""
    skill, material = np.asarray(meta['skill']), np.asarray(meta['material_id'])
    dec = decomposition(X, skill, material)
    rows = {'decomposition': dec, 'norm_ratio': norm_ratios(X, meta), 'pairwise_cosine': {}}
    for level in ('none', 'global', 'skill', 'material'):
        rows['pairwise_cosine'][level] = within_material_cosine(center(X, meta, level), material)
    if tokens is not None:
        rows['token_control'] = question_token_control(X, meta, tokens)
    return rows


def run_diag_template(*, capture_run, label_file, layers=(22,), positions=('prompt_end',), materials=None, project_root='.',
                      output_root='outputs', run_id=None, dry_run=False):
    base = Path(project_root).resolve() / output_root
    rows = CT.build_table(capture_run=capture_run, label_file=label_file, project_root=project_root, output_root=output_root)
    if materials:
        ids = {line.strip() for line in Path(materials).read_text().splitlines() if line.strip()}
        rows = [r for r in rows if r['parent_id'] in ids or r['material_id'] in ids]
    if not rows:
        raise ValueError('no candidates after material filtering')
    if dry_run:
        layers, positions = tuple(layers)[:1], tuple(positions)[:1]
        print(f'[dry-run] diag-template: {len(rows)} candidates, layers={list(layers)}, positions={list(positions)}; no GPU/API used')
    meta = CT.meta(rows)
    feats = CT.load_feats(capture_run=capture_run, rows=rows, layers=layers, positions=positions,
                          project_root=project_root, output_root=output_root)
    records = []
    for pos in positions:
        for li, layer in enumerate(layers):
            res = analyze(feats[pos][:, li], meta, tokens=[r['n_question_tokens'] or 0 for r in rows])
            record = {'position': pos, 'layer': layer, 'n_candidates': len(rows), **res}
            records.append(record)
            d, nr, pc = res['decomposition'], res['norm_ratio'], res['pairwise_cosine']
            print(f'[diag-template] pos={pos} layer={layer} shares skill={d["share"]["skill"]:.3f} '
                  f'mat={d["share"]["material"]:.3f} cand={d["share"]["candidate"]:.3f} '
                  f'cos(none)={pc["none"]:.3f} cos(mat)={pc["material"]:.3f}')
    if dry_run:
        return {'dry_run': True, 'rows': len(records), 'preview': records[0]}
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'diag_template.json', {'capture_run': capture_run, 'records': records})
    fields = ['position', 'layer', 'n_candidates', 'share_skill', 'share_material', 'share_candidate',
              'ratio_global', 'ratio_skill', 'ratio_material', 'cos_none', 'cos_global', 'cos_skill', 'cos_material',
              'rho_residual_norm_tokens', 'rho_material_cosine_tokens', 'rho_material_mean_norm_tokens']
    with (out / 'diag_template.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in records:
            d, nr, pc = r['decomposition'], r['norm_ratio'], r['pairwise_cosine']
            tc = r.get('token_control', {})
            w.writerow([r['position'], r['layer'], r['n_candidates'], d['share']['skill'], d['share']['material'],
                        d['share']['candidate'], nr['global']['ratio'], nr['skill']['ratio'], nr['material']['ratio'],
                        pc['none'], pc['global'], pc['skill'], pc['material'],
                        tc.get('residual_norm_vs_tokens', (float('nan'),))[0],
                        tc.get('material_cosine_vs_tokens', (float('nan'),))[0],
                        tc.get('material_mean_norm_vs_tokens', (float('nan'),))[0]])
    return {'run_id': run_id, 'records': len(records), 'csv': str(out / 'diag_template.csv')}
