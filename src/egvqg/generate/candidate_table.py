"""Unified candidate table (candidates.jsonl) and the hidden-state loader aligned to it.

The table is a *conversion* of existing artifacts: `captures.jsonl` + `labels_<eval_run>.jsonl`, optionally
augmented with `labels_new.jsonl` produced by `label_repair.py`. Original files are never modified.

Fields: cand_id, parent_id, material_id, skill, condition, question, label_old {redundant, direction},
label_new {verdict, direction, evidence} or null, valid, plus the surface fields used by the input-feature and
lexical baselines and `n_question_tokens` for the length control.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.records import read_jsonl, write_jsonl

POSITIONS = ('prompt_end', 'early', 'question_end')


def _label_new_map(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'label_new file not found: {p}')
    return {r['cand_id'] if 'cand_id' in r else r['candidate_id']: r for r in read_jsonl(p)}


def build_table(*, capture_run, label_file, project_root='.', output_root='outputs', label_new=None, run_id=None):
    """Convert captures + cascade labels into candidate rows; writes `outputs/<run_id>/candidates.jsonl` when run_id is given."""
    base = Path(project_root).resolve() / output_root
    cap = base / capture_run
    labels = {r['candidate_id']: r for r in read_jsonl(cap / label_file)}
    new = _label_new_map(label_new)
    identity_path = cap / 'model_identity.json'
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    default_condition = identity.get('intervention') or identity.get('prompt_template') or 'base'
    rows = []
    for c in read_jsonl(cap / 'captures.jsonl'):
        lab = labels.get(c['candidate_id'], {})
        new_row = new.get(c['candidate_id'])
        rows.append({
            'cand_id': c['candidate_id'],
            'parent_id': c.get('parent_material_id') or c['material_id'],
            'material_id': c['material_id'],
            'skill': c['skill'],
            'condition': c.get('condition') or default_condition,
            'question': c['question'],
            'label_old': {'redundant': lab.get('redundant'), 'direction': lab.get('direction')},
            'label_new': ({'verdict': new_row.get('verdict'), 'direction': new_row.get('direction'), 'evidence': new_row.get('evidence')}
                          if new_row else None),
            'valid': lab.get('valid'),
            'evaluated': bool(lab.get('evaluated', False)),
            'features': {'text_chars': c.get('text_chars'), 'n_choices': c.get('n_choices'), 'image': c.get('image'),
                         'zh_share': c.get('zh_share'), 'generated_token_ids': c.get('generated_token_ids')},
            'n_question_tokens': (c.get('plan') or {}).get('n_question_tokens', c.get('n_generated_tokens')),
            'states_file': c.get('states_file'),
        })
    if run_id:
        out = base / run_id
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / 'candidates.jsonl', rows)
    return rows


def load_table(path):
    rows = read_jsonl(Path(path))
    for r in rows:
        r.setdefault('parent_id', r.get('material_id'))
    return rows


def meta(rows):
    """Dict of arrays for centering / grouping (parent_id is the bootstrap and CV unit)."""
    return {'skill': np.array([r['skill'] for r in rows]), 'material_id': np.array([r['material_id'] for r in rows]),
            'parent_id': np.array([r.get('parent_id') or r['material_id'] for r in rows])}


def load_feats(*, capture_run, rows, layers=(22,), positions=POSITIONS, project_root='.', output_root='outputs'):
    """{position: ndarray [N, n_layers, d]} aligned to `rows` by cand_id; `early` is pooled by its token mean."""
    from safetensors.torch import load_file
    base = Path(project_root).resolve() / output_root / capture_run
    layers = list(layers)
    out = {p: [] for p in positions}
    for r in rows:
        path = base / r['states_file']
        if not path.exists():
            raise FileNotFoundError(f'hidden states missing for {r["cand_id"]}: {path}')
        t = load_file(str(path))
        for p in positions:
            if p not in t:
                raise ValueError(f'{p} state missing for {r["cand_id"]} (empty question?)')
            a = t[p].float().numpy()
            if p == 'early':
                a = a.mean(0)
            out[p].append(a[layers])
    return {p: np.stack(v) for p, v in out.items()}


def split_targets(rows, kind='redundancy'):
    """Indices / targets used by the diagnostics.

    kind='redundancy': valid and decided; label_new verdict=='dual_necessary' is 1 (non-redundant), the three
      suff-suff verdicts are 0, undetermined dropped. Falls back to label_old when no new labels exist.
    kind='direction': single-direction candidates; image=1, text=0, 'both'/dual dropped.
    """
    idx, y = [], []
    for i, r in enumerate(rows):
        if r.get('valid') != 1:
            continue
        new = r.get('label_new') or {}
        verdict = new.get('verdict')
        if kind == 'redundancy':
            if verdict:
                if verdict == 'undetermined':
                    continue
                val = 1 if verdict == 'dual_necessary' else 0
            elif r['label_old'].get('redundant') is not None:
                val = int(r['label_old']['redundant'] == 0)
            else:
                continue
            idx.append(i); y.append(val)
        elif kind == 'direction':
            d = new.get('direction') or r['label_old'].get('direction')
            if d in ('image', 'text'):
                idx.append(i); y.append(int(d == 'image'))
    return np.array(idx, int), np.array(y, int)
