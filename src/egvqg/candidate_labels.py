"""Candidate labels from a cascade evaluation and the material-level propensity profile (P1/P2). No numpy.

Label semantics follow the cascade evaluator: a candidate is *valid* when question_answerability == 1; among valid candidates the
single-modality screen says whether the image alone / the text alone gives a unique full answer. `redundant` = screen cut;
`nonredundant` = both screens explicitly false; a null screen result that cuts nothing is *undecided* and never enters a classifier.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .records import read_jsonl, write_json, write_jsonl
from .stats import wilson_lower

DIRECTIONS = ('image', 'text', 'both')


def derive_label(record):
    """Label of one evaluation record: {'valid', 'redundant', 'direction', 'decided'}; None entries mean not applicable."""
    answerable = record.get('question_answerability')
    if answerable != 1:
        return {'valid': 0 if answerable == 0 else None, 'redundant': None, 'direction': None, 'decided': answerable == 0}
    screen = record.get('screen') or {}
    img, txt = screen.get('image_only'), screen.get('text_only')
    if record.get('screen_cut'):
        direction = 'both' if img is True and txt is True else 'image' if img is True and txt is False else \
                    'text' if txt is True and img is False else None       # a cut with the other side unknown has no direction
        return {'valid': 1, 'redundant': 1, 'direction': direction, 'decided': True}
    if img is False and txt is False:
        return {'valid': 1, 'redundant': 0, 'direction': None, 'decided': True}
    return {'valid': 1, 'redundant': None, 'direction': None, 'decided': False}


def build_label_table(*, capture_run, source_run, eval_run, project_root='.', output_root='outputs'):
    """Join candidates, their pseudo-materials and the evaluation records; write labels.jsonl next to the captures."""
    base = Path(project_root).resolve() / output_root
    cap = base / capture_run
    table = json.loads((cap / f'source_map_{source_run}.json').read_text())
    by_pseudo = {r['material_id']: r for r in read_jsonl(base / eval_run / 'records.jsonl')}
    candidate_pseudo = {cid: pid for pid, v in table.items() for cid in v['candidate_ids']}
    rows = []
    for c in read_jsonl(cap / 'captures.jsonl'):
        pid = candidate_pseudo[c['candidate_id']]
        rec = by_pseudo.get(pid)
        label = derive_label(rec) if rec and rec.get('status') == 'complete' else {'valid': None, 'redundant': None, 'direction': None, 'decided': False}
        rows.append({'candidate_id': c['candidate_id'], 'material_id': c['material_id'], 'skill': c['skill'], 'index': c['index'],
                     'pseudo_id': pid, 'question': c['question'], 'zh_share': c['zh_share'], 'evaluated': bool(rec and rec.get('status') == 'complete'),
                     **label})
    write_jsonl(cap / f'labels_{eval_run}.jsonl', rows)
    return rows


def material_profile(rows, *, min_valid=4):
    """Per-material propensity: redundant fraction among decided valid candidates and the direction mix."""
    by = defaultdict(list)
    for r in rows:
        by[r['material_id']].append(r)
    out = []
    for mid, rs in sorted(by.items()):
        valid = [r for r in rs if r['valid'] == 1]
        decided = [r for r in valid if r['redundant'] is not None]
        dirs = Counter(r['direction'] for r in decided if r['direction'])
        out.append({'material_id': mid, 'skill': rs[0]['skill'], 'n_candidates': len(rs), 'n_valid': len(valid), 'n_decided': len(decided),
                    'n_distinct_questions': len({r['question'].strip() for r in rs}),
                    'redundant_fraction': (sum(r['redundant'] for r in decided) / len(decided)) if decided else None,
                    'directions': dict(dirs), 'eligible': len(valid) >= min_valid})
    return out


def g1_check(profile, *, threshold=0.25):
    """Pre-registered G1: among eligible materials, (a) 0 < redundant fraction < 1 or (b) >= 2 distinct directions, each in >= threshold."""
    eligible = [p for p in profile if p['eligible'] and p['redundant_fraction'] is not None]
    n = len(eligible)
    a = sum(0 < p['redundant_fraction'] < 1 for p in eligible)
    b = sum(len(p['directions']) >= 2 for p in eligible)
    return {'eligible_materials': n, 'fraction_mixed_redundancy': a / n if n else None, 'fraction_mixed_direction': b / n if n else None,
            'pass_a': bool(n) and a / n >= threshold, 'pass_b': bool(n) and b / n >= threshold,
            'passed': bool(n) and (a / n >= threshold or b / n >= threshold), 'threshold': threshold}


def coverage(rows):
    """Share of candidates with a decided label (valid/invalid); evaluation failures and pending items are not covered."""
    return sum(bool(r['decided']) for r in rows) / len(rows) if rows else None


def profile_report(*, capture_run, eval_run, project_root='.', output_root='outputs'):
    cap = Path(project_root).resolve() / output_root / capture_run
    rows = read_jsonl(cap / f'labels_{eval_run}.jsonl')
    prof = material_profile(rows)
    counts = Counter((r['valid'], r['redundant'], r['direction']) for r in rows)
    result = {'candidates': len(rows), 'materials': len(prof), 'coverage': coverage(rows), 'g1': g1_check(prof),
              'valid': sum(r['valid'] == 1 for r in rows), 'redundant': sum(r['redundant'] == 1 for r in rows),
              'nonredundant': sum(r['redundant'] == 0 and r['valid'] == 1 for r in rows),
              'undecided_valid': sum(r['valid'] == 1 and r['redundant'] is None for r in rows),
              'directions': dict(Counter(r['direction'] for r in rows if r['direction'])),
              'label_counts': {str(k): v for k, v in counts.items()},
              'materials_with_any_nonredundant': sum(any(r['redundant'] == 0 for r in rows if r['material_id'] == p['material_id']) for p in prof)}
    write_json(cap / f'profile_{eval_run}.json', {**result, 'per_material': prof})
    return result
