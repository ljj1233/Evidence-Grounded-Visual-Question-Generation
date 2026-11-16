"""Helpers of the baseline and evaluation workflow (the remaining part of an older module; everything else was removed)."""
from __future__ import annotations

from collections import Counter
from .records import CONDITIONS


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None

def _three_and(*values):
    if any(v == 0 for v in values):
        return 0
    return 1 if all(v == 1 for v in values) else None

def _three_not(value):
    return None if value is None else 1 - value

def _label_value(item):
    if not isinstance(item, dict) or item.get('status') != 'decided':
        return None
    value = item.get('value')
    if type(value) not in (int, bool) or value not in (0, 1):
        raise ValueError('decided label must be 0 or 1')
    return int(value)

def record_states(record, source='automatic'):
    """Three-valued states; reject stale answer/score bindings before aggregating."""
    from .records import canonical_json, sha256_text
    q = record['question']; ref = record['reference']
    absent = not q.get('normalized_text') and record['generation']['status'] in (
        'generation_failure', 'technical_failure', 'no_question')
    if q.get('normalized_text'):
        if q['sha256'] != sha256_text(q['normalized_text']):
            raise ValueError('question hash mismatch')
        if ref.get('question_sha256') != q['sha256'] or ref.get('question_id') != q['question_id']:
            raise ValueError('reference is bound to another question')
    labels = record['labels'][source]
    v = 0 if absent else _label_value(labels['validity'])
    acceptable_statuses = ('verified', 'verified_automatic') if source == 'automatic' else ('verified',)
    verified = ref['status'] in acceptable_statuses
    reference_sha = sha256_text(canonical_json({k: ref[k] for k in ('verified_answer', 'acceptable_answers', 'rubric')}))
    scores = {}
    for c in CONDITIONS:
        r = record['responses'][c]
        score = None
        if r:
            if r['question_id'] != q['question_id'] or r['question_sha256'] != q['sha256']:
                raise ValueError('response is bound to another question')
            if verified and r['status'] == 'answered' and not r.get('abstain') and r.get('score_status') == 'scored':
                if r.get('score_reference_sha256') != reference_sha:
                    raise ValueError('score reference changed; re-score affected responses')
                value = r.get('correct')
                if type(value) not in (int, bool) or value not in (0, 1):
                    raise ValueError('scored response must have 0/1 correctness')
                score = int(value)
        scores[c] = score
    b = _three_and(v, scores['TI'], _three_not(scores['T']), _three_not(scores['I']))
    shortcuts = {k: _three_and(v, _label_value(item)) for k, item in record['shortcuts'][source].items()}
    shortcut_values = list(shortcuts.values())
    any_shortcut = 1 if 1 in shortcut_values else (0 if all(x == 0 for x in shortcut_values) else None)
    return {'validity': v, 'bdr': b, 'conditions': {c: _three_and(v, scores[c]) for c in CONDITIONS},
            'shortcuts': shortcuts, 'any_shortcut': any_shortcut,
            'labels': {k: (0 if absent or item.get('status')=='not_applicable' else _label_value(item)) for k, item in labels.items()}}

def _fixed_summary(values, n, *, n_not_applicable=0):
    values = list(values)
    if len(values) != n:
        raise ValueError('metric lost planned attempts')
    passed, failed = values.count(1), values.count(0)
    unresolved = values.count(None)
    return {'n_planned': n, 'n_pass': passed, 'n_fail': failed - n_not_applicable,
            'n_not_applicable': n_not_applicable, 'n_unresolved': unresolved,
            'rate': _rate(passed, n), 'unresolved_rate': _rate(unresolved, n),
            'lower': _rate(passed, n), 'upper': _rate(passed + unresolved, n)}

def compute_record_metrics(rows, *, source='automatic'):
    n = len(rows); states = [record_states(r, source) for r in rows]
    absent = sum(not r['question']['normalized_text'] and r['generation']['status'] in (
        'generation_failure', 'technical_failure', 'no_question') for r in rows)
    result = {key: _fixed_summary((s[key] for s in states), n) for key in ('validity', 'bdr', 'any_shortcut')}
    result['n_planned'] = n
    for group in ('conditions', 'shortcuts', 'labels'):
        keys = CONDITIONS if group == 'conditions' else (
            ('text_only', 'image_only', 'common_knowledge', 'question_leakage', 'independent_stitching') if group == 'shortcuts'
            else ('validity', 'relatedness', 'text_necessary', 'image_necessary', 'integration'))
        result[group] = {k: _fixed_summary((s[group][k] for s in states), n,
            n_not_applicable=sum((not r['question']['normalized_text'] and r['generation']['status'] in (
                'generation_failure','technical_failure','no_question')) or r['labels'][source][k].get('status')=='not_applicable'
                for r in rows) if group == 'labels' and k != 'validity' else 0) for k in keys}
    # Disjoint cross-tab cells retain every row, including invalid and pending.
    cross = Counter()
    for s in states:
        l = s['labels']
        row = f"V={s['validity']} R={l['relatedness']}"
        if l['text_necessary'] == 0 or l['image_necessary'] == 0:
            column = 'necessity_fail'
        elif l['text_necessary'] == l['image_necessary'] == 1:
            column = 'both_necessary_integrated' if l['integration'] == 1 else (
                'both_necessary_stitched' if l['integration'] == 0 else 'unresolved')
        else:
            column = 'unresolved'
        cross[(row, column)] += 1
    result['cross_table'] = [{'row': r, 'column': c, 'n': count, 'share': _rate(count, n)}
                             for (r, c), count in sorted(cross.items())]
    core = Counter({k:0 for k in ('joint_integrated','single_modality','knowledge_or_leakage','stitched','unresolved')})
    for state in states:
        l=state['labels']; shortcuts=state['shortcuts']
        if state['validity']!=1 or l['relatedness']!=1:
            continue
        if shortcuts['independent_stitching']==1:
            key='stitched'
        elif shortcuts['common_knowledge']==1 or shortcuts['question_leakage']==1:
            key='knowledge_or_leakage'
        elif l['text_necessary']==0 or l['image_necessary']==0:
            key='single_modality'
        elif l['text_necessary']==l['image_necessary']==l['integration']==1:
            key='joint_integrated'
        else:
            key='unresolved'
        core[key]+=1
    result['core_cross_table']={'validity':1,'relatedness':1,**dict(core),'row_total':sum(core.values()),'denominator':n}
    return result
