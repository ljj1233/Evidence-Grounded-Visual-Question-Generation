"""Audit sampling (stratified, unique, per-material cap, reproducible) and agreement statistics."""
import pytest

from egvqg.label_audit import audit_summary, sample_audit


def rows():
    out = []
    for m in range(12):
        for j, d in enumerate(['image', 'text', 'both']):
            out.append({'candidate_id': f'm{m}~c{j}', 'pseudo_id': f'm{m}~p{j}', 'material_id': f'm{m}', 'redundant': 1, 'direction': d})
    out.append({'candidate_id': 'dup', 'pseudo_id': 'm0~p0', 'material_id': 'm0', 'redundant': 1, 'direction': 'image'})   # duplicate pseudo item
    out.append({'candidate_id': 'non', 'pseudo_id': 'x', 'material_id': 'm0', 'redundant': 0, 'direction': None})
    return out


def test_sample_is_stratified_unique_capped_and_reproducible():
    quota = (('image', 4), ('text', 3), ('both', 4))
    a = sample_audit(rows(), seed=1, quota=quota, per_material=1)
    assert [r['direction'] for r in a].count('image') == 4 and [r['direction'] for r in a].count('text') == 3
    assert len({r['pseudo_id'] for r in a}) == len(a) and all(r['redundant'] == 1 for r in a)
    assert [r['candidate_id'] for r in a] == [r['candidate_id'] for r in sample_audit(rows(), seed=1, quota=quota, per_material=1)]
    per = {}
    for r in a:
        per[(r['material_id'], r['direction'])] = per.get((r['material_id'], r['direction']), 0) + 1
    assert max(per.values()) == 1             # cap applies per material within a direction stratum


def test_summary_counts_agreement_and_direction_separately():
    reviews = [{'n': 0, 'claimed': 'image', 'reviewer_image_alone': True, 'reviewer_text_alone': False},
               {'n': 1, 'claimed': 'both', 'reviewer_image_alone': True, 'reviewer_text_alone': False},     # redundant yes, direction no
               {'n': 2, 'claimed': 'text', 'reviewer_image_alone': False, 'reviewer_text_alone': False}]    # not redundant
    s = audit_summary(reviews)
    assert s['agree_redundant'] == 2 and s['direction_agree'] == 1 and s['disagreements'] == [2]
    assert s['wilson_lower_95'] < s['agree_redundant_rate']


def test_two_class_sample_covers_both_labels_and_stays_unique():
    from egvqg.label_audit import sample_two_class
    rws = []
    for m in range(14):
        for j, d in enumerate(['image', 'text', 'both']):
            rws.append({'candidate_id': f'm{m}~r{j}', 'pseudo_id': f'm{m}~rp{j}', 'material_id': f'm{m}',
                        'valid': 1, 'redundant': 1, 'direction': d})
    for m in range(8):
        rws.append({'candidate_id': f'm{m}~n', 'pseudo_id': f'm{m}~np', 'material_id': f'm{m}',
                    'valid': 1, 'redundant': 0, 'direction': None})
    rws.append({'candidate_id': 'dup', 'pseudo_id': 'm0~rp0', 'material_id': 'm0', 'valid': 1, 'redundant': 1, 'direction': 'image'})
    rws.append({'candidate_id': 'invalid', 'pseudo_id': 'bad', 'material_id': 'm0', 'valid': 0, 'redundant': 0, 'direction': None})
    items = sample_two_class(rws, seed=2, n_redundant=9, n_nonredundant=5, quota=(('image', 3), ('text', 3), ('both', 3)), per_material=1)
    assert sum(r['audit_class'] == 'redundant' for r in items) == 9
    assert sum(r['audit_class'] == 'nonredundant' for r in items) == 5
    assert len({r['pseudo_id'] for r in items}) == len(items)
    assert all(r['pseudo_id'] != 'bad' for r in items)
    assert set(r['direction'] for r in items if r['audit_class'] == 'redundant') == {'image', 'text', 'both'}


def test_two_class_summary_precision_recall_and_bounds():
    from egvqg.label_audit import two_class_summary
    reviews = ([{'n': n, 'audit_class': 'redundant', 'reviewer_image_alone': n < 8, 'reviewer_text_alone': False} for n in range(10)] +
               [{'n': 10 + n, 'audit_class': 'nonredundant', 'reviewer_image_alone': n >= 9, 'reviewer_text_alone': False} for n in range(10)])
    s = two_class_summary(reviews, p_label_redundant=0.8, p_label_nonredundant=0.2)
    assert s['redundant_precision'] == 0.8 and s['nonredundant_precision'] == 0.9
    assert abs(s['nonredundant_recall'] - 0.18 / 0.34) < 1e-9
    assert s['redundant_disagreements'] == [8, 9] and s['nonredundant_disagreements'] == [19]
    lo, hi = s['nonredundant_precision_wilson_95']
    assert lo < 0.9 < hi and 0 <= lo <= hi <= 1
