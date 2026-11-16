"""Best-of-k selection: random baseline is the within-material mean, ties are averaged, G4 needs a positive interval."""
import pytest

np = pytest.importorskip('numpy')

from egvqg.selection_eval import compare, per_material


def mat(m, k=8, good=2, seed=0):
    rng = np.random.default_rng(seed + hash(m) % 1000)
    rows = []
    for j in range(k):
        success = j < good
        rows.append({'candidate_id': f'{m}~{j}', 'material_id': m, 'valid': 1, 'redundant': 0 if success else 1, 'evaluated': True})
    return rows


def test_oracle_selector_beats_random_and_constant_selector_equals_random():
    rows = [r for m in range(30) for r in mat(f'm{m}')]
    oracle = {r['candidate_id']: 1.0 - r['redundant'] for r in rows}
    res = compare(rows, oracle, B=100)
    assert res['success_selector'] == 1.0 and res['success_random'] == pytest.approx(0.25) and res['g4_pass']
    const = {r['candidate_id']: 0.0 for r in rows}
    flat = compare(rows, const, B=100)
    assert flat['diff_success'] == pytest.approx(0.0) and not flat['g4_pass']


def test_picking_invalid_candidates_fails_the_validity_condition():
    rows = []
    for m in range(30):
        rs = mat(f'm{m}')
        rs[7]['valid'], rs[7]['redundant'] = 0, None
        rows += rs
    bad = {r['candidate_id']: (1.0 if r['valid'] == 0 else 0.0) for r in rows}      # always picks the invalid one
    res = compare(rows, bad, B=100)
    assert res['diff_valid'] < -0.02 and not res['g4_pass']


def test_mixed_subset_and_missing_scores():
    rows = [r for m in range(10) for r in mat(f'm{m}', good=0)] + [r for m in range(10, 20) for r in mat(f'm{m}', good=3)]
    scores = {r['candidate_id']: float(r['redundant'] == 0) for r in rows}
    assert compare(rows, scores, B=50, subset='mixed')['n_materials'] == 10
    assert len(per_material(rows, {rows[0]['candidate_id']: 1.0})) == 1


def test_min_shift_scores_takes_the_smaller_removal_shift():
    from egvqg.selection_eval import min_shift_scores
    rows = [{'candidate_id': 'a'}, {'candidate_id': 'b'}, {'candidate_id': 'missing'}]
    distances = [
        {'candidate_id': 'a', 'rel_l2': {'text_removed': {'prompt_end': [0, 0.2]}, 'image_removed': {'prompt_end': [0, 0.5]}}},
        {'candidate_id': 'b', 'rel_l2': {'text_removed': {'prompt_end': [0, 0.7]}, 'image_removed': {'prompt_end': [0, 0.3]}}}]
    assert min_shift_scores(rows, distances, layer=1) == {'a': 0.2, 'b': 0.3}
