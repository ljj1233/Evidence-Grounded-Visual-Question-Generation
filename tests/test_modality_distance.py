"""Pure parts of the modality-distance readout: swap partners and score assembly (no GPU)."""
from egvqg.modality_distance import SCHEMES, modality_scores, swap_partner


def test_swap_partner_is_deterministic_cyclic_and_never_self():
    ids = ['c', 'a', 'b']
    assert [swap_partner(m, ids) for m in 'abc'] == ['b', 'c', 'a']
    assert swap_partner('a', ['a']) is None and swap_partner('z', ids) is None


def test_variants_of_one_source_never_swap_with_each_other():
    parent = {'x~A': 'x', 'x~C': 'x', 'x~P': 'x', 'y~A': 'y'}
    ids = list(parent)
    assert swap_partner('x~A', ids, parent) == 'y~A' and swap_partner('y~A', ids, parent) == 'x~A'
    assert swap_partner('x~C', ['x~A', 'x~C', 'x~P'], parent) is None


def test_every_scheme_changes_exactly_one_modality():
    for image, text in SCHEMES.values():
        assert (image == 'same') != (text == 'same')


def test_scores_use_the_requested_layer_and_position():
    row = {'rel_l2': {n: {'question_end': [0.0, 0.2, 0.4], 'prompt_end': [0.0, 0.1, 0.1]} for n in SCHEMES}}
    row['rel_l2']['image_removed']['question_end'] = [0.0, 0.5, 0.9]
    s = modality_scores(row, position='question_end', layer=2)['image_removed|text_removed']
    assert s['asymmetry'] == 0.9 - 0.4 and s['min_shift'] == 0.4
