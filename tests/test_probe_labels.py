"""Label derivation from cascade-evaluation records, propensity profile and the pre-registered G1 rule."""
import pytest

from egvqg.probe_labels import coverage, derive_label, g1_check, material_profile, wilson_lower


def rec(answerable, img=None, txt=None, cut=None):
    return {'question_answerability': answerable, 'screen': None if answerable != 1 else {'image_only': img, 'text_only': txt}, 'screen_cut': cut}


@pytest.mark.parametrize('record,expected', [
    (rec(1, True, False, True), (1, 1, 'image')),
    (rec(1, False, True, True), (1, 1, 'text')),
    (rec(1, True, True, True), (1, 1, 'both')),
    (rec(1, True, None, True), (1, 1, None)),          # cut, other side unknown: redundant without a direction
    (rec(1, False, False, None), (1, 0, None)),        # survivor of both screens
    (rec(1, False, None, None), (1, None, None)),      # screen failed, nothing cut: undecided
    (rec(0), (0, None, None)),
    (rec(None), (None, None, None)),
])
def test_label_derivation_matches_cascade_semantics(record, expected):
    label = derive_label(record)
    assert (label['valid'], label['redundant'], label['direction']) == expected


def test_decided_means_valid_or_invalid_with_a_usable_screen():
    assert derive_label(rec(0))['decided'] and not derive_label(rec(None))['decided'] and not derive_label(rec(1, False, None, None))['decided']


def row(m, valid, red, direction=None, q='q'):
    return {'material_id': m, 'skill': 's', 'valid': valid, 'redundant': red, 'direction': direction, 'question': q, 'decided': valid is not None}


def test_profile_fraction_excludes_undecided_and_invalid_and_g1_follows_the_rule():
    rows = [row('a', 1, 1, 'image', 'x'), row('a', 1, 1, 'text', 'y'), row('a', 1, 0), row('a', 1, None), row('a', 0, None),
            row('b', 1, 1, 'image'), row('b', 1, 1, 'image'), row('b', 1, 1, 'image'), row('b', 1, 1, 'image')]
    prof = {p['material_id']: p for p in material_profile(rows)}
    assert prof['a']['redundant_fraction'] == pytest.approx(2 / 3) and prof['a']['n_valid'] == 4 and prof['a']['eligible']
    assert prof['b']['redundant_fraction'] == 1.0 and prof['b']['directions'] == {'image': 4}
    g1 = g1_check(list(prof.values()))
    assert g1['eligible_materials'] == 2 and g1['pass_a'] and g1['pass_b'] and g1['passed']          # 1/2 of materials is mixed
    flat = g1_check([prof['b']])
    assert not flat['passed']


def test_wilson_and_coverage():
    assert wilson_lower(40, 40) == pytest.approx(0.912, abs=0.002) and wilson_lower(0, 0) == 0.0
    assert coverage([{'decided': True}, {'decided': False}]) == 0.5
