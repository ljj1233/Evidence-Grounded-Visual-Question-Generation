"""MMQA refer-v2 comparison: fixed denominators and the official-cascade screen mapping."""
import json

from egvqg.studies.mmqa_refer import _cascade_condition, _condition


def _material(mid='m1'):
    return {'material_id': mid, 'text': 'The film was released in 2005 and directed by Joe.', 'image_path': 'x.jpg'}


def _row(cid, verdict, question='Who directed the film?'):
    return {'cand_id': cid, 'material_id': 'm1', 'question': question,
            'label_consensus': {'verdict': verdict}, 'label_new': {'verdict': verdict, 'evidence': {'groups': {}}}}


def test_condition_keeps_fixed_denominator_and_decided_redundancy():
    rows = [_row('c0', 'dual_necessary'), _row('c1', 'text_suff'),
            _row('c2', 'undetermined'), _row('c3', 'both_suff')]
    result = _condition(rows, {'m1': _material()})
    assert result['n'] == 4 and result['decided'] == 3
    assert result['dual_rate']['k'] == 1 and result['dual_rate']['n'] == 4
    # `text_suff` means the text alone suffices, i.e. the IMAGE is redundant.
    assert result['text_redundancy']['k'] == 1 and result['text_redundancy']['n'] == 3
    assert result['image_redundancy']['k'] == 2 and result['image_redundancy']['n'] == 3


def test_cascade_mapping_requires_validity_and_both_screens_false(tmp_path):
    path = tmp_path / 'map.json'
    path.write_text(json.dumps({
        'c0': {'material_id': 'm1', 'question': 'Who directed the film?', 'answerability': 1,
               'dual': True, 'text_suff': None, 'verdict': 'dual_necessary'},
        'c1': {'material_id': 'm1', 'question': 'Who directed the film?', 'answerability': 0,
               'dual': True, 'text_suff': None, 'verdict': 'dual_necessary'},
        'c2': {'material_id': 'm1', 'question': 'Who directed the film?', 'answerability': 1,
               'dual': False, 'text_suff': 'text_suff', 'verdict': 'single_sufficient'},
    }))
    result = _cascade_condition(path, {'m1': _material()})
    assert result['n'] == 3 and result['valid'] == 2
    assert result['dual_rate_all']['k'] == 1
    assert result['dual_rate_valid']['k'] == 1 and result['dual_rate_valid']['n'] == 2
    # the cascade field holds the verdict that SUCCEEDED, so `text_suff` = image redundant
    assert result['text_redundancy_valid']['k'] == 0 and result['text_redundancy_valid']['n'] == 2
    assert result['image_redundancy_valid']['k'] == 1 and result['image_redundancy_valid']['n'] == 2
