import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from vvqg.rewards.support.calibration import (
    LABELS, aggregate_requirements, overall_support, parse_labels, select_pairs, split_pairs,
)
from vvqg.evaluation.metrics.visual_support import classification_metrics, paired_comparison


def test_all_component_combinations():
    for p, r, a in itertools.product(*(LABELS[k] for k in ('premise_consistency', 'reference_clarity', 'visual_answerability'))):
        expected = 'unsupported' if p == 'unsupported' or r == 'ambiguous' or a == 'unanswerable' else (
            'supported' if p == 'supported' and r in ('clear', 'not_applicable') and a == 'answerable' else 'uncertain')
        assert overall_support(p, r, a) == expected


def test_no_and_zero_do_not_require_existence():
    for target in ('whether a dog is visible', 'count of visible dogs'):
        extracted = {'presuppositions': [], 'reference_constraints': [], 'query_target': {'type': 'target', 'target': target}}
        assert aggregate_requirements(extracted, ['answerable']) == {
            'premise_consistency': 'supported', 'reference_clarity': 'not_applicable',
            'visual_answerability': 'answerable', 'overall_visual_support': 'supported'}


def test_definite_failure_overrides_uncertainty():
    extracted = {'presuppositions': [{'type': 'existence', 'target': 'dog'}],
                 'reference_constraints': [{'type': 'reference', 'target': 'the dog'}],
                 'query_target': {'type': 'target', 'target': 'color'}}
    assert aggregate_requirements(extracted, ['uncertain', 'ambiguous', 'answerable'])['overall_visual_support'] == 'unsupported'
    assert aggregate_requirements(extracted, ['uncertain', 'clear', 'answerable'])['overall_visual_support'] == 'uncertain'


def test_external_json_vocabulary_is_checked():
    with pytest.raises(ValueError):
        parse_labels('{"premise_consistency":"yes","reference_clarity":"clear","visual_answerability":"answerable","failure_type":"none","reason":"visible"}')


def test_selection_and_split_are_deterministic_disjoint():
    rows = [{'image_id': str(i), 'image': f'{i}.jpg', 'question': f'{i}-{g}', 'generation_id': g, 'source': 'final_sft'}
            for i in range(1000) for g in range(4)]
    pairs = select_pairs(rows[::-1])
    assert len(pairs) == len({x['image_id'] for x in pairs}) == 1000
    assert [sum(r['generation_id'] == g for r in pairs) for g in range(4)] == [250] * 4
    assert pairs == select_pairs(rows)
    splits = split_pairs(pairs)
    assert [len(splits[k]) for k in ('dev', 'val', 'test')] == [600, 200, 200]
    sets = [set(r['image_id'] for r in s) for s in splits.values()]
    assert not any(a & b for a, b in itertools.combinations(sets, 2))
    assert splits == split_pairs(pairs)


def test_metrics_preserve_uncertain_and_absent_classes():
    m = classification_metrics(['supported', 'unsupported', 'uncertain'], ['supported', 'supported', 'uncertain'], LABELS['overall_visual_support'])
    assert m['accuracy'] == pytest.approx(2/3)
    assert m['balanced_accuracy'] == pytest.approx(2/3)
    assert m['macro_f1'] == pytest.approx(5/9)
    assert m['confusion_matrix'] == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    m = classification_metrics(['supported'], ['supported'], LABELS['overall_visual_support'])
    assert m['per_class']['unsupported']['recall'] is None


def test_paired_bootstrap_identical_systems():
    y = np.array([0, 1, 2, 0])
    result = paired_comparison(y, y, y, 3, resamples=50)
    for metric in result.values():
        assert metric['difference'] == metric['ci95'][0] == metric['ci95'][1] == 0
