"""Synthetic composition benchmark: programmatic scoring, hop consistency, grammar labels, bootstrap."""
import json

from egvqg.studies.synthetic_composition import (_bootstrap_gap, _metrics, build_synthetic_composition,
                                         classify_generated_question, score_answer)


def test_build_has_consistent_single_hops(tmp_path):
    result = build_synthetic_composition(project_root=tmp_path, output_dir='data', per_category=4)
    rows = [json.loads(line) for line in (tmp_path / 'data' / 'materials.jsonl').read_text().splitlines()]
    assert result['n_materials'] == 12
    categories = {row['category'] for row in rows}
    assert categories == {'derived', 'bridging', 'stitching'}
    for row in rows:
        assert (tmp_path / row['image_path']).exists()
        assert set(row['evaluation']) == {'full', 'text_hop', 'image_hop'}
        if row['category'] == 'derived':
            text_hop = row['evaluation']['text_hop']['expected']
            values = row['evaluation']['image_hop']['expected']
            assert row['evaluation']['full']['expected'] == sum(v for v in values if v >= text_hop)
            assert sum(v >= text_hop for v in values) in (2, 3)
            assert row['evaluation']['full']['expected'] != text_hop
        if row['category'] == 'bridging':
            column = row['evaluation']['text_hop']['expected']
            expected = row['values'][('left', 'middle', 'right').index(column)]
            assert row['evaluation']['full']['expected'] == expected == row['evaluation']['image_hop']['expected']
        if row['category'] == 'stitching':
            threshold = row['evaluation']['text_hop']['expected']
            value_c = row['evaluation']['image_hop']['expected']
            assert threshold != value_c
            assert row['evaluation']['full']['expected'] == [threshold, value_c]


def test_programmatic_scoring_contract():
    assert score_answer('{"answer":"11","abstain":false}', 'numeric', 11)[0]
    assert score_answer('```json\n{"answer": 11}\n```', 'numeric', 11)[0]
    assert not score_answer('{"answer":null,"abstain":false}', 'numeric', 11)[0]
    assert not score_answer('{"answer":"11","abstain":true}', 'numeric', 11)[0]
    assert score_answer('{"answer":"A: 6, B: 2, C: 5"}', 'item_values', [6, 2, 5])[0]
    assert score_answer('{"answer":"6,2,5"}', 'item_values', [6, 2, 5])[0]
    assert not score_answer('{"answer":"6,2"}', 'item_values', [6, 2, 5])[0]
    assert score_answer('{"answer":"threshold 5 and item C 4"}', 'both_numbers', [5, 4])[0]
    assert not score_answer('{"answer":"5"}', 'both_numbers', [5, 4])[0]
    assert score_answer('{"answer":"the right column"}', 'column', 'right')[0]
    assert score_answer('{"answer":"third"}', 'column', 'right')[0]
    assert not score_answer('{"answer":"middle"}', 'column', 'right')[0]


def test_grammar_labels_canonical_questions():
    material = {'category': 'derived'}
    assert classify_generated_question('What is the total value of the items that count?', material) == 'dual_derived'
    assert classify_generated_question('What is the value of alpha?', material) == 'dual_bridging'
    assert classify_generated_question('What is the rule threshold, and what is the value of item C?', material) == 'dual_stitching'
    assert classify_generated_question('What is the minimum value an item must have to count?', material) == 'single_text'
    assert classify_generated_question('Which items have a value of at least 5?', material) == 'single_image'
    assert classify_generated_question('Which items have a value of at least six?', material) == 'single_image'
    assert classify_generated_question('What items meet the criteria of having a value at least 4?', material) == 'single_image'
    assert classify_generated_question('Which items count?', material) == 'dual_derived'
    assert classify_generated_question('How many items count?', material) == 'dual_derived'
    assert classify_generated_question('Which items satisfy the given condition?', material) == 'dual_derived'
    assert classify_generated_question('Which items satisfy the stated rule?', material) == 'dual_derived'
    assert classify_generated_question('What is the value of item C?', material) == 'single_image'
    assert classify_generated_question('Why is the table useful?', material) == 'other'


def test_gap_bootstrap_and_stop_rule_inputs():
    stats = [
        {'full': False, 'text_hop': True, 'image_hop': True},
        {'full': False, 'text_hop': True, 'image_hop': True},
        {'full': True, 'text_hop': True, 'image_hop': True},
        {'full': False, 'text_hop': False, 'image_hop': True},
    ]
    metrics = _metrics(stats)
    assert metrics['acc_full'] == 0.25
    assert metrics['product'] == 0.75
    assert metrics['both_hops_correct'] == 3
    assert metrics['both_hops_full_wrong'] == 2
    boot = _bootstrap_gap(stats, B=200, seed=1)
    assert boot['gap_ci_95'][0] <= metrics['gap'] <= boot['gap_ci_95'][1]

def test_count_variant_isolates_set_selection(tmp_path):
    build_synthetic_composition(project_root=tmp_path, output_dir='data', per_category=5, derived_full='count')
    rows = [json.loads(line) for line in (tmp_path / 'data' / 'materials.jsonl').read_text().splitlines()]
    derived = [row for row in rows if row['category'] == 'derived']
    assert derived
    for row in derived:
        threshold = row['threshold']
        assert threshold not in row['values']
        count = sum(v >= threshold for v in row['values'])
        assert count in (2, 3)
        assert row['evaluation']['full']['expected'] == count
        assert row['q'] == 'How many items count?'
        assert row['derived_full'] == 'count'

