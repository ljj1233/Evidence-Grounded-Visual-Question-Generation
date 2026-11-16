import json

from egvqg.studies.composition_grid import (CELLS, _item_level, _rescore, build_grid_materials, score_items)


def _materials(tmp_path, n=2):
    build_grid_materials(project_root=tmp_path, output_dir='grid', n_materials=n, seed=7)
    path = tmp_path / 'grid' / 'materials.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_build_grid_places_each_fact_in_the_declared_modality(tmp_path):
    rows = _materials(tmp_path, n=2)
    assert len(rows) == 2 * len(CELLS)
    by_material = {}
    for row in rows:
        by_material.setdefault(row['material_id'], {})[row['cell']] = row
    for cells in by_material.values():
        assert set(cells) == set(CELLS)
        # the same values and threshold are used in every cell: the cells are paired
        assert len({tuple(cell['values']) for cell in cells.values()}) == 1
        assert len({cell['threshold'] for cell in cells.values()}) == 1
        tt, ti, it, ii = (cells[name] for name in CELLS)
        assert tt['image_path'] is None and 'item values' in tt['text'] and 'count' in tt['text']
        assert ti['image_path'].endswith('-values.png') and 'item values' not in ti['text']
        assert it['image_path'].endswith('-rule.png') and 'item values' in it['text']
        assert ii['text'] == '' and ii['image_path'].endswith('-both.png')


def test_material_expectations_follow_the_rule(tmp_path):
    for row in _materials(tmp_path, n=3):
        values, threshold = row['values'], row['threshold']
        counting = [chr(65 + i) for i, value in enumerate(values) if value >= threshold]
        assert row['evaluation']['full']['expected'] == counting
        assert row['evaluation']['count']['expected'] == len(counting)
        assert row['evaluation']['sum']['expected'] == sum(v for v in values if v >= threshold)
        assert row['evaluation']['rule']['expected'] == threshold


def test_score_items_is_order_insensitive_and_handles_quantifiers():
    assert score_items('{"answer":"C,D","abstain":false}', ['C', 'D'])[0] is True
    assert score_items('{"answer":"A, C, and D","abstain":false}', ['A', 'C', 'D'])[0] is True
    assert score_items('{"answer":"C, D","abstain":false}', ['A', 'C', 'D'])[0] is False
    assert score_items('{"answer":"all of them","abstain":false}', ['A', 'B', 'C', 'D'])[0] is True
    assert score_items('{"answer":"none","abstain":false}', [])[0] is True
    assert score_items('{"abstain":true}', ['A'])[0] is False


def test_rescore_recomputes_from_the_raw_output():
    row = {'raw_output': '{"answer":"C,D","abstain":false}', 'answer_kind': 'items', 'expected': ['C', 'D'],
           'correct': False}
    assert _rescore(row)['correct'] is True
    numeric = {'raw_output': '{"answer":"4","abstain":false}', 'answer_kind': 'numeric', 'expected': 4,
               'correct': False}
    assert _rescore(numeric)['correct'] is True


def test_item_level_separates_missing_and_extra_items():
    materials = [{'material_id': 'm', 'cell': 'TT', 'values': [10, 2, 12, 4], 'threshold': 5}]
    answers = [
        {'material_id': 'm', 'cell': 'TT', 'question_type': 'full', 'expected': ['A', 'C'],
         'parsed_answer': ['C'], 'correct': False},
        {'material_id': 'm', 'cell': 'TT', 'question_type': 'sum', 'expected': 22, 'parsed_answer': [10, 12],
         'correct': False},
    ]
    level = _item_level(answers, materials)
    assert level['error_shape'] == {'missed_one_A': 1}
    assert level['sum_error_shape'] == {'listed_values_not_summed': 1}
    # items below the threshold sit in bucket 0; both counting items are 4+ above it
    assert level['by_margin']['0']['n'] == 2 and level['by_margin']['4+']['n'] == 2
    assert level['by_margin']['0']['k'] == 2
    # the signed view keeps the two non-counting items apart: value 2 is -3, value 4 is -1
    assert level['by_signed_margin']['-3']['n'] == 1 and level['by_signed_margin']['-1']['n'] == 1
    # the omitted counting item (A, value 10) is a miss, not a boundary reversal
    assert level['signed_margin_direction']['-1'] == {'missed': 0, 'extra': 0, 'correct': 1}
    assert level['signed_margin_direction']['>=+4'] == {'missed': 1, 'extra': 0, 'correct': 1}
    assert level['signed_margin_by_cell']['TT']['-1']['n'] == 1
