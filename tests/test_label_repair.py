"""Label repair: parent selection must not skip parents when it round-robins across skills."""

from egvqg.assess.label_repair import select_parents


def _rows(skill, parents, valid=1):
    return [{'skill': skill, 'parent_id': p, 'valid': valid} for p in parents]


def test_select_parents_round_robins_without_skipping():
    rows = _rows('a', [f'a{i}' for i in range(5)]) + _rows('b', [f'b{i}' for i in range(2)]) + _rows('c', ['c0'], valid=0)
    assert select_parents(rows, 3) == {'a0', 'b0', 'a1'}
    assert select_parents(rows, 100) == {'a0', 'a1', 'a2', 'a3', 'a4', 'b0', 'b1'}


def test_select_parents_excludes_parents_with_only_invalid_rows():
    rows = _rows('a', ['a0', 'a1']) + _rows('b', ['b0'], valid=0)
    assert select_parents(rows, 100) == {'a0', 'a1'}
