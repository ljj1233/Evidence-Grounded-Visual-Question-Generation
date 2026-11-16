"""Inline detection: material numbers/rules copied into the generated question."""
from egvqg.studies.inline_analysis import contains_number, inline_features, lexical_inline, rule_inline


def test_number_inline_synthetic():
    text = 'Only items with value at least 14 count.'
    assert contains_number('Which items have a value of at least 14?', text) == ['14']
    assert contains_number('Which items count?', text) == []
    assert contains_number('Which items are at least fourteen?', text) == ['14']


def test_rule_inline_synthetic():
    text = 'Only items with value from 8 to 22 count.'
    assert rule_inline('Which items are from 8 to 22?', text, category='interval') == 'interval wording'
    assert rule_inline('Which items count?', text, category='interval') is None
    assert rule_inline('What is the value of alpha?', 'The value in the right column is called alpha.', category='bridging') == 'alpha'


def test_lexical_inline_mmqa():
    text = 'The film was released on June 29, 2005, and it was directed by Joe.'
    assert lexical_inline('When was the film released on June 29, 2005?', text) is not None
    assert lexical_inline('Who directed the film?', text) is None


def test_inline_features_any():
    text = 'Only items with value at least 9 count.'
    features = inline_features('Which items have a value of at least 9?', text, category='derived', use_lexical=False)
    assert features['inline_number'] and features['inline_rule'] and features['inline_any']
