"""Local validity parsing and the programmatic well-formedness fallback."""
from egvqg.validity_check import _parse_valid, _programmatic_valid


def test_parse_valid():
    assert _parse_valid('{"valid":1,"reason":"clear"}')[0] == 1
    assert _parse_valid('```json\n{"valid":0,"reason":"ambiguous"}\n```')[0] == 0
    assert _parse_valid('not json')[0] is None
    assert _parse_valid('{"valid":true}')[0] is None


def test_programmatic_valid():
    assert _programmatic_valid('Which items have a value of at least six?')
    assert not _programmatic_valid('Which items')
    assert not _programmatic_valid('Output only the question?')
