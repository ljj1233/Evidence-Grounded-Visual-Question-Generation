from vvqg.vc_qse.concepts import flatten, parse_concepts
from vvqg.vc_qse.selector import select_image


def test_parse_concepts_object_form():
    raw = ('{"objects":[{"text":"dog","confidence":0.9}],'
           '"attributes":[{"text":"brown","confidence":0.5}],'
           '"relations":[],"scene":[]}')
    concepts, status, _ = parse_concepts(raw)
    assert status == "ok"
    assert flatten(concepts) == [
        {"concept_type": "object", "concept_text": "dog", "concept_confidence": 0.9},
        {"concept_type": "attribute", "concept_text": "brown", "concept_confidence": 0.5},
    ]


def test_parse_concepts_caps_dedup_and_rejects_questions():
    raw = ('{"objects":[{"text":"cat","confidence":1},{"text":"cat","confidence":1},'
           '{"text":"is it a cat","confidence":1},{"text":"a"},{"text":"b"},{"text":"c"}],'
           '"attributes":[],"relations":[],"scene":[]}')
    concepts, status, _ = parse_concepts(raw)
    assert status == "ok"
    texts = [c["text"] for c in concepts["objects"]]
    assert texts == ["cat", "a", "b", "c"]


def test_parse_concepts_failure_and_string_form():
    assert parse_concepts("not json")[1] == "parser_failure"
    concepts, status, _ = parse_concepts('{"objects":["dog"],"attributes":[],"relations":[],"scene":[]}')
    assert status == "ok" and concepts["objects"][0]["confidence"] == 0.5


def _row(iid, gen, ctype, conf, state, adh="yes"):
    return {"image_id": iid, "generation_id": gen, "concept_type": ctype, "concept_text": ctype,
            "concept_confidence": conf, "flat_state": state, "concept_adherence": adh,
            "question": f"{ctype} question", "image": "x.jpg"}


def test_selector_drops_unsupported_and_adherence_no():
    rows = [_row("1", 0, "object", 0.9, "CONFIRMED_VIOLATION"),
            _row("1", 1, "attribute", 0.9, "NO_CONFIRMED_VIOLATION", adh="no"),
            _row("1", 2, "relation", 0.7, "NO_CONFIRMED_VIOLATION")]
    result = select_image(rows)
    assert not result["fallback"]
    assert result["final"]["concept_type"] == "relation"


def test_selector_prefers_supported_and_one_per_type():
    rows = [_row("1", 0, "object", 0.5, "UNCERTAIN"),
            _row("1", 1, "object", 0.6, "NO_CONFIRMED_VIOLATION"),
            _row("1", 2, "object", 0.9, "NO_CONFIRMED_VIOLATION")]
    result = select_image(rows)
    assert len(result["selected"]) == 1
    assert result["final"]["concept_confidence"] == 0.9


def test_selector_fallback():
    rows = [_row("1", 0, "object", 0.9, "CONFIRMED_VIOLATION")]
    fallback = {"question": "direct4?", "flat_state": "NO_CONFIRMED_VIOLATION"}
    result = select_image(rows, direct4_fallback=fallback)
    assert result["fallback"] and result["final"] is fallback
