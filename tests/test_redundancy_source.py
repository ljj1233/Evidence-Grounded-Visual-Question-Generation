"""E1 redundancy-source: direction mapping, deduplication, and the programmatic screen."""
import json

from egvqg.redundancy_source import (combine_source, load_units, redundant_sides, screen_side,
                                     screen_unit)


def test_redundant_sides_follows_which_side_suffices():
    # `direction` names the side that alone suffices, so it is NOT the redundant side.
    assert redundant_sides('image') == ('text',)
    assert redundant_sides('text') == ('image',)
    assert redundant_sides('both') == ('text', 'image')
    assert redundant_sides(None) == ()


def test_combine_source_leak_wins_then_anchor_then_other():
    def sides(text, image):
        return {'text': {'source': text}, 'image': {'source': image}}

    assert combine_source('both', sides('anchor', 'leak')) == 'leak'
    assert combine_source('both', sides('anchor', 'anchor')) == 'anchor'
    assert combine_source('both', sides('anchor', 'other')) == 'other'
    assert combine_source('both', sides(None, 'leak')) == 'other'  # an unjudged side is never a leak
    # only the redundant side counts: a single-side leak on the retained side must not leak into it
    assert combine_source('image', sides('leak', None)) == 'leak'
    assert combine_source('text', sides(None, 'other')) == 'other'


def _write(tmp_path, labels, materials):
    (tmp_path / 'labels.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in labels))
    (tmp_path / 'materials.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in materials))


def test_load_units_dedups_and_keeps_undecided_out_of_the_redundant_count(tmp_path):
    materials = [{'material_id': 'm1', 'text': 'The value is 12.', 'image_path': 'i.jpg'}]
    labels = [
        {'material_id': 'm1', 'question': 'What is the value?',
         'label_consensus': {'verdict': 'text_suff'}},
        {'material_id': 'm1', 'question': 'What is the value?',
         'label_consensus': {'verdict': 'dual_necessary'}},  # duplicate question, different judge call
        {'material_id': 'm1', 'question': 'Which is larger?',
         'label_consensus': {'verdict': 'undetermined'}},
        {'material_id': 'missing', 'question': 'Gone?',
         'label_consensus': {'verdict': 'text_suff'}},
    ]
    _write(tmp_path, labels, materials)
    units, n_rows = load_units(kind='mmqa', labels_path='labels.jsonl', materials_path='materials.jsonl',
                              dataset='d', root=tmp_path)
    assert n_rows == 4                      # the raw row count stays the fixed denominator
    assert len(units) == 2                  # duplicates collapse to one unit
    by_question = {unit['question']: unit for unit in units}
    assert by_question['What is the value?']['duplicates'] == 2
    assert by_question['What is the value?']['direction'] == 'text'
    assert by_question['What is the value?']['redundant'] == 1
    undecided = by_question['Which is larger?']
    assert undecided['direction'] is None and undecided['redundant'] is None and undecided['decided'] is False


def test_screen_side_marks_text_copy_and_novel_image_number():
    text = 'The film was released on June 29, 2005 and directed by Joe.'
    copied = screen_side('When was the film released on June 29, 2005?', text, 'text')
    assert copied['source'] == 'leak'
    needed = screen_side('Who directed the film?', text, 'text')
    assert needed['source'] == 'anchor'
    # the number must come from the image: it is absent from the text
    assert screen_side('How many seats are shown?', text, 'image')['source'] == 'anchor'
    novel = screen_side('Which film earned 4200 million?', text, 'image')
    assert novel['source'] == 'leak' and novel['novel_numbers'] == ['4200']


def test_screen_unit_uses_only_the_redundant_side():
    text = 'Only items with value at least 6 count. Alpha is 12.'
    both = {'direction': 'image', 'question': 'Which items have a value of at least 6?', 'text': text}
    detail = screen_unit(both)
    assert set(detail) == {'text'} and detail['text']['source'] == 'leak'
    anchored = {'direction': 'text', 'question': 'How many items count?', 'text': text}
    assert screen_unit(anchored)['image']['source'] == 'anchor'
