"""Faithfulness vs reference failure, fixed N, source binding and offline resume."""
import copy
import json
from pathlib import Path

import pytest

from egvqg.faithfulness import CHECKS, apply_review, derive_faithfulness, review_binding
from egvqg.metrics import record_states
from egvqg.records import read_jsonl, sha256_file, sha256_text, write_json, write_jsonl
from test_experiments import _record

ROOT = Path(__file__).resolve().parents[1]


def fixture_review(row, *, gate=1, status='reviewed', reference=1):
    return {'attempt_id': row['attempt_id'], 'item_id': row['attempt_id'], **review_binding(row),
        'knowledge_sha256': sha256_text('fixture K'), 'source': 'user_delegated_ai',
        'user_authorized': True, 'authorization': 'offline fixture', 'reviewer': 'test',
        'assessment_status': status, 'question_checks': {
            k: {'value': gate if k == 'answer_supported' else 1, 'reason': 'fixture evidence'} for k in CHECKS},
        'reference_supported': {'value': reference, 'reason': 'fixture reference review'},
        'reason': 'explicit fixture review', 'failure_codes': [],
        'evidence': [{'source': 'T', 'quote_or_location': 'fixture text', 'explanation': 'fixture support'}]}


def fixture_record(tmp_path, index=0, v=1):
    row = _record(index, v=v)
    row['labels']['adjudicated'] = copy.deepcopy(row['labels']['automatic'])
    row['shortcuts']['adjudicated'] = copy.deepcopy(row['shortcuts']['automatic'])
    row['reference']['status'] = 'verified'
    image = tmp_path/'image.png'; image.write_bytes(b'fixture image; never submitted to model')
    row['material'] = {'text': 'fixture text', 'text_sha256': sha256_text('fixture text'),
                       'image_path': 'image.png', 'image_sha256': sha256_file(image)}
    return row


@pytest.mark.parametrize('old,gate,status,ref,expected', [
    (None,0,'reviewed',0,0),       # missing decisive premise is a generation failure
    (1,1,'reviewed',0,1),          # wrong reference does not prove a bad Q
    (1,1,'reviewed',None,1),       # unavailable reference also does not prove a bad Q
    (None,None,'technical_failure',None,None),
    (None,None,'evaluation_uncertain',None,None),
])
def test_faithfulness_failure_and_uncertainty_are_distinct(tmp_path, old, gate, status, ref, expected):
    row = fixture_record(tmp_path, v=old); original = copy.deepcopy(row)
    result = apply_review(row, fixture_review(row, gate=gate, status=status, reference=ref))
    assert result['labels']['adjudicated']['validity']['value'] == expected
    assert row == original and result['responses'] == row['responses']
    assert result['labels']['adjudicated']['relatedness']['value'] == 1
    if expected == 0:
        assert record_states(result, 'adjudicated')['bdr'] == 0
        assert all(result['labels']['adjudicated'][k]['status'] == 'not_applicable'
                   for k in ('text_necessary','image_necessary','integration'))
    if ref == 0:
        assert result['reference']['verified_answer'] == row['reference']['verified_answer']
        assert result['reference']['status'] == 'unresolved'
        assert result['faithfulness']['source_reference'] == row['reference']
        if expected == 1:
            assert record_states(result, 'adjudicated')['bdr'] is None


def test_full_denominator_read_only_sources_and_resume(tmp_path):
    (tmp_path/'configs').mkdir(); write_json(tmp_path/'configs/necessity_assessment_v2.json', {'knowledge':'fixture K'})
    (tmp_path/'prompts').mkdir()
    (tmp_path/'prompts/review-question-faithfulness-v1.txt').write_bytes((ROOT/'prompts/review-question-faithfulness-v1.txt').read_bytes())
    rows = [fixture_record(tmp_path,i,v=None if i == 1 else 1) for i in range(4)]
    rows[2]['labels']['adjudicated']['text_necessary'].update(value=None, status='undecided')
    rows[3]['labels']['adjudicated']['image_necessary'].update(value=0, status='decided')
    source=tmp_path/'outputs/source'; source.mkdir(parents=True)
    write_jsonl(source/'records.jsonl', rows)
    reviews=[fixture_review(rows[0]), fixture_review(rows[1],gate=0),
             fixture_review(rows[2]), fixture_review(rows[3])]
    path=tmp_path/'review.jsonl'; write_jsonl(path,reviews)
    original=sha256_file(source/'records.jsonl')
    args=dict(source_run='source',run_id='derived',reviews_path='review.jsonl',project_root=tmp_path)
    result=derive_faithfulness(**args)
    assert result['n_planned']==4 and result['new_model_calls']==0
    assert result['answerability']=={'pass':3,'fail':1,'unresolved':0}
    assert result['modality_necessity']=={'both_required':1,'not_both_required':1,
                                         'unresolved':1,'not_applicable':1}
    assert result['target_acceptance']=={'pass':1,'fail':2,'unresolved':1}
    assert len(read_jsonl(tmp_path/'outputs/derived/records.jsonl'))==4
    assert len(read_jsonl(tmp_path/'outputs/derived/assessment_table.jsonl'))==4
    out=tmp_path/'outputs/derived'; before={p.name:sha256_file(p) for p in out.iterdir() if p.is_file()}
    assert derive_faithfulness(**args)['resumed']
    assert before=={p.name:sha256_file(p) for p in out.iterdir() if p.is_file()}
    assert sha256_file(source/'records.jsonl')==original
    for key in ('question_sha256','material_sha256','reference_sha256','knowledge_sha256'):
        bad=copy.deepcopy(reviews);bad[0][key]='stale';write_jsonl(path,bad)
        with pytest.raises(ValueError, match='binding'):
            derive_faithfulness(**args)
    write_jsonl(path,reviews[:-1])
    with pytest.raises(ValueError, match='every unique planned slot'):
        derive_faithfulness(**args)
    write_jsonl(path,reviews)
    (tmp_path/'image.png').write_bytes(b'changed actual image')
    with pytest.raises(ValueError, match='source question/image changed'):
        derive_faithfulness(**args)
