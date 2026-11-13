"""Quality labels versus answer correctness; actual multimodal payload isolation and resume."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from egvqg.evaluation import build_condition_request
from egvqg.pipeline import RunConfigMismatch
from egvqg.prompts import PromptSet
from egvqg.providers import Completion, JudgeProvider, RequestBudgetExceeded
from egvqg.quality import DIMENSIONS, _check_assessment, diagnose_quality, quality_label
from egvqg.records import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = 'tests/fixtures/stage_a/materials.jsonl'


@pytest.mark.parametrize('changes,reference,expected', [
    ({}, 'verified', 'high'),
    ({'text_needed': False}, 'verified', 'valid_non_target'),
    ({'integrated': False}, 'verified', 'valid_non_target'),
    ({'validity': 'invalid'}, 'unresolved', 'invalid'),
    ({}, 'unresolved', 'unresolved'),
    ({'image_needed': None}, 'verified', 'unresolved'),
    ({'validity': 'unresolved'}, 'verified', 'unresolved'),
])
def test_quality_is_not_answer_accuracy(changes, reference, expected):
    assessment = {'validity': 'valid', **{k: True for k in DIMENSIONS}, **changes}
    assert quality_label(assessment, reference) == expected
    with pytest.raises(ValueError):
        _check_assessment({**assessment, 'text_needed': 'true'})


@pytest.mark.parametrize('api', ['responses', 'chat_completions'])
def test_actual_image_wire_isolation_and_image_cache_invalidation(tmp_path, monkeypatch, api):
    material = read_jsonl(ROOT/MATERIALS)[0]
    image = tmp_path/'image.png'; image.write_bytes((ROOT/material['image_path']).read_bytes())
    payloads = []
    def post(self, payload):
        payloads.append(payload)
        answer = '{"status":"answered","answer":"test"}'
        usage = {'input_tokens': 5, 'output_tokens': 2, 'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7}
        if api == 'responses':
            return {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': answer}]}], 'usage': usage}
        return {'choices': [{'message': {'content': answer}, 'finish_reason': 'stop'}], 'usage': usage}
    monkeypatch.setattr(JudgeProvider, '_post_json', post)
    provider = JudgeProvider(model='test', base_url='https://example.invalid/v1', api_key='SECRET',
                             cache_dir=tmp_path/'cache', ledger_path=tmp_path/'ledger.jsonl', visual=True, api=api)
    for condition in ('TI','T','I','0'):
        request = build_condition_request(PromptSet(ROOT/'prompts'), material=material, question='CURRENT QUESTION', condition=condition, image_path=image)
        completion = provider.complete(request)
        body = payloads[-1]
        serialized = json.dumps(body, ensure_ascii=False)
        assert (material['text'] in serialized.replace('\\n','\n')) == (condition in ('TI','T'))
        assert ('data:image/png;base64,' in serialized) == (condition in ('TI','I'))
        assert 'reference_answer' not in serialized and material['material_id'] not in serialized
        assert completion.usage['input_audit']['image_transmitted'] == (condition in ('TI','I'))
        assert provider.complete(request).usage['cache_hit'] is True
    assert len(payloads) == 4
    image.write_bytes(image.read_bytes()+b'changed')
    request = build_condition_request(PromptSet(ROOT/'prompts'), material=material, question='CURRENT QUESTION', condition='I', image_path=image)
    assert provider.complete(request).usage['cache_hit'] is False
    assert len(payloads) == 5


def test_diagnostic_budget_resume_blind_inputs_and_fixed_denominator(tmp_path):
    cases = [dict(case_id='c1', material_id='m_thermo', question='valid question', expected_label='high',
                  expected_reason='EXPECTED_CANARY', reference_answer='REFERENCE_CANARY', rubric='a result',
                  expected_label_source='AI_authored_before_model_evaluation',human_review_status='pending'),
             dict(case_id='c2', material_id='m_thermo', question='invalid question', expected_label='invalid',
                  expected_reason='EXPECTED_CANARY', reference_answer=None, rubric=None,
                  expected_label_source='AI_authored_before_model_evaluation',human_review_status='pending')]
    file=tmp_path/'cases.jsonl';write_jsonl(file,cases)
    class Fake:
        model='offline-test'
        identity={'adapter':'offline-test'}
        def __init__(self, cap=100): self.requests=[];self.cap=cap
        def complete(self, request):
            if len(self.requests) >= self.cap: raise RequestBudgetExceeded('test budget')
            self.requests.append(request)
            assert 'EXPECTED_CANARY' not in request.prompt
            if request.task in ('assess_quality','answer_question','probe_image'):
                assert 'REFERENCE_CANARY' not in request.prompt
            if request.task=='probe_image': data={'visible_text':[], 'description':'fixture'}
            elif request.task=='assess_quality':
                data={'validity':'invalid' if request.question.startswith('invalid') else 'valid',
                      **{k: True for k in DIMENSIONS},'reason':'fixture','evidence_text':'fixture','evidence_image':'fixture','flags':[]}
            elif request.task=='verify_reference': data={'status':'verified','reason':'fixture','evidence_text':'fixture','evidence_image':'fixture'}
            elif request.task=='answer_question': data={'status':'answered','answer':'a response'}
            else: data={'status':'scored','correct':True,'rationale':'fixture'}
            return Completion(json.dumps(data),self.model)
    kwargs=dict(materials_path=MATERIALS,cases_path=str(file),run_id='quality',project_root=ROOT,output_root=tmp_path,limit=1)
    first=Fake(3)
    with pytest.raises(RequestBudgetExceeded):diagnose_quality(**kwargs,provider=first)
    resumed=Fake()
    result=diagnose_quality(**kwargs,provider=resumed)
    assert resumed.requests[0].task=='answer_question'
    assert result['quality']['n_planned']==2 and result['quality']['n_agree']==2
    assert len(read_jsonl(tmp_path/'quality/responses.jsonl'))==8
    assert len(read_jsonl(tmp_path/'quality/scores.jsonl'))==8
    again=Fake()
    repeat=diagnose_quality(**kwargs,provider=again)
    assert repeat['new_calls']==repeat['new_records']==0 and again.requests==[]
    cases[0]['reference_answer']='changed';write_jsonl(file,cases)
    with pytest.raises(RunConfigMismatch):diagnose_quality(**kwargs,provider=Fake())


def test_checklist_repair_preserves_failed_attempt_and_blind_checks(tmp_path):
    from egvqg.quality import refine_questions
    file=tmp_path/'repairs.jsonl'
    write_jsonl(file,[{'case_id':str(i),'material_id':'m_thermo','question':f'original {i}',
                      'feedback':'FEEDBACK_CANARY'} for i in range(2)])
    class Fake:
        model='test';identity={'adapter':'test'}
        def __init__(self,cap=100):self.requests=[];self.cap=cap
        def complete(self,r):
            if len(self.requests)>=self.cap:raise RequestBudgetExceeded('budget')
            self.requests.append(r)
            if r.task!='refine_question':assert 'FEEDBACK_CANARY' not in r.prompt
            if r.task=='refine_question':
                assert 'a candidate' not in r.prompt  # Proposed reference answers are not rewriter inputs.
                data=({'status':'revised','question':'revised question','reason':'uses the missing fact'}
                      if r.question=='original 0' else {'status':'not_repairable','question':None,'reason':'materials redundant'})
            elif r.task=='check_question':
                data={'validity':'invalid' if r.question=='original 1' else 'valid',**{k:True for k in DIMENSIONS},'text_needed':r.question=='revised question',
                      'no_leakage':True,'evidence_text':'a fact','evidence_image':'a fact','flags':[],
                      'reason':'checked','shortcut_check':'checked','text_change_effect':'checked','image_change_effect':'checked',
                      'answer':'a candidate','rubric':'required points'}
            else:data={'status':'verified','reason':'supported','evidence_text':'a fact','evidence_image':'a fact'}
            return Completion(json.dumps(data),self.model)
    kwargs=dict(materials_path=MATERIALS,cases_path=str(file),run_id='repair',project_root=ROOT,output_root=tmp_path)
    with pytest.raises(RequestBudgetExceeded):refine_questions(**kwargs,provider=Fake(1))
    provider=Fake();result=refine_questions(**kwargs,provider=provider)
    assert provider.requests[0].task=='verify_reference'
    assert result['metrics']['n_planned_repairs']==2
    assert result['metrics']['after_counts']=={'high':1,'no_proposal':1}
    assert result['metrics']['n_improved_to_automatic_high']==1
    assert len(read_jsonl(tmp_path/'repair/repairs.jsonl'))==2
    assert len(read_jsonl(tmp_path/'repair/checks.jsonl'))==3
    assert read_jsonl(tmp_path/'repair/results.jsonl')[1]['before']['label']=='unresolved'
    repeat=Fake();r=refine_questions(**kwargs,provider=repeat)
    assert r['new_calls']==r['new_records']==0 and repeat.requests==[]
    externally_edited=read_jsonl(file)
    for i,c in enumerate(externally_edited):
        c['proposed_revision']={'status':'revised' if i==0 else 'not_repairable',
                               'question':'revised question' if i==0 else None,
                               'reason':'documented edit','source':'test AI editor, not human'}
    external=tmp_path/'external_cases.jsonl';write_jsonl(external,externally_edited)
    spy=Fake()
    edited=refine_questions(**{**kwargs,'cases_path':str(external),'run_id':'external'},provider=spy)
    assert not any(req.task=='refine_question' for req in spy.requests)
    assert edited['metrics']['n_planned_repairs']==2 and edited['metrics']['n_no_proposal']==1
    assert {r['source'] for r in read_jsonl(tmp_path/'external/repairs.jsonl')}=={'test AI editor, not human'}
