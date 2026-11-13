"""Portable inputs, fixed denominators, actual edits, answer isolation and resumability."""
import json
from pathlib import Path

import pytest

from egvqg.dataset import normalize_materials
from egvqg.providers import Completion, ProviderError, RequestBudgetExceeded, JudgeProvider
from egvqg.refinement import GATES, gate_question, normalized_question, refinement_metrics, run_refinement_pipeline, summarize_refinement
from egvqg.records import read_jsonl, write_jsonl

ROOT=Path(__file__).resolve().parents[1]
MATERIALS='tests/fixtures/stage_a/materials.jsonl'


@pytest.mark.parametrize('change,expected',[
    ({},'accepted'),({'text_needed':False},'rejected'),({'reference_status':'rejected'},'unresolved'),
    ({'image_needed':None},'unresolved'),({'validity':'invalid'},'unresolved'),({'error':'timeout'},'unresolved')])
def test_gate_distinguishes_failed_checks_from_unknown_reference(change,expected):
    check={'validity':'valid',**{k:True for k in GATES}}
    audit={**check,'reference_status':'verified',**change}
    assert gate_question(check,audit)[0]==expected


def test_portable_pipeline_fixed_cohort_actual_rewrite_and_resume(tmp_path):
    class Fake:
        model='offline';identity={'model':'offline'}
        def __init__(self,cap=1000):self.requests=[];self.cap=cap
        def complete(self,r):
            if len(self.requests)>=self.cap:raise RequestBudgetExceeded('offline budget')
            self.requests.append(r)
            if r.task=='generate_question':
                mid=r.metadata['material_id']
                assert 'REFERENCE_SECRET' not in r.prompt
                return Completion(json.dumps({'question':'Q '+mid}),self.model,
                                  finish_reason='length' if mid=='m_material' else 'stop')
            mid=r.question.split()[-1]
            if r.task=='refine_question':
                assert 'REFERENCE_SECRET' not in r.prompt
                if mid=='m_reagent':data={'status':'not_repairable','question':None,'reason':'no joint space'}
                elif mid=='m_invalid':data={'status':'revised','question':r.question,'reason':'claimed edit but identical'}
                else:data={'status':'revised','question':'FIXED '+mid,'reason':'uses missing fact'}
            else:
                if mid=='m_device':raise ProviderError('offline transport failure')
                good=mid=='m_thermo' or r.question.startswith('FIXED')
                data={'validity':'valid',**{k:True for k in GATES},'text_needed':good,
                      'reason':'REFERENCE_SECRET in evaluator reasoning','evidence_text':'a fact','evidence_image':'a fact',
                      'flags':[],'answer':'REFERENCE_SECRET','rubric':'REFERENCE_SECRET',
                      'reference_status':'verified','reference_reason':'verified for '+r.question}
                if r.task=='audit_question':assert 'REFERENCE_SECRET' in r.prompt
            return Completion(json.dumps(data),self.model)
    def roles(fake):return {k:fake for k in ('generator','checker','auditor','rewriter')}
    kwargs=dict(materials_path=MATERIALS,run_id='portable',project_root=ROOT,output_root=tmp_path,limit=6)
    with pytest.raises(RequestBudgetExceeded):run_refinement_pipeline(**kwargs,providers=roles(Fake(3)))
    partial=json.loads((tmp_path/'portable/metrics.json').read_text())
    assert partial['n_planned']==6 and partial['n_completed']==1 and partial['n_pending']==5
    spy=Fake();result=run_refinement_pipeline(**kwargs,providers=roles(spy))
    m=result['metrics']
    assert (m['n_planned'],m['n_initial_questions'],m['n_initial_survivors'],m['n_final_survivors'])==(6,5,1,2)
    assert m['n_rewrite_attempted']==3 and m['n_actually_changed']==1 and m['n_successfully_rewritten']==1
    assert m['rewrite_rate']==1/6 and m['rewrite_trigger_rate']==.5 and m['rewrite_success_rate']==1/3
    assert m['n_unresolved']==1 and m['n_pending']==0
    assert m['final_states']=={'accepted':2,'rejected':1,'no_question':1,'unresolved':1,'no_proposal':1}
    survivors=read_jsonl(tmp_path/'portable/survivors.jsonl')
    assert len(survivors)==2 and all('answer' not in r for r in survivors)
    assert all(r['run_id']=='portable' and 'protocol_version' in r and 'source_group_id' in r for r in survivors)
    assert len(read_jsonl(tmp_path/'portable/candidates.jsonl'))==9
    again=Fake();second=run_refinement_pipeline(**kwargs,providers=roles(again))
    assert second['new_calls']==second['new_records']==0 and not again.requests
    assert second['metrics']==m
    assert summarize_refinement(run_id='portable',project_root=ROOT,output_root=tmp_path)['metrics']==m
    assert normalized_question('  allele I?  ') == 'allele I?'
    assert normalized_question('allele I?') != normalized_question('allele i?')
    assert len(read_jsonl(tmp_path/'portable/initial_questions.jsonl'))==6
    assert len({r['material_id'] for r in read_jsonl(tmp_path/'portable/results.jsonl')})==6


def test_transfer_flat_dataset_preserves_source_and_supplied_q(tmp_path):
    raw=tmp_path/'foreign.jsonl'
    image=ROOT/'tests/fixtures/stage_a/images/m_thermo.png'
    write_jsonl(raw,[{'uid':'abc','context':'A passage','photo':str(image),'partition':'dev',
                     'gold':'MUST_NOT_COPY','solution':'MUST_NOT_COPY','question':'Supplied question'}])
    original=raw.read_bytes()
    out=tmp_path/'normalized.jsonl'
    qfile=tmp_path/'questions.jsonl'
    normalize_materials(input_path=raw,output_path=out,dataset_name='other_dataset',id_field='uid',
                        text_field='context',image_field='photo',split_field='partition',
                        question_field='question',questions_output=qfile)
    rows=read_jsonl(out);assert len(rows)==1 and rows[0]['split']=='dev'
    assert raw.read_bytes()==original and 'MUST_NOT_COPY' not in out.read_text()
    assert read_jsonl(qfile)[0]['material_id']==rows[0]['material_id']
    assert 'MUST_NOT_COPY' not in qfile.read_text()
    class Fake:
        model='offline';identity={'model':'offline'}
        def complete(self,r):
            assert r.task in ('check_question','audit_question')
            assert 'MUST_NOT_COPY' not in r.prompt and r.question=='Supplied question'
            data={'validity':'valid',**{k:True for k in GATES},'reason':'supported',
                  'evidence_text':'a fact','evidence_image':'a fact','flags':[],'answer':'new answer',
                  'rubric':'current question only','reference_status':'verified','reference_reason':'supported'}
            return Completion(json.dumps(data),self.model)
    roles={k:Fake() for k in ('generator','checker','auditor','rewriter')}
    result=run_refinement_pipeline(materials_path=str(out),questions_path=str(qfile),run_id='transfer',
                                  project_root=ROOT,output_root=tmp_path,providers=roles,limit=1)
    assert result['metrics']['n_final_survivors']==1 and result['metrics']['n_rewrite_attempted']==0
    assert read_jsonl(tmp_path/'transfer/initial_questions.jsonl')[0]['source'].startswith('other_dataset:')
    bad=tmp_path/'bad_questions.jsonl';write_jsonl(bad,[{'material_id':'wrong_dataset_id','question':'Q'}])
    with pytest.raises(ValueError,match='material_id is not in'):
        run_refinement_pipeline(materials_path=str(out),questions_path=str(bad),run_id='bad-transfer',
                                project_root=ROOT,output_root=tmp_path,providers=roles,limit=1)
    with pytest.raises(ValueError,match='overwrite raw'):
        normalize_materials(input_path=raw,output_path=raw,dataset_name='other',id_field='uid',text_field='context',image_field='photo')


def test_structured_actionable_feedback_does_not_copy_answers_or_rationale():
    from egvqg.refinement import rewrite_feedback, should_rewrite
    first={'validity':'valid',**{k:True for k in GATES},'repair_tags':['quantity_scope'],
           'answer':'SECRET_ANSWER','reason':'SECRET_ANSWER in reasoning'}
    second={**first,'reference_status':'verified'}
    decision,reason=gate_question(first,second)
    assert decision=='unresolved'
    assert should_rewrite({'decision':decision,'check':first,'audit':second})
    feedback=rewrite_feedback(first,second)
    assert feedback['repair_tags']==['quantity_scope'] and feedback['required_changes']
    assert 'SECRET_ANSWER' not in json.dumps(feedback)
    assert not should_rewrite({'decision':'unresolved','check':{'error':'timeout'},'audit':second})


def test_roles_can_use_independent_environment_settings_without_credentials_in_identity(tmp_path,monkeypatch):
    for key,value in {'MODEL_NAME':'requested-model','BASE_URL':'https://example.invalid/v1/chat/completions',
                      'API_KEY':'FIXTURE_CREDENTIAL','API':'responses','TIMEOUT':'12','MAX_RETRIES':'0'}.items():
        monkeypatch.setenv('FIXTURE_ROLE_'+key,value)
    provider=JudgeProvider.from_env(root=tmp_path,run_dir=tmp_path/'out',env_prefix='FIXTURE_ROLE',model='overridden-model',visual=True)
    assert provider.model=='overridden-model' and provider.timeout==12 and provider.max_attempts==1
    assert provider._url=='https://example.invalid/v1/responses'
    assert 'FIXTURE_CREDENTIAL' not in json.dumps(provider.identity)


def test_dry_run_needs_no_credentials_and_unknown_config_fields_are_not_snapshotted(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('dry run must not initialize a provider')
    monkeypatch.setattr(JudgeProvider,'from_env',forbidden)
    result=run_refinement_pipeline(materials_path=MATERIALS,run_id='dry',project_root=ROOT,
                                   output_root=tmp_path,limit=2,dry_run=True)
    assert result['n_planned']==2 and result['logical_calls_upper_without_retries']==12
    assert not (tmp_path/'dry').exists()
    cfg=json.loads((ROOT/'configs/refinement.json').read_text())
    cfg['roles']['checker']['api_key']='MUST_NOT_SAVE'
    path=tmp_path/'bad-config.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='each role'):
        run_refinement_pipeline(materials_path=MATERIALS,config_path=str(path),run_id='bad-config',project_root=ROOT,
                                output_root=tmp_path,limit=1,dry_run=True)
    assert not (tmp_path/'bad-config').exists()
