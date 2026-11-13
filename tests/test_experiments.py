"""Fixed-N protocol regressions: manual arithmetic, isolation, crashes and provenance."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from egvqg.experiments import DIMENSIONS, SHORTCUTS, TEMPLATES, init_experiment, label, new_record, run_experiment
from egvqg.metrics import compute_record_metrics, record_states, summarize_records
from egvqg.prompts import PromptSet
from egvqg.providers import Completion, ModelCaller, ProviderError
from egvqg.records import (ExperimentStore, atomic_upsert, canonical_json, read_jsonl,
                          sha256_file, sha256_text, write_json, write_jsonl)

ROOT = Path(__file__).resolve().parents[1]


def _record(index=0, v=1, scores=(1, 0, 0, 0), generation='ok', verified=True):
    r = new_record({'attempt_id': f'a{index}', 'run_id': 'unit', 'method': 'plain', 'plan_index': index},
        {}, {'protocol_version': 'three-phase-prompt-v1'}, 'prompts', 'Question?')
    r['generation']['status'] = generation
    if generation != 'ok':
        r['question'].update(normalized_text=None, sha256=None, question_id=None)
        r['reference'].update(question_id=None, question_sha256=None)
    r['labels']['automatic'] = {k: label(v if k == 'validity' else 1, 'synthetic') for k in DIMENSIONS}
    r['shortcuts']['automatic'] = {k: label(0, 'synthetic') for k in SHORTCUTS}
    ref = r['reference']; ref.update(status='verified_automatic' if verified else 'unresolved',
                                      verified_answer='yes', acceptable_answers=['yes'], rubric=['yes'])
    basis = sha256_text(canonical_json({k: ref[k] for k in ('verified_answer','acceptable_answers','rubric')}))
    for c, value in zip(('TI','T','I','0'), scores):
        if generation != 'ok': continue
        r['responses'][c] = {'question_id':r['question']['question_id'], 'question_sha256':r['question']['sha256'],
            'condition':c, 'status':'answered', 'abstain':False, 'correct':value,
            'score_status':'scored' if value is not None else 'unresolved', 'score_reference_sha256':basis}
    return r


def test_manual_fixed_n_bounds_and_human_never_inherit_automatic(tmp_path):
    rows = [_record(0, scores=(1,0,0,1)), _record(1, v=None, scores=(1,1,0,1)),
            _record(2, v=0), _record(3, generation='technical_failure'),
            _record(4, scores=(1,0,None,1)), _record(5)]
    rows[0]['shortcuts']['automatic']['question_leakage'] = label(1, 'synthetic')
    rows[0]['shortcuts']['automatic']['common_knowledge'] = label(1, 'synthetic')
    m = compute_record_metrics(rows)
    assert m['n_planned'] == 6
    assert (m['validity']['n_pass'],m['validity']['n_fail'],m['validity']['n_unresolved']) == (3,2,1)
    assert (m['validity']['lower'],m['validity']['upper']) == (3/6,4/6)
    assert (m['bdr']['n_pass'],m['bdr']['n_fail'],m['bdr']['n_unresolved']) == (2,3,1)
    assert (m['bdr']['lower'],m['bdr']['upper']) == (2/6,3/6)
    assert m['conditions']['0']['rate'] == 2/6
    assert m['conditions']['TI']['rate'] == 3/6
    assert m['shortcuts']['question_leakage']['rate'] == 1/6
    assert m['shortcuts']['common_knowledge']['rate'] == 1/6
    assert m['any_shortcut']['rate'] == 1/6
    assert m['labels']['relatedness']['n_not_applicable'] == 1
    for group in ('conditions','labels','shortcuts'):
        for metric in m[group].values():
            assert sum(metric[k] for k in ('n_pass','n_fail','n_unresolved','n_not_applicable')) == 6
    assert sum(c['n'] for c in m['cross_table']) == 6
    write_jsonl(tmp_path/'records.jsonl', rows)
    report = summarize_records(tmp_path/'records.jsonl')
    assert report['human']['validity']['n_pass'] == 0
    assert report['human']['bdr']['n_pass'] == 0
    assert report['human']['bdr']['n_unresolved'] == 5
    empty = compute_record_metrics([])
    assert empty['validity']['rate'] is None and empty['bdr']['upper'] is None


@pytest.mark.parametrize('change', ['reference','response_question','rubric','refusal'])
def test_binding_and_refusal_not_wrong(change):
    r = _record()
    if change == 'reference': r['reference']['question_sha256'] = 'foreign'
    if change == 'response_question': r['responses']['I']['question_id'] = 'foreign'
    if change == 'rubric': r['reference']['rubric'] = ['changed']
    if change == 'refusal':
        r['responses']['I'].update(status='refused', abstain=True, correct=0)
        assert record_states(r)['bdr'] is None
    else:
        with pytest.raises(ValueError): record_states(r)


@pytest.fixture
def project(tmp_path):
    shutil.copytree(ROOT/'prompts', tmp_path/'prompts')
    (tmp_path/'configs').mkdir(); shutil.copy(ROOT/'configs/experiments.json',tmp_path/'configs/experiments.json')
    image = tmp_path/'image.png'; image.write_bytes((ROOT/'tests/fixtures/stage_a/images/m_thermo.png').read_bytes())
    m = {'material_id':'dev','source_group_id':'dev-group','split':'train',
         'text':'UNIQUE_TEXT_RULE_17','image_path':'image.png','image_sha256':sha256_file(image),
         'provenance':'synthetic_test','text_source':'synthetic'}
    write_jsonl(tmp_path/'materials.jsonl',[m])
    return tmp_path


class Scripted:
    model = 'offline-test'
    identity = {'model':'offline-test','temperature':0,'do_sample':False}
    def __init__(self): self.requests=[]
    def complete(self, request):
        self.requests.append(request); name=request.metadata['prompt_version']
        if name in ('plain-v1','strong-v1','generate-anchored-v1','generate-anchored-v2','generate-no-image-anchor-v1'):
            value='What result follows from the shown experiment?'
        elif name == 'answer-v1':
            value={'status':'answerable','answer':'yes','acceptable_answers':['yes'],'rubric':['yes'],
                   'text_evidence':[], 'image_evidence':[]}
        elif name == 'solve-v1':
            value={'answer':'yes' if request.condition in ('TI','0') else 'no','abstain':False,'reason_brief':'synthetic'}
        elif name == 'score-v1':
            value={'correct':int('"answer": "yes"' in request.prompt),'reason':'synthetic'}
        elif name == 'relatedness-v1': value={'relatedness':1,'reason':'synthetic'}
        elif name == 'necessity-v1':
            value={k:{'value':1,'reason':'synthetic'} for k in DIMENSIONS if k!='relatedness'}
        elif name == 'shortcut-v1': value={k:{'value':0,'reason':'synthetic'} for k in SHORTCUTS}
        elif name == 'extract-text-v1': value={'text_evidence':[{'id':'t1'}]}
        elif name == 'extract-image-v1': value={'image_evidence':[{'id':'i1'}]}
        elif name in ('pair-v1','pair-v2','plan-no-image-anchor-v1'):
            value={'status':'ready','text_ids':['t1'],'image_ids':['i1'],**{
                k:{'passed':True} for k in ('redundancy_check','common_knowledge_check','integration_check')}}
        else: raise AssertionError(name)
        text=value if isinstance(value,str) else json.dumps(value)
        return Completion(text,self.model,{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5,'cost':None,'network_requests':1})


def test_versioned_probe_prompts_preserve_old_runs_and_resume(project):
    overrides={'pair-v1':'pair-v2','generate-anchored-v1':'generate-anchored-v2'}
    init_experiment(materials_path='materials.jsonl',run_id='old',methods=('anchored',),project_root=project)
    old=project/'outputs/old'; protected={str(p):sha256_file(p) for p in old.rglob('*') if p.is_file()}
    init_experiment(materials_path='materials.jsonl',run_id='probe',methods=('anchored',),
                    project_root=project,prompt_overrides=overrides)
    provider=Scripted();kw=dict(run_id='probe',project_root=project,provider=provider,stages=('generation',))
    assert run_experiment(**kw)['new_steps']==1
    assert [r.metadata['prompt_version'] for r in provider.requests]==[
        'extract-text-v1','extract-image-v1','pair-v2','generate-anchored-v2']
    row=ExperimentStore(project/'outputs/probe').rows()[0]
    assert row['prompt_versions']['pair-v1']=='pair-v2'
    assert compute_record_metrics([row])['n_planned']==1
    assert run_experiment(**kw)['new_steps']==0 and len(provider.requests)==4
    assert all(sha256_file(Path(p))==h for p,h in protected.items())
    with pytest.raises(ValueError,match='changed'):
        init_experiment(materials_path='materials.jsonl',run_id='probe',methods=('anchored',),project_root=project)
    (project/'outputs/probe/prompts/pair-v2.txt').write_text('tampered')
    with pytest.raises(ValueError,match='snapshot changed'):run_experiment(**kw)
    with pytest.raises(ValueError,match='template roles'):
        init_experiment(materials_path='materials.jsonl',run_id='bad',project_root=project,
                        prompt_overrides={'unknown':'pair-v2'})
    (project/'prompts/pair-bad.txt').write_text('$UNEXPECTED')
    with pytest.raises(ValueError,match='variables'):
        init_experiment(materials_path='materials.jsonl',run_id='bad',project_root=project,
                        prompt_overrides={'pair-v1':'pair-bad'})


def test_resume_crash_after_raw_output_and_isolated_inputs(project, monkeypatch):
    import egvqg.experiments as exp
    init_experiment(materials_path='materials.jsonl',run_id='dev',project_root=project)
    provider=Scripted()
    run_experiment(run_id='dev',project_root=project,provider=provider,max_steps=1)
    assert len(provider.requests)==1
    original=exp._evaluate_step
    def interrupted(*args):
        original(*args)
        raise KeyboardInterrupt
    monkeypatch.setattr(exp,'_evaluate_step',interrupted)
    with pytest.raises(KeyboardInterrupt): run_experiment(run_id='dev',project_root=project,provider=provider)
    assert len(provider.requests)==2
    monkeypatch.setattr(exp,'_evaluate_step',original)
    run_experiment(run_id='dev',project_root=project,provider=provider)
    out=project/'outputs/dev'; records=(out/'records.jsonl').read_bytes(); calls=(out/'calls.jsonl').read_bytes()
    count=len(provider.requests)
    run_experiment(run_id='dev',project_root=project,provider=provider)
    assert len(provider.requests)==count
    assert (out/'records.jsonl').read_bytes()==records and (out/'calls.jsonl').read_bytes()==calls
    m=json.loads((out/'metrics.json').read_text())
    assert m['automatic']['bdr']['rate']==1
    for request in provider.requests:
        if request.metadata['prompt_version']=='solve-v1':
            assert ('UNIQUE_TEXT_RULE_17' in request.prompt)==(request.condition in ('TI','T'))
            assert (request.image_path is not None)==(request.condition in ('TI','I'))
            assert 'candidate_answer' not in request.prompt and 'rubric' not in request.prompt
    assert len(read_jsonl(out/'records.jsonl'))==1
    changed=copy.deepcopy(ExperimentStore(out).rows()[0]); previous=copy.deepcopy(changed)
    changed['method']='strong'
    with pytest.raises(ValueError): ExperimentStore(out).put(changed,previous=previous)
    (project/'image.png').write_bytes(b'changed')
    with pytest.raises(ValueError): run_experiment(run_id='dev',project_root=project,provider=provider)


def test_technical_retries_parse_failure_and_cache_identity(project, monkeypatch):
    import egvqg.providers as module
    monkeypatch.setattr(module.time,'sleep',lambda n:None)
    class Retry(Scripted):
        def complete(self, request):
            self.requests.append(request)
            if len(self.requests)<3: raise ProviderError('judge_http_503')
            return Completion('{"relatedness":1}',self.model,{'total_tokens':4})
    provider=Retry(); out=project/'calls'
    caller=ModelCaller(provider=provider,prompts=PromptSet(project/'prompts',names=TEMPLATES),
                       run_dir=out,protocol_version='test')
    kwargs=dict(operation_id='op',attempt_id='a',role='review',template='relatedness-v1',
                variables={'T':'text','Q':'q'},task='assess_quality')
    result=caller.call_model(**kwargs)
    assert result['status']=='ok' and len(provider.requests)==3
    assert len(read_jsonl(out/'calls.jsonl'))==3
    # Simulate raw saved, parsed result not yet saved.
    result['parsed_output']=None; atomic_upsert(out/'calls.jsonl','call_id',result)
    assert caller.call_model(**kwargs)['parsed_output']=={'relatedness':1}
    assert len(provider.requests)==3
    with pytest.raises(ValueError): caller.call_model(**{**kwargs,'variables':{'T':'changed','Q':'q'}})
    class Bad(Scripted):
        def complete(self,request):
            self.requests.append(request); return Completion('not json',self.model,{})
    bad=Bad(); caller.provider=bad
    assert caller.call_model(**{**kwargs,'operation_id':'bad'})['status']=='parse_failure'
    assert len(bad.requests)==1
    assert caller.call_model(**{**kwargs,'operation_id':'bad'})['raw_output']=='not json'
    assert len(bad.requests)==1


def test_four_methods_and_all_templates_render(project):
    init_experiment(materials_path='materials.jsonl',run_id='methods',project_root=project,
                    methods=('plain','strong','anchored','no_image_anchor'))
    provider=Scripted(); run_experiment(run_id='methods',project_root=project,provider=provider)
    rows=ExperimentStore(project/'outputs/methods').rows()
    assert len(rows)==4 and all(r['status']=='automatic_complete' for r in rows)
    assert all(record_states(r)['bdr']==1 for r in rows)
    prompts=PromptSet(project/'prompts',names=TEMPLATES)
    for name in TEMPLATES:
        rendered=prompts.render(name,**{k:'example' for k in prompts.variables(name)})
        assert '$T' not in rendered and '$Q' not in rendered
    for request in provider.requests:
        if request.metadata['prompt_version']=='extract-image-v1':
            assert request.text is None and 'UNIQUE_TEXT_RULE_17' not in request.prompt


def test_atomic_updates_preserve_other_rows_and_fixed_plan(project):
    m=read_jsonl(project/'materials.jsonl')[0]
    write_jsonl(project/'materials.jsonl',[m,{**m,'material_id':'second'}])
    init_experiment(materials_path='materials.jsonl',run_id='atomic',project_root=project)
    out=project/'outputs/atomic'; store=ExperimentStore(out); rows=store.rows()
    before=copy.deepcopy(rows[0]); changed=copy.deepcopy(before); changed['notes'].append('test')
    store.put(changed,previous=before)
    assert store.rows()[1]==rows[1]
    with pytest.raises(Exception): store.put(changed,previous=before)
    write_jsonl(out/'records.jsonl',[changed])
    with pytest.raises(ValueError): summarize_records(out/'records.jsonl')


def test_human_reference_correction_reuses_q_answers_and_invalidates_only_affected_steps(project):
    from egvqg.experiments import apply_human_reviews
    init_experiment(materials_path='materials.jsonl',run_id='review',project_root=project)
    provider=Scripted(); run_experiment(run_id='review',project_root=project,provider=provider)
    out=project/'outputs/review'; old=ExperimentStore(out).rows()[0]
    reviews=read_jsonl(out/'review_template.jsonl'); item=reviews[0]
    item.update(reviewer='synthetic-test-reviewer',labels={k:label(1,'synthetic') for k in DIMENSIONS},
                shortcuts={k:label(0,'synthetic') for k in SHORTCUTS})
    item['reference']={'status':'verified','answer':'changed human answer','acceptable_answers':[],
                       'rubric':['changed required point']}
    write_jsonl(project/'human.jsonl',reviews)
    apply_human_reviews(run_id='review',reviews_path='human.jsonl',adjudicate=True,project_root=project)
    updated=ExperimentStore(out).rows()[0]
    assert updated['question']==old['question']
    assert updated['steps']['candidate_answer']=='done' and updated['steps']['generation']=='done'
    assert updated['steps']['score_TI']=='pending'
    assert updated['responses']['TI']['correct'] is None
    count=len(provider.requests)
    run_experiment(run_id='review',project_root=project,provider=provider)
    assert all(r.metadata['prompt_version'] not in ('plain-v1','solve-v1','answer-v1') for r in provider.requests[count:])
    assert ExperimentStore(out).rows()[0]['reference']['verified_answer']=='changed human answer'
    count=len(provider.requests)
    apply_human_reviews(run_id='review',reviews_path='human.jsonl',adjudicate=True,project_root=project)
    assert len(provider.requests)==count


def test_lost_remote_response_does_not_resend_or_lose_request_accounting(project, monkeypatch):
    from egvqg.providers import JudgeProvider
    from egvqg.experiments import call_usage
    p=JudgeProvider(model='offline-test',base_url='http://example.invalid/v1',api_key='fixture-key',
        cache_dir=project/'cache',ledger_path=project/'requests.jsonl',visual=True,api='chat_completions')
    def interrupted(payload): raise KeyboardInterrupt
    monkeypatch.setattr(p,'_post_json',interrupted)
    caller=ModelCaller(provider=p,prompts=PromptSet(project/'prompts',names=TEMPLATES),
                       run_dir=project/'lost',protocol_version='test')
    args=dict(operation_id='lost',attempt_id='a',role='review',template='relatedness-v1',
              variables={'T':'text','Q':'q'},task='assess_quality')
    with pytest.raises(KeyboardInterrupt): caller.call_model(**args)
    assert len(read_jsonl(project/'requests.jsonl'))==1
    result=caller.call_model(**args)
    assert result['status']=='technical_failure' and result['remote_requests']==1
    assert 'billing_unknown' in result['error']
    assert len(read_jsonl(project/'requests.jsonl'))==1
    usage=call_usage(read_jsonl(project/'lost/calls.jsonl'))
    assert usage['remote_requests']==1 and usage['total_tokens'] is None


def test_cached_response_survives_crash_before_call_commit_without_double_billing(project, monkeypatch):
    import egvqg.records as records
    from egvqg.providers import JudgeProvider
    from egvqg.experiments import call_usage
    p=JudgeProvider(model='offline-test',base_url='http://example.invalid/v1',api_key='fixture-key',
        cache_dir=project/'cache',ledger_path=project/'requests.jsonl',visual=True,api='chat_completions')
    count=[]
    def complete(payload):
        count.append(payload)
        return {'model':'offline-test','choices':[{'message':{'content':'{"relatedness":1}'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}}
    monkeypatch.setattr(p,'_post_json',complete)
    original=records.atomic_upsert
    def crash(path,key,row,**kwargs):
        if row.get('status')=='ok': raise KeyboardInterrupt
        return original(path,key,row,**kwargs)
    monkeypatch.setattr(records,'atomic_upsert',crash)
    caller=ModelCaller(provider=p,prompts=PromptSet(project/'prompts',names=TEMPLATES),
                       run_dir=project/'recovered',protocol_version='test')
    args=dict(operation_id='recovered',attempt_id='a',role='review',template='relatedness-v1',
              variables={'T':'text','Q':'q'},task='assess_quality')
    with pytest.raises(KeyboardInterrupt): caller.call_model(**args)
    monkeypatch.setattr(records,'atomic_upsert',original)
    result=caller.call_model(**args)
    assert result['recovered_from_cache'] and not result['cache_hit']
    assert result['parsed_output']=={'relatedness':1} and len(count)==1
    usage=call_usage(read_jsonl(project/'recovered/calls.jsonl'))
    assert usage['remote_requests']==1 and usage['total_tokens']==5


def test_preplanned_methods_wait_without_shrinking_n_or_overwriting_human_template(project):
    init_experiment(materials_path='materials.jsonl',run_id='staged',project_root=project,
                    methods=('plain','strong','anchored','no_image_anchor'))
    provider=Scripted()
    run_experiment(run_id='staged',project_root=project,provider=provider,methods=('plain','strong'))
    out=project/'outputs/staged'; rows=ExperimentStore(out).rows()
    assert len(rows)==4 and sum(r['status']=='automatic_complete' for r in rows)==2
    assert json.loads((out/'metrics.json').read_text())['n_planned']==4
    reviews=read_jsonl(out/'review_template.jsonl'); assert len(reviews)==2
    reviews[0]['notes']='human draft must survive'; write_jsonl(out/'review_template.jsonl',reviews)
    run_experiment(run_id='staged',project_root=project,provider=provider,methods=('anchored','no_image_anchor'))
    after=read_jsonl(out/'review_template.jsonl')
    assert len(after)==4 and after[0]['notes']=='human draft must survive'


def test_delegated_adjudication_preserves_ai_source_and_allows_final_unknown(project):
    from egvqg.experiments import apply_human_reviews,freeze_evaluation
    cases=[{'case_id':str(i),'material_id':'dev','question':f'Question {i}?'} for i in range(12)]
    write_jsonl(project/'cases.jsonl',cases)
    init_experiment(materials_path='materials.jsonl',cases_path='cases.jsonl',run_id='delegated',
                    project_root=project,partition='diagnostic')
    out=project/'outputs/delegated';store=ExperimentStore(out)
    for r in store.rows():
        previous=copy.deepcopy(r);r['steps']={s:'done' for s in r['steps']};r['status']='automatic_complete'
        store.put(r,previous=previous)
    reviews=read_jsonl(out/'review_template.jsonl')
    for item in reviews:
        item.update(source='user_delegated_ai',reviewer='Codex test',user_authorized=True,
                    authorization='explicit synthetic user delegation',
                    labels={k:label(0 if k=='text_necessary' else 1,'fixture source-only review') for k in DIMENSIONS})
        item['reference']={'status':'verified','answer':'yes','acceptable_answers':[],'rubric':['yes']}
        item['condition_scores']={c:{'value':None,'reason':'fixture unresolved'} for c in ('TI','T','I','0')}
    reviews[-1]['labels']['image_necessary']=label(None,'explicitly reviewed but undecided')
    write_jsonl(project/'delegated.jsonl',reviews)
    bad=copy.deepcopy(reviews);bad[0]['user_authorized']=False;write_jsonl(project/'unauthorized.jsonl',bad)
    with pytest.raises(ValueError):apply_human_reviews(run_id='delegated',reviews_path='unauthorized.jsonl',adjudicate=True,project_root=project)
    apply_human_reviews(run_id='delegated',reviews_path='delegated.jsonl',adjudicate=True,project_root=project)
    rows=ExperimentStore(out).rows()
    assert all(r['labels']['human']['validity']['status']=='pending' for r in rows)
    assert rows[0]['labels']['adjudicated']['validity']['source']=='user_delegated_ai'
    (project/'protocol.md').write_text('synthetic immutable protocol')
    result=freeze_evaluation(run_id='delegated',project_root=project,protocol_path='protocol.md')
    assert result['status']=='frozen_delegated_reviewed' and result['human_confirmed_examples']==0
    previous=copy.deepcopy(rows[0]);changed=copy.deepcopy(previous);changed['notes'].append('tamper')
    with pytest.raises(ValueError):ExperimentStore(out).put(changed,previous=previous)


def test_batch_contract_rejects_config_prompt_material_and_model_drift_before_calls(project):
    from egvqg.batch import validate_batch_contract
    from egvqg.records import canonical_json,sha256_text
    out=project/'outputs/contract';out.mkdir(parents=True)
    files={k:project/f'{k}.txt' for k in ('protocol','evaluation_manifest','evaluation_records','selection_manifest','materials','repeatability')}
    for k,p in files.items():p.write_text(k)
    write_jsonl(out/'materials.jsonl',read_jsonl(project/'materials.jsonl'))
    write_jsonl(out/'plan.jsonl',[{'attempt_id':'a','method':'plain'}])
    cfg=json.loads((project/'configs/experiments.json').read_text())
    contract={**{k+'_path':str(p) for k,p in files.items() if k in ('protocol','materials','repeatability')},
              **{k:str(p) for k,p in files.items() if k not in ('protocol','materials','repeatability')},
              **{k+'_sha256':sha256_file(p) for k,p in files.items()},
              'material_snapshot_sha256':sha256_file(out/'materials.jsonl'),
              'plan_sha256':sha256_text(canonical_json(read_jsonl(out/'plan.jsonl'))),
              'prompt_sha256':PromptSet(project/'prompts',names=TEMPLATES).snapshot_sha256(),
              'config_path':str(project/'configs/experiments.json'),
              'configuration_sha256':sha256_text(canonical_json(cfg)), 'model_identity':{'model':'fixed'}}
    write_json(out/'manifest.json',{'batch_contract':contract,'config':cfg})
    validate_batch_contract(out,project,identity={'model':'fixed'})
    with pytest.raises(ValueError):validate_batch_contract(out,project,identity={'model':'changed'})
    for path in (files['protocol'],files['materials'],out/'materials.jsonl',project/'prompts/plain-v1.txt',project/'configs/experiments.json'):
        original=path.read_bytes()
        if path.name=='experiments.json':
            changed=json.loads(original);changed['max_output_tokens']+=1;write_json(path,changed)
        else:
            path.write_bytes(original+b' ')
        with pytest.raises(ValueError):validate_batch_contract(out,project)
        path.write_bytes(original)


def _gated_project(project):
    shutil.copy(ROOT/'configs/experiments_review_gated.json', project/'configs/experiments_review_gated.json')
    init_experiment(materials_path='materials.jsonl', run_id='gated', methods=('strong','anchored'),
                    project_root=project, config_path='configs/experiments_review_gated.json')
    return project/'outputs/gated'


def test_review_gated_stages_block_scoring_until_reference_decision_and_resume_without_repeats(project):
    from egvqg.experiments import apply_reference_reviews
    from egvqg.metrics import outcome_patterns
    out=_gated_project(project); provider=Scripted(); kw=dict(run_id='gated',project_root=project,provider=provider)
    plan=run_experiment(**kw,dry_run=True,stages=('generation',))
    assert plan['max_logical_calls']==5 and plan['pending_by_step']=={'generation':2}
    assert run_experiment(**kw,stages=('generation',))['status']=='selected_stages_complete'
    assert [r.metadata['role'] for r in provider.requests]==['generation','extract_text','extract_image','pair','generation']
    rows=ExperimentStore(out).rows()
    assert all(r['steps']['generation']=='done' and r['steps']['candidate_answer']=='pending' and r['status']=='running' for r in rows)
    questions=[r['question'] for r in rows]
    result=run_experiment(**kw)
    assert result['status']=='awaiting_reference_review' and result['awaiting_reference_review']==2
    roles=[r.metadata['role'] for r in provider.requests[5:]]
    # One candidate reference, four isolated answers, relatedness; nothing reference-dependent.
    assert sorted(roles)==sorted(['candidate_answer','answer_TI','answer_T','answer_I','answer_0','relatedness']*2)
    rows=ExperimentStore(out).rows()
    for r in rows:
        assert r['reference']['status']=='candidate_unreviewed' and r['reference']['verified_answer'] is None
        assert r['steps']['independent_answer']==r['steps']['reference_resolution']=='not_applicable'
        assert all(r['steps'][s]=='blocked' for s in ('score_TI','score_T','score_I','score_0','necessity','shortcuts'))
    m=compute_record_metrics(rows)
    # Fixed N; an unreviewed reference is unresolved, never 0.
    assert m['n_planned']==2 and m['bdr']['n_unresolved']==2 and m['bdr']['n_fail']==0
    assert m['conditions']['TI']['n_unresolved']==2 and m['validity']['n_unresolved']==2
    assert outcome_patterns(rows)['by_method']['strong']['patterns']=={'U/U/U/U':1}
    count=len(provider.requests)
    assert run_experiment(**kw)['new_steps']==0 and len(provider.requests)==count
    sheet=read_jsonl(out/'reference_review_template.jsonl')
    assert len(sheet)==2 and all(s['decision'] is None and s['candidate']['answer']=='yes' for s in sheet)
    assert apply_reference_reviews(run_id='gated',reviews_path='outputs/gated/reference_review_template.jsonl',
                                   project_root=project)=={'updated_records':0,'undecided_rows_skipped':2}
    who=dict(reviewer='synthetic-test-reviewer',source='human')
    sheet[0].update(decision='corrected',answer='no',rubric=['no'],note='synthetic correction',**who)
    sheet[1].update(decision='unresolved',note='synthetic ambiguity',**who)
    write_jsonl(project/'refs.jsonl',sheet)
    foreign=copy.deepcopy(sheet); foreign[0]['question_sha256']='foreign'; write_jsonl(project/'bad.jsonl',foreign)
    with pytest.raises(ValueError,match='another Q'):
        apply_reference_reviews(run_id='gated',reviews_path='bad.jsonl',project_root=project)
    assert ExperimentStore(out).rows()==rows  # all-or-nothing validation
    assert apply_reference_reviews(run_id='gated',reviews_path='refs.jsonl',project_root=project)['updated_records']==2
    assert apply_reference_reviews(run_id='gated',reviews_path='refs.jsonl',project_root=project)['updated_records']==0
    assert run_experiment(**kw)['status']=='automatic_complete'
    new=provider.requests[count:]
    # Corrected Q: 4 scores + 2 audits. Unresolved Q: audits only, scores stay blocked.
    assert sorted(r.metadata['role'] for r in new)==sorted(
        ['score_TI','score_T','score_I','score_0','necessity','shortcuts','necessity','shortcuts'])
    assert all('参考答案：no' in r.prompt for r in new if r.metadata['prompt_version']=='score-v1')
    first,second=ExperimentStore(out).rows()
    assert [first['question'],second['question']]==questions
    assert first['reference']['status']=='verified' and first['reference']['verification_source'].endswith(':corrected')
    assert first['reference']['candidate_answer']['answer']=='yes'  # candidate preserved beside the correction
    assert record_states(first)['conditions']['TI'] is not None
    assert second['reference']['status']=='unresolved' and second['steps']['score_TI']=='blocked'
    assert record_states(second)['bdr'] is None
    # No answering request ever carried a reference, review text or generation evidence.
    for r in provider.requests:
        if r.metadata['prompt_version']=='solve-v1':
            assert 'synthetic correction' not in r.prompt and 'rubric' not in r.prompt and 't1' not in r.prompt
    assert len(read_jsonl(out/'reviews.jsonl'))==2
    count=len(provider.requests)
    assert run_experiment(**kw)['new_steps']==0 and len(provider.requests)==count
    init_experiment(materials_path='materials.jsonl',run_id='legacy',project_root=project)
    with pytest.raises(ValueError,match='review_gated'):
        apply_reference_reviews(run_id='legacy',reviews_path='refs.jsonl',project_root=project)


def test_json_backslash_repair_is_opt_in_recorded_and_never_rescues_truncation(project):
    from egvqg.providers import OutputParseError, parse_model_output
    bs=chr(92)
    raw='{"answer":"'+bs+'(P_A:P_B'+bs+')=9:4 '+bs+'(x'+bs+') a'+bs+'nb '+bs+'u00e9 '+bs+bs+'bar","abstain":false}'
    with pytest.raises(OutputParseError):
        parse_model_output(raw,'json')
    value,repair=parse_model_output(raw,'json','escape_invalid_backslashes')
    # Legal escapes (newline, unicode, escaped backslash) keep their JSON meaning.
    assert value['answer']==bs+'(P_A:P_B'+bs+')=9:4 '+bs+'(x'+bs+') a'+chr(10)+'b '+chr(233)+' '+bs+'bar'
    assert repair['kind']=='escape_invalid_backslashes' and repair['replacements']==4
    assert repair['raw_sha256']==sha256_text(raw)
    assert parse_model_output('{"a":"b"}','json','escape_invalid_backslashes')==({'a':'b'},None)
    with pytest.raises(OutputParseError):
        parse_model_output(raw[:30],'json','escape_invalid_backslashes')
    class One:
        model='offline-test'; identity={'model':'offline-test'}
        def complete(self,request):
            return Completion(raw,self.model,{'total_tokens':1,'network_requests':1},'stop')
    args=dict(operation_id='op',attempt_id='a',role='answer_TI',template='solve-v1',
              variables={'VISIBLE_MATERIALS':'','Q':'q'},task='answer_question')
    prompts=PromptSet(project/'prompts',names=TEMPLATES)
    strict=ModelCaller(provider=One(),prompts=prompts,run_dir=project/'strict',protocol_version='test')
    assert strict.call_model(**args)['status']=='parse_failure'
    lenient=ModelCaller(provider=One(),prompts=prompts,run_dir=project/'lenient',protocol_version='test',
                        json_repair='escape_invalid_backslashes')
    call=lenient.call_model(**args)
    assert call['status']=='ok' and call['raw_output']==raw and call['parse_repair']['replacements']==4
    assert read_jsonl(project/'lenient/calls.jsonl')[0]['parse_repair']['kind']=='escape_invalid_backslashes'


def test_source_review_packet_hides_method_candidate_and_is_never_regenerated(project):
    from egvqg.experiments import export_source_review
    out=_gated_project(project); provider=Scripted()
    with pytest.raises(ValueError,match='generation'):
        export_source_review(run_id='gated',project_root=project)
    run_experiment(run_id='gated',project_root=project,provider=provider,stages=('generation','reference'))
    assert all(r.metadata['prompt_version']!='solve-v1' for r in provider.requests)
    before=sha256_file(out/'records.jsonl')
    assert export_source_review(run_id='gated',project_root=project)['n_in_packet']==2
    packet=(out/'source_review/source_inputs.jsonl').read_text()+(out/'source_review/labels_template.jsonl').read_text()
    rows=ExperimentStore(out).rows()
    for hidden in ('strong','anchored','attempt_','"yes"','t1','i1'):
        assert hidden not in packet
    items=read_jsonl(out/'source_review/source_inputs.jsonl')
    assert {i['question_sha256'] for i in items}=={r['question']['sha256'] for r in rows}
    assert set(items[0])=={'review_id','text','image_path','image_sha256','question','question_sha256'}
    assert {k['attempt_id'] for k in read_jsonl(out/'source_review/key.jsonl')}=={r['attempt_id'] for r in rows}
    assert sha256_file(out/'records.jsonl')==before
    with pytest.raises(ValueError,match='already exists'):
        export_source_review(run_id='gated',project_root=project)


def test_attempt_filter_runs_only_selected_and_keeps_the_rest_unresolved_in_n(project):
    out=_gated_project(project); provider=Scripted(); kw=dict(run_id='gated',project_root=project,provider=provider)
    run_experiment(**kw,stages=('generation','reference'))
    first,second=ExperimentStore(out).rows(); count=len(provider.requests)
    with pytest.raises(ValueError,match='planned'):
        run_experiment(**kw,attempt_ids=['foreign'])
    run_experiment(**kw,stages=('answers',),attempt_ids=[first['attempt_id']])
    assert [r.metadata['role'] for r in provider.requests[count:]]==['answer_TI','answer_T','answer_I','answer_0']
    rows=ExperimentStore(out).rows()
    assert rows[1]==second and rows[1]['steps']['answer_TI']=='pending'
    m=compute_record_metrics(rows)
    assert m['n_planned']==2 and m['conditions']['TI']['n_unresolved']==2 and m['bdr']['n_fail']==0

