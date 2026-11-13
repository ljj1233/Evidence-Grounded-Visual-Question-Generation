"""Verifier failures remain unresolved/rejected and never rewrite source results."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from egvqg.evidence_verification import CHECKS,verification_gate,verify_open
from egvqg.assessment_rules import choice_conflicts, check_frozen_sources
from egvqg.providers import Completion
from egvqg.records import canonical_json,read_jsonl,sha256_file,sha256_text,write_json,write_jsonl


@pytest.mark.parametrize('change,expected',[(None,'accept'),('well_posed','reject'),('premises_supported','unresolved'),('missing','unresolved')])
def test_verifier_is_fail_closed(change,expected):
    output={**dict.fromkeys(CHECKS,'pass'),'reason':'fixture','recomputed_answers':['A','B']}
    if change=='well_posed':output[change]='fail'
    elif change=='premises_supported':output[change]='uncertain'
    elif change=='missing':output.pop('conclusion_checked')
    assert verification_gate(output)==expected


@pytest.mark.parametrize('answer,bad', [('B. beaver',True),('C. beaver',False),('B. silver maple',False),
                                      ('海狸',False),('C',False),('D. beaver',True)])
def test_choice_letter_text_binding(answer,bad):
    assert bool(choice_conflicts({'answer':answer},['gray fox','silver maple','beaver']))==bad


def test_explicit_failure_with_empty_recomputation_is_rejected_in_v2_only():
    output={**dict.fromkeys(CHECKS,'pass'),'premises_supported':'fail','reason':'unsupported premise','recomputed_answers':[]}
    assert verification_gate(output)=='unresolved'  # unchanged historical semantics
    assert verification_gate(output,strict=True)=='reject'


def test_archived_code_only_allows_completed_readonly_history(tmp_path):
    code=tmp_path/'src/egvqg/runtime.py';code.parent.mkdir(parents=True);code.write_text('old')
    expected=sha256_file(code);out=tmp_path/'run';(out/'code_snapshots').mkdir(parents=True)
    (out/'code_snapshots/runtime.py').write_text('old');code.write_text('new')
    check_frozen_sources({str(code):expected},out,completed=True)
    with pytest.raises(ValueError,match='source changed'):
        check_frozen_sources({str(code):expected},out,completed=False)
    data=tmp_path/'image.png';data.write_bytes(b'old');h=sha256_file(data);data.write_bytes(b'new')
    with pytest.raises(ValueError,match='source changed'):
        check_frozen_sources({str(data):h},out,completed=True)


@pytest.mark.parametrize('validity',[1,0,None])
def test_v2_shared_contract_verifier_gate_and_fixed_denominator(tmp_path,validity):
    from egvqg.open_necessity import init_open_probe,run_open_probe,report_open_probe
    root=Path(__file__).resolve().parents[1];(tmp_path/'prompts').mkdir();(tmp_path/'configs').mkdir()
    for name in ('open-necessity-v2','verify-necessity-evidence-v2'):
        shutil.copy(root/f'prompts/{name}.txt',tmp_path/'prompts')
    shutil.copy(root/'configs/necessity_assessment_v2.json',tmp_path/'configs')
    shutil.copy(root/'tests/fixtures/stage_a/images/m_thermo.png',tmp_path/'image.png')
    case={'case_id':'C','question':'Who eats it?\nOptions:\nA. gray fox\nB. silver maple\nC. beaver',
          'choices':['gray fox','silver maple','beaver'],'text':'TEXT_ONLY',
          'image_path':'image.png','image_sha256':sha256_file(tmp_path/'image.png')}
    write_jsonl(tmp_path/'cases.jsonl',[case]);write_jsonl(tmp_path/'key.jsonl',[{'case_id':'C','expected':{'T':None,'I':None},'reference':'SECRET_KEY','validity':validity}])
    config='configs/necessity_assessment_v2.json'
    init_open_probe(run_id='source',cases_path='cases.jsonl',key_path='key.jsonl',project_root=tmp_path,config_path=config)
    class Mock:
        model='offline';identity={'model':'offline'}
        def __init__(self):self.requests=[]
        def complete(self,r):
            self.requests.append(r)
            output=({**dict.fromkeys(CHECKS,'pass'),'reason':'fixture','recomputed_answers':['C. beaver','A. gray fox']}
                    if r.metadata['role']=='verify_proof' else
                    {'kind':'ambiguous','scenario_a':'beaver eats it','answer_a':'B. beaver',
                     'scenario_b':'fox eats it','answer_b':'A. gray fox','reason':'fixture'})
            return Completion(json.dumps(output),self.model,{'network_requests':1,'total_tokens':4})
    mock=Mock();run_open_probe(run_id='source',project_root=tmp_path,provider=mock)
    out=tmp_path/'outputs/source';row=read_jsonl(out/'records.jsonl')[0]
    items=[{'item_id':'C_T','case_id':'C','removed':'T','candidate':row['analysis']['T']['output'],
            'source_analysis_sha256':sha256_text(canonical_json(row['analysis']['T']))}]
    write_jsonl(tmp_path/'items.jsonl',items)
    reviews=[{'case_id':'C','attempt_id':row['attempt_id'],'removed':m,'analysis_sha256':sha256_text(canonical_json(row['analysis'][m])),
              'result':'supported','source':'user_delegated_ai','reviewer':'fixture','reason':'fixture'} for m in ('T','I')]
    write_jsonl(out/'evidence_reviews.jsonl',reviews)
    kw={'run_id':'verify','source_run':'source','items_path':'items.jsonl','project_root':tmp_path,'provider':mock}
    assert verify_open(**kw,config_path=config)['counts']=={'accept':0,'reject':1,'unresolved':0}
    assert verify_open(**kw)['new_steps']==0
    report=report_open_probe(run_id='source',project_root=tmp_path,verification_run='verify')
    assert report['verified_both_required']=={'pass':0,'fail':int(validity==0),'unresolved':int(validity!=0),'lower':0.0,'upper':float(validity!=0)}
    assert report['verified_necessity']['T']=={'n_planned':1,'pass':0,'fail':0,'unresolved':1,'lower':0.0,'upper':1.0}
    knowledge=json.loads((tmp_path/config).read_text())['knowledge']
    assert all(knowledge in r.prompt and 'SECRET_KEY' not in r.prompt for r in mock.requests)
    assert 'two_hypothetical_completions' in mock.requests[-1].prompt
    assert all(('TEXT_ONLY' not in r.prompt and r.text is None) for r in mock.requests if r.condition=='I')
    write_jsonl(out/'validity_reviews.jsonl',[{'case_id':'C','case_sha256':sha256_text(canonical_json(case)),
        'knowledge_sha256':sha256_text(knowledge),'validity':None,'source':'user_delegated_ai',
        'reviewer':'fixture','reason':'full input premise disputed'}])
    assert report_open_probe(run_id='source',project_root=tmp_path,verification_run='verify')['verified_both_required']['unresolved']==1
    review=read_jsonl(out/'validity_reviews.jsonl')[0];review['knowledge_sha256']='stale'
    write_jsonl(out/'validity_reviews.jsonl',[review])
    with pytest.raises(ValueError,match='validity review binding'):
        report_open_probe(run_id='source',project_root=tmp_path,verification_run='verify')
    (tmp_path/config).write_text('{}')
    with pytest.raises(ValueError,match='source changed'):
        run_open_probe(run_id='source',project_root=tmp_path,provider=mock)


def test_verifier_input_isolation_source_preservation_and_resume(tmp_path):
    root=Path(__file__).resolve().parents[1];(tmp_path/'prompts').mkdir()
    shutil.copy(root/'prompts/verify-necessity-evidence-v1.txt',tmp_path/'prompts')
    shutil.copy(root/'tests/fixtures/stage_a/images/m_thermo.png',tmp_path/'image.png')
    source=tmp_path/'outputs/source';source.mkdir(parents=True)
    case={'case_id':'C','question':'Q sentinel','text':'HIDDEN_TEXT_SENTINEL',
          'image_path':'image.png','image_sha256':sha256_file(tmp_path/'image.png')}
    a={'status':'ok','output':{'kind':'ambiguous','scenario_a':'x=1','answer_a':'1','scenario_b':'x=2','answer_b':'2','reason':'ORIGINAL_REASON_SENTINEL'}}
    write_jsonl(source/'cases.jsonl',[case]);write_jsonl(source/'records.jsonl',[{'case_id':'C','analysis':{'T':a}}]);write_json(source/'manifest.json',{'fixture':True})
    item={'item_id':'V','case_id':'C','removed':'T','source_analysis_sha256':sha256_text(canonical_json(a)),
          'candidate':copy.deepcopy(a['output']),'expected_review':'EXPECTED_SENTINEL'}
    write_jsonl(tmp_path/'items.jsonl',[item]);before={f:sha256_file(source/f) for f in ('cases.jsonl','records.jsonl','manifest.json')}
    class Mock:
        model='offline';identity={'model':'offline'}
        def __init__(self):self.requests=[]
        def complete(self,r):
            self.requests.append(r)
            output={**dict.fromkeys(CHECKS,'pass'),'reason':'fixture','recomputed_answers':['1','2']}
            return Completion(json.dumps(output),self.model,{'network_requests':1,'prompt_tokens':2,'completion_tokens':2,'total_tokens':4})
    mock=Mock();kw={'run_id':'verify','source_run':'source','items_path':'items.jsonl','project_root':tmp_path,'provider':mock}
    assert verify_open(**kw)['new_steps']==1
    r=mock.requests[0]
    assert r.image_path and r.text is None
    assert not any(s in r.prompt for s in ('HIDDEN_TEXT_SENTINEL','EXPECTED_SENTINEL','ORIGINAL_REASON_SENTINEL','"kind"'))
    assert verify_open(**kw)['new_steps']==0 and len(mock.requests)==1
    assert before=={f:sha256_file(source/f) for f in before}
    item['candidate']['answer_a']='changed';write_jsonl(tmp_path/'items.jsonl',[item])
    with pytest.raises(ValueError,match='changed'):verify_open(**kw)
