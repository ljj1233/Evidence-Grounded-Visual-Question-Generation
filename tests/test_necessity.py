"""Evidence semantics, uncertainty, identity and evaluator-only resume checks."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from egvqg.metrics import decide_necessity,validate_necessity_item
from egvqg.necessity import init_necessity,run_necessity,review_necessity
from egvqg.providers import Completion
from egvqg.records import read_jsonl,write_jsonl,write_json,sha256_file,sha256_text,canonical_json
from egvqg.evidence_contract import normalize_v1,decode_v3

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('variant',['extra','equal','opposite','placeholder','alias','unknown_alias','zero_conflict'])
def test_norm_v1_retains_raw_and_never_repairs_real_conflicts(variant):
    raw={'removed':'T','components':{'c1':evidence(1)},'decisive_facts':[]}
    item=raw['components']['c1']
    if variant=='extra':item['path']=evidence(0)['path']
    if variant=='equal':
        item['path']=evidence(0)['path'];item['witness']['completion_b']['values']['c1']='A'
    if variant=='opposite':raw['removed']='I'
    if variant=='placeholder':raw['removed']='T或I'
    if variant in ('alias','unknown_alias'):
        raw['components']['c1']=evidence(0)
        raw['components']['c1']['path']['premises'][0]['source']='K1' if variant=='alias' else 'K99'
    if variant=='zero_conflict':raw['components']['c1']={**evidence(0),'witness':evidence(1)['witness']}
    original=copy.deepcopy(raw);out,log=normalize_v1(raw,'T',{'K1':'arithmetic'},['c1'])
    assert raw==original and out['components']['c1']['proposed']==original['components']['c1']['proposed']
    r=record([(1,1)]);r['analysis']['T'][0]['output']=out
    if variant in ('equal','opposite','unknown_alias','zero_conflict'):
        assert decide_necessity(r)['text_necessary'] is None
    if variant=='placeholder':assert out['removed']=='T' and log
    if variant=='alias':assert out['components']['c1']['path']['premises'][0]['source']=='K'
    if variant=='extra':
        assert out['components']['c1']['annotations']['extra_path']==original['components']['c1']['path']
        digest=sha256_text(canonical_json(out))
        who={'removed':'T','analysis_sha256':digest,'evidence_key':'component:c1','result':'supported','modality_check':'correct'}
        for verdict in ('sufficient','uncertain',None):
            r['reviews']=[{**who,'extra_path_verdict':verdict}]
            assert decide_necessity(r,confirmed=True)['text_necessary'] is None
        r['reviews']=[{**who,'extra_path_verdict':'not_sufficient'}]
        assert decide_necessity(r,confirmed=True)['text_necessary']==1


def test_v3_uses_actual_reference_and_cannot_hide_unchanged_answer():
    raw={'components':{'c1':{'kind':'witness','evidence':{'edits':[{'locator':'cell','from':'x','to':'y'}],
         'alternative_content':'coherent fixture','compatible_because':'fixture','alternative_values':{'c1':'A'}}}},'decisive_facts':[]}
    out,_=decode_v3(raw,'T',{'components':[{'id':'c1','value':'A'}]}, {'text':'actual T','image_sha256':'image'}, {'K1':'arithmetic'})
    assert out['components']['c1']['witness']['completion_a']['content']=='actual T'
    r=record([(1,1)]);r['analysis']['T'][0]['output']=out
    assert decide_necessity(r)['text_necessary'] is None


def evidence(value,ids=('c1',)):
    if value is None:return {'proposed':None,'path':None,'witness':None,'unresolved_reason':'synthetic unknown'}
    if value==0:return {'proposed':0,'path':{'premises':[{'statement':'fixture fact','source':'K','locator':'K1'}]},'witness':None}
    return {'proposed':1,'path':None,'witness':{
        'completion_a':{'content':'coherent fixture A','compatible_because':'synthetic','values':{c:'A' for c in ids}},
        'completion_b':{'content':'coherent fixture B','compatible_because':'synthetic','values':{c:'B' for c in ids}}}}


def record(pairs):
    ids=[f'c{i+1}' for i in range(len(pairs))]
    return {'reference':{'validity':1,'components':[{'id':c} for c in ids]},'reviews':[],
        'analysis':{m:[{'output':{'removed':m,'components':{c:evidence(pair[j],ids) for c,pair in zip(ids,pairs)},
                                 'decisive_facts':[]}}] for j,m in enumerate(('T','I'))}}


@pytest.mark.parametrize('pairs,expected',[
    ([(1,1)],(1,1,1)), ([(1,0),(0,1)],(1,1,0)),
    ([(1,None),(None,1)],(1,1,None)), ([(1,0),(None,1)],(1,1,None)),
    ([(0,1),(0,1)],(0,1,None)), ([(0,1),(None,0)],(None,1,None)),
    ([(1,1),(0,1)],(1,1,1)), # computed ratio can lose T dependence; no inheritance
])
def test_component_logic_does_not_turn_uncertainty_into_stitching(pairs,expected):
    r=record(pairs);d=decide_necessity(r)
    assert (d['text_necessary'],d['image_necessary'],d['integration'])==expected
    assert decide_necessity(r,confirmed=True)['text_necessary'] is None
    assert decide_necessity(r,confirmed=True)['status']=='pending_evidence_review'


@pytest.mark.parametrize('change', ['partial','hidden_source','conflict','unchanged','missing'])
def test_invalid_evidence_stays_unresolved(change):
    r=record([(0,1)]);item=r['analysis']['T'][0]['output']['components']['c1']
    if change=='partial':item['path']['case_analysis']={'space':'some cases','closure':'enumerated_partial','closure_reason':'not all'}
    elif change=='hidden_source':item['path']['premises'][0]['source']='T'
    elif change=='conflict':item['witness']=evidence(1)['witness']
    elif change=='unchanged':
        item=evidence(1);item['witness']['completion_b']['values']['c1']='A'
        r['analysis']['T'][0]['output']['components']['c1']=item
    else:r['analysis']['T'][0]['output']['components']['c1']=None
    assert decide_necessity(r)['text_necessary'] is None


class Scripted:
    model='offline-fixture';identity={'model':'offline-fixture'}
    def __init__(self):self.requests=[]
    def complete(self,request):
        self.requests.append(request)
        if request.metadata['role'].startswith('isolated'):
            value={'answer':'fixture','abstain':False,'premises':[],'missing_facts':[]}
        else:
            removed='T' if 'analysis_T' in request.metadata['role'] else 'I'
            value={'removed':removed,'components':{'c1':evidence(1)},'decisive_facts':[]}
        return Completion(json.dumps(value),self.model,{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5,'network_requests':1})


def test_probe_binding_fixed_n_review_and_resume(tmp_path):
    shutil.copytree(ROOT/'prompts',tmp_path/'prompts');(tmp_path/'configs').mkdir()
    shutil.copy(ROOT/'configs/evaluation_v2.json',tmp_path/'configs/evaluation_v2.json')
    (tmp_path/'image.png').write_bytes((ROOT/'tests/fixtures/stage_a/images/m_thermo.png').read_bytes())
    material={'material_id':'dev','source_group_id':'g','split':'train','text':'unique material fact',
        'image_path':'image.png','image_sha256':sha256_file(tmp_path/'image.png'),'provenance':'synthetic','text_source':'synthetic'}
    write_jsonl(tmp_path/'materials.jsonl',[material])
    who={'source':'user_delegated_ai','user_authorized':True,'authorization':'synthetic explicit delegation','reviewer':'fixture'}
    cases=[{'case_id':'E1','material_id':'dev','question':'fixture Q?', 'reference':{'status':'verified','validity':1,
        'components':[{'id':'c1','asks':'fixture','value':'fixture'}],'review':who}},
        {'case_id':'E8','material_id':'dev','question':'unresolved Q?', 'reference':{'status':'unresolved','validity':None,
        'components':[{'id':'c1','asks':'fixture','value':None}],'review':who}}]
    write_jsonl(tmp_path/'cases.jsonl',cases)
    expected=[{'case_id':'E1','sentinel':'EXPECTED_SECRET'}, {'case_id':'E8'}];write_jsonl(tmp_path/'expected.jsonl',expected)
    kw=dict(run_id='probe',project_root=tmp_path)
    init_necessity(**kw,materials_path='materials.jsonl',cases_path='cases.jsonl',expected_path='expected.jsonl')
    provider=Scripted();assert run_necessity(**kw,provider=provider)['new_steps']==2
    out=tmp_path/'outputs/probe';rows=read_jsonl(out/'records.jsonl');assert len(rows)==2
    assert all('EXPECTED_SECRET' not in r.prompt for r in provider.requests)
    assert rows[1]['analysis']=={'T':[],'I':[]}  # unresolved validity spends zero calls
    hashes={f:sha256_file(out/f) for f in ('records.jsonl','calls.jsonl','plan.jsonl')}
    assert run_necessity(**kw,provider=provider)['new_steps']==0 and len(provider.requests)==2
    assert hashes=={f:sha256_file(out/f) for f in hashes}
    reviews=read_jsonl(out/'evidence_review_template.jsonl')
    for r in reviews:r.update(result='supported',reason='synthetic evidence verification',**who)
    bad=copy.deepcopy(reviews);bad[-1]['question_sha256']='foreign';write_jsonl(tmp_path/'bad.jsonl',bad)
    with pytest.raises(ValueError,match='binding'):review_necessity(**kw,reviews_path='bad.jsonl')
    assert read_jsonl(out/'records.jsonl')==rows
    write_jsonl(tmp_path/'reviews.jsonl',reviews);assert review_necessity(**kw,reviews_path='reviews.jsonl')['updated_evidence_items']==2
    assert review_necessity(**kw,reviews_path='reviews.jsonl')['updated_evidence_items']==0
    assert decide_necessity(read_jsonl(out/'records.jsonl')[0],confirmed=True)['integration']==1
    # A rejected shortcut fact alone must not spend an isolated-answer follow-up.
    before=read_jsonl(out/'records.jsonl');r=copy.deepcopy(before[0]);extra=copy.deepcopy(r['reviews'][0])
    extra.update(evidence_key='fact:f1',result='not_supported',review_id='fixture-fact')
    r['reviews'].append(extra);write_jsonl(out/'records.jsonl',[r,before[1]])
    assert run_necessity(**kw,provider=provider,follow_up=True)['new_steps']==0
    write_jsonl(out/'records.jsonl',before)
    # A rejected witness becomes unresolved; it never becomes a sufficiency proof.
    r=read_jsonl(out/'records.jsonl')[0]
    next(x for x in r['reviews'] if x['removed']=='T')['result']='not_supported'
    assert decide_necessity(r,confirmed=True)['text_necessary'] is None
    # Follow-up solver receives the retained modality and Q, without reference/analysis/labels.
    write_jsonl(out/'records.jsonl',[r,rows[1]])
    before_calls=len(provider.requests)
    limited=run_necessity(**kw,provider=provider,follow_up=True,max_requests=3)
    assert limited['status']=='budget_exhausted' and limited['new_steps']==0
    assert len(provider.requests)==before_calls  # reserve the two-call follow-up as a unit
    assert run_necessity(**kw,provider=provider,follow_up=True)['new_steps']==2
    isolated=next(req for req in provider.requests if req.metadata['role'].startswith('isolated'))
    assert isolated.text is None and isolated.image_path and isolated.condition=='I'
    assert 'REFERENCE' not in isolated.prompt and 'EXPECTED_SECRET' not in isolated.prompt
    count=len(provider.requests);assert run_necessity(**kw,provider=provider,follow_up=True)['new_steps']==0
    assert len(provider.requests)==count
    # Changing a supplied reference refuses resume before any new call.
    rows=read_jsonl(out/'records.jsonl');rows[0]['reference']['components'][0]['value']='changed'
    write_jsonl(out/'records.jsonl',rows)
    with pytest.raises(ValueError,match='binding'):run_necessity(**kw,provider=provider)
