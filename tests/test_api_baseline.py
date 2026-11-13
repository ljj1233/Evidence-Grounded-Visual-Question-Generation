"""Full test baseline: fixed denominator, role isolation, failed evidence and resume."""
import json
import shutil

import pytest

from egvqg.api_baseline import NAMES,init_api_baseline,run_api_baseline
from egvqg.providers import Completion
from egvqg.records import read_jsonl,sha256_file,write_jsonl
from test_faithfulness import ROOT


@pytest.mark.parametrize('bad_review',[False,True])
def test_api_baseline_fixed_n_isolation_partial_run_failure_and_resume(tmp_path,bad_review):
    (tmp_path/'prompts').mkdir();(tmp_path/'configs').mkdir()
    for name in NAMES:shutil.copy(ROOT/f'prompts/{name}.txt',tmp_path/'prompts')
    shutil.copy(ROOT/'configs/necessity_assessment_v2.json',tmp_path/'configs')
    image=tmp_path/'image.png';shutil.copy(ROOT/'tests/fixtures/stage_a/images/m_thermo.png',image)
    materials=[{'material_id':f'm{i}','source_group_id':f'group{i}','split':'test',
        'text':'TEXT_SENTINEL','image_path':'image.png','image_sha256':sha256_file(image),
        'provenance':'offline fixture','text_source':'hint'} for i in range(2)]
    write_jsonl(tmp_path/'materials.jsonl',materials)
    args=dict(run_id='api',project_root=tmp_path)
    init_api_baseline(**args,materials_path='materials.jsonl',workers=1)
    calls=[]
    class Mock:
        model='offline';identity={'model':'offline'}
        def complete(self,r):
            calls.append(r);role=r.metadata['role']
            if role=='generation':text='What is measured?'
            else:
                if role=='candidate_answer':d={'status':'answerable','answer':'REFERENCE_SENTINEL','rubric':['fixture'],
                    'acceptable_answers':['REFERENCE_SENTINEL'],'text_evidence':[],'image_evidence':[]}
                elif role=='faithfulness':
                    if bad_review:return Completion('{malformed',self.model,{'network_requests':1,'total_tokens':2})
                    d={'question_checks':{k:{'value':1,'reason':'fixture'} for k in ('presupposition_supported','answer_supported','scope_clear')},
                       'reference_supported':{'value':1,'reason':'fixture'},'reason':'fixture','evidence':[],'failure_codes':[]}
                elif role.startswith('answer_'):d={'answer':None if r.condition=='T' else 'fixture',
                    'abstain':r.condition=='T','reason_brief':'fixture answer'}
                elif role.startswith('score_'):d={'correct':1,'reason':'fixture'}
                elif role.startswith('analysis_'):d={'kind':'unique','answer':'fixture','reason':'fixture sufficient'}
                else:d={**{k:'pass' for k in ('question_preserved','retained_preserved','well_posed','premises_supported','conclusion_checked')},
                        'recomputed_answers':['fixture'],'reason':'fixture'}
                text=json.dumps(d)
            return Completion(text,self.model,{'network_requests':1,'total_tokens':2})
    mock=Mock();factory=lambda item,tokens:mock
    first=run_api_baseline(**args,limit=1,provider_factory=factory)
    assert first['n_planned']==2 and first['completed']==1
    assert first['answerability']['unresolved']==(2 if bad_review else 1)
    result=run_api_baseline(**args,limit=2,provider_factory=factory)
    assert result['completed']==2 and result['hard_http_cap']==30
    assert len(calls)==(14 if bad_review else 28)
    rows=read_jsonl(tmp_path/'outputs/api/records.jsonl')
    for row in rows:
        assert all(v['value'] is None for v in row['labels']['adjudicated'].values())
        assert row['responses']['T']['status']=='refused' and row['responses']['T']['correct'] is None
        if bad_review:
            assert row['labels']['automatic']['validity']['value'] is None
            assert row['labels']['automatic']['text_necessary']['value'] is None
    for r in calls:
        role=r.metadata['role']
        if role=='generation' or role.startswith(('answer_','analysis_','verification_')):
            assert 'REFERENCE_SENTINEL' not in r.prompt
        if r.condition=='I':assert r.image_path and r.text is None and 'TEXT_SENTINEL' not in r.prompt
        if r.condition=='T':assert r.image_path is None and r.text=='TEXT_SENTINEL'
        if r.condition=='0':assert r.image_path is None and r.text is None and 'TEXT_SENTINEL' not in r.prompt
    out=tmp_path/'outputs/api';before={p:sha256_file(p) for p in out.glob('items/*/calls.jsonl')};count=len(calls)
    run_api_baseline(**args,limit=2,provider_factory=factory)
    assert len(calls)==count and before=={p:sha256_file(p) for p in before}
    if bad_review:
        assert any(c['status']=='parse_failure' and c['raw_output']=='{malformed' for c in read_jsonl(out/'calls.jsonl'))
