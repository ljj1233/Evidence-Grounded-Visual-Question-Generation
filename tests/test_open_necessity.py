"""Open witnesses must retain fixed slots, isolated inputs and reviewed provenance."""
import json
import shutil
from pathlib import Path

import pytest

from egvqg.open_necessity import init_open_probe,run_open_probe,report_open_probe
from egvqg.providers import Completion
from egvqg.records import canonical_json,read_jsonl,sha256_file,sha256_text,write_jsonl


def test_open_probe_isolation_uncertainty_resume_and_exact_review_binding(tmp_path):
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'prompts').mkdir()
    shutil.copy(root/'prompts/open-necessity-v1.txt',tmp_path/'prompts')
    shutil.copy(root/'tests/fixtures/stage_a/images/m_thermo.png',tmp_path/'image.png')
    case={'case_id':'dev','question':'What is the quantity?','text':'TEXT_SENTINEL',
          'image_path':'image.png','image_sha256':sha256_file(tmp_path/'image.png')}
    key={'case_id':'dev','expected':{'T':1,'I':0},'reference':'KEY_REFERENCE_SENTINEL'}
    write_jsonl(tmp_path/'cases.jsonl',[case]);write_jsonl(tmp_path/'key.jsonl',[key])
    init_open_probe(run_id='probe',cases_path='cases.jsonl',key_path='key.jsonl',project_root=tmp_path)
    class Mock:
        model='offline';identity={'model':'offline'}
        def __init__(self):self.requests=[]
        def complete(self,request):
            self.requests.append(request)
            output=({'kind':'ambiguous','scenario_a':'x=1','scenario_b':'x=2',
                     'answer_a':'1','answer_b':'2','reason':'fixture'} if request.condition=='I'
                    else {'kind':'unresolved','reason':'fixture unknown'})
            return Completion(json.dumps(output),self.model,{'network_requests':1,'prompt_tokens':2,'completion_tokens':2,'total_tokens':4})
    mock=Mock();run_open_probe(run_id='probe',project_root=tmp_path,provider=mock)
    i,t=mock.requests
    assert i.image_path and i.text is None and 'TEXT_SENTINEL' not in i.prompt
    assert t.image_path is None and 'TEXT_SENTINEL' in t.prompt
    assert all('KEY_REFERENCE_SENTINEL' not in r.prompt for r in mock.requests)
    out=tmp_path/'outputs/probe';r=read_jsonl(out/'records.jsonl')[0]
    result=report_open_probe(run_id='probe',project_root=tmp_path)
    assert result['confirmed_necessity']['T']['unresolved']==1  # raw witness is not self-certified
    review={'case_id':'dev','attempt_id':r['attempt_id'],'removed':'T','analysis_sha256':sha256_text(canonical_json(r['analysis']['T'])),
            'result':'supported','source':'user_delegated_ai','reviewer':'fixture','reason':'fixture semantic check'}
    write_jsonl(out/'evidence_reviews.jsonl',[review])
    assert report_open_probe(run_id='probe',project_root=tmp_path)['confirmed_necessity']['T']['pass']==1
    assert report_open_probe(run_id='probe',project_root=tmp_path)['confirmed_necessity']['I']['unresolved']==1
    before={f:sha256_file(out/f) for f in ('plan.jsonl','records.jsonl','calls.jsonl')}
    assert run_open_probe(run_id='probe',project_root=tmp_path,provider=mock)['new_steps']==0
    assert len(mock.requests)==2 and before=={f:sha256_file(out/f) for f in before}
    review['analysis_sha256']='stale';write_jsonl(out/'evidence_reviews.jsonl',[review])
    with pytest.raises(ValueError,match='binding'):
        report_open_probe(run_id='probe',project_root=tmp_path)
    (tmp_path/'image.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):
        run_open_probe(run_id='probe',project_root=tmp_path,provider=mock)
