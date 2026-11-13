"""Closed-world proof checks plus model-input isolation and durable resume."""
import json
import shutil
from pathlib import Path

import pytest

from egvqg.identifiability import (answer_set, finite_oracle, init_identifiability,
                                  run_identifiability, report_identifiability)
from egvqg.providers import Completion
from egvqg.records import read_jsonl, sha256_file, write_jsonl


@pytest.mark.parametrize('mode,expected',[
    ('joint',(1,1,1)), ('image',(0,1,None)), ('redundant',(0,0,None)),
    ('cancel',(0,0,None)), ('stitched',(1,1,0)), ('invalid',(None,None,None))])
def test_target_dependence_not_latent_uncertainty(mode,expected):
    worlds=[]
    for x in (1,2):
        for y in (1,2):
            answer = [x*y] if mode=='joint' else [y]
            if mode in ('cancel','invalid'): answer=[1 if mode=='cancel' else x]
            if mode=='stitched':answer=[x,y]
            worlds.append({'T':{'x':x} if mode not in ('redundant','invalid') else ({'y':y} if mode=='redundant' else {}),
                           'I':{'y':y},'answer':answer})
    result=finite_oracle({'worlds':worlds,'actual_index':0,'components':['a','b'] if mode=='stitched' else ['a']})
    assert tuple(result[k] for k in ('text_necessary','image_necessary','integration'))==expected
    assert answer_set([['0.5'],['1/2']],1)==[('1/2',)]


def test_isolation_fixed_n_parse_failure_resume_and_input_pinning(tmp_path):
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'prompts').mkdir()
    shutil.copy(root/'prompts/identify-answers-v1.txt',tmp_path/'prompts')
    shutil.copy(root/'tests/fixtures/stage_a/images/m_thermo.png',tmp_path/'image.png')
    worlds=[{'T':{'x':x},'I':{'y':y},'answer':[x*y]} for x in (1,2) for y in (1,2)]
    case={'case_id':'synthetic','source_kind':'synthetic','text':'HIDDEN_TEXT_SENTINEL',
          'image_path':'image.png','image_sha256':sha256_file(tmp_path/'image.png'),
          'question':'What is the product?','components':['product'],'completion_profile':'x,y each range over 1,2',
          'worlds':worlds,'actual_index':0}
    write_jsonl(tmp_path/'input.jsonl',[case])
    init_identifiability(run_id='dev',cases_path='input.jsonl',project_root=tmp_path)
    class Mock:
        model='offline';identity={'model':'offline'}
        def __init__(self):self.requests=[]
        def complete(self,request):
            self.requests.append(request)
            text=json.dumps({'answers':[['1'],['2']]}) if request.condition=='I' else 'bad JSON'
            return Completion(text,self.model,{'network_requests':1,'prompt_tokens':3,'completion_tokens':2,'total_tokens':5})
    mock=Mock()
    assert run_identifiability(run_id='dev',project_root=tmp_path,provider=mock)['new_steps']==2
    image_request,text_request=mock.requests
    assert image_request.image_path and image_request.text is None and 'HIDDEN_TEXT_SENTINEL' not in image_request.prompt
    assert text_request.image_path is None and 'HIDDEN_TEXT_SENTINEL' in text_request.prompt
    result=report_identifiability(run_id='dev',project_root=tmp_path)
    assert result['answer_set_checks']['I']=={'n_planned':1,'exact':0,'mismatch':0,'unresolved':1}
    out=tmp_path/'outputs/dev';before={f:sha256_file(out/f) for f in ('plan.jsonl','records.jsonl','calls.jsonl')}
    assert run_identifiability(run_id='dev',project_root=tmp_path,provider=mock)['new_steps']==0
    assert len(mock.requests)==2 and before=={f:sha256_file(out/f) for f in before}
    (tmp_path/'image.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):
        run_identifiability(run_id='dev',project_root=tmp_path,provider=mock)
