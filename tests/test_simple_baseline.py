"""Simple baseline: async cap, image limits, exact reuse, malformed outputs and resume."""
import json
import shutil
import threading
import time

import pytest
from PIL import Image

from egvqg.providers import Completion
from egvqg.records import read_jsonl,sha256_file,write_jsonl
from egvqg.simple_baseline import bounded_image,init_simple_baseline,run_simple_baseline
from test_faithfulness import ROOT


def project(tmp_path,n):
    (tmp_path/'prompts').mkdir()
    for name in ('plain-v1','solve-v1'):shutil.copy(ROOT/f'prompts/{name}.txt',tmp_path/'prompts')
    image=tmp_path/'image.png';shutil.copy(ROOT/'tests/fixtures/stage_a/images/m_thermo.png',image)
    rows=[{'material_id':f'm{i}','source_group_id':f'g{i}','split':'test','text':f'TEXT_SENTINEL_{i}',
        'image_path':'image.png','image_sha256':sha256_file(image),'provenance':'offline fixture','text_source':'hint'} for i in range(n)]
    write_jsonl(tmp_path/'materials.jsonl',rows);return rows


class Mock:
    model='offline';identity={'model':'offline'}
    def __init__(self,bad=False):self.calls=[];self.bad=bad;self.lock=threading.Lock();self.active=0;self.peak=0
    def complete(self,r):
        with self.lock:self.calls.append(r);self.active+=1;self.peak=max(self.peak,self.active)
        time.sleep(.02)
        with self.lock:self.active-=1
        if r.metadata['role']=='generation':text='Question about '+r.text
        elif self.bad:text='{malformed'
        else:text=json.dumps({'answer':'fixture','abstain':False,'reason_brief':'fixture TI answer'})
        return Completion(text,self.model,{'network_requests':1,'total_tokens':2})


def test_async_sixteen_and_no_evaluation_or_cross_question_leak(tmp_path):
    project(tmp_path,20);args=dict(run_id='simple',project_root=tmp_path)
    init_simple_baseline(**args,materials_path='materials.jsonl',concurrency=16)
    mock=Mock();factory=lambda item,tokens:mock
    result=run_simple_baseline(**args,provider_factory=factory)
    assert result['completed']==20 and result['hard_http_cap']==40 and len(mock.calls)==40
    assert 2<=mock.peak<=16 and result['evaluation_status']=='ready_not_started'
    assert {r.metadata['role'] for r in mock.calls}=={'generation','answer_TI'}
    for r in mock.calls:
        assert r.image_path and r.text
        if r.metadata['role']=='answer_TI':assert r.question=='Question about '+r.text
        else:assert r.question is None
    rows=read_jsonl(tmp_path/'outputs/simple/records.jsonl')
    assert all(r['evaluation_status']=='not_started' and r['responses']['TI']['score_status']=='not_evaluated' for r in rows)
    assert all(all(x['value'] is None for x in r['labels']['automatic'].values()) for r in rows)
    before=len(mock.calls);run_simple_baseline(**args,provider_factory=factory)
    assert len(mock.calls)==before


@pytest.mark.parametrize('bad',[False,True])
def test_exact_reuse_preserves_failures_and_changed_input_invalidates(tmp_path,bad):
    rows=project(tmp_path,2);mock=Mock(bad);factory=lambda item,tokens:mock
    init_simple_baseline(run_id='old',materials_path='materials.jsonl',project_root=tmp_path,concurrency=2)
    run_simple_baseline(run_id='old',project_root=tmp_path,provider_factory=factory)
    assert len(mock.calls)==4
    old=tmp_path/'outputs/old';protected={p:sha256_file(p) for p in old.glob('items/*/calls.jsonl')}
    init_simple_baseline(run_id='reused',materials_path='materials.jsonl',project_root=tmp_path,concurrency=2,reuse_run='old')
    result=run_simple_baseline(run_id='reused',project_root=tmp_path,provider_factory=factory)
    assert len(mock.calls)==4 and result['reused_prior_calls']==4 and result['usage']['total_tokens']==0
    assert result['TI_failures']==(2 if bad else 0)
    if bad:
        copies=[r for p in (tmp_path/'outputs/reused').glob('items/*/reused_calls.jsonl') for r in read_jsonl(p)]
        assert any(r['raw_output']=='{malformed' and r['status']=='parse_failure' for r in copies)
    changed=[{**r,'text':r['text']+' changed'} for r in rows]
    write_jsonl(tmp_path/'changed.jsonl',changed)
    init_simple_baseline(run_id='changed',materials_path='changed.jsonl',project_root=tmp_path,concurrency=2,reuse_run='old')
    result=run_simple_baseline(run_id='changed',project_root=tmp_path,provider_factory=factory)
    assert len(mock.calls)==8 and result['reused_prior_calls']==0
    assert protected=={p:sha256_file(p) for p in protected}


def test_image_bound_and_reference_original_when_compliant(tmp_path):
    project(tmp_path,1)
    source=tmp_path/'noise.png';Image.effect_noise((600,400),100).convert('RGB').save(source)
    m={'material_id':'large','image_path':'noise.png','image_sha256':sha256_file(source)}
    original=source.read_bytes();policy={'max_edge':128,'max_pixels':16384,'max_bytes':16384}
    bounded=bounded_image(tmp_path,m,tmp_path/'bounded',policy)
    p=tmp_path/bounded['image_path']
    with Image.open(p) as im:assert max(im.size)<=128 and im.width*im.height<=16384
    assert p.stat().st_size<=16384 and source.read_bytes()==original and bounded['image_size_control']['changed']
    unchanged=bounded_image(tmp_path,bounded,tmp_path/'unused',policy)
    assert unchanged['image_path']==bounded['image_path'] and not (tmp_path/'unused').exists()
