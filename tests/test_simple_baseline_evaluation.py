"""Post-batch gate, separate answer support, isolated proofs and conservative short circuit."""
import json
import shutil

import pytest

from egvqg.providers import Completion
from egvqg.records import read_jsonl,sha256_file,write_jsonl
from egvqg.simple_baseline import init_simple_baseline,run_simple_baseline
from egvqg.simple_baseline_evaluation import init_evaluation,run_evaluation,NAMES
from test_simple_baseline import project,Mock
from test_faithfulness import ROOT


@pytest.mark.parametrize('scenario',['sufficient','reject_first','malformed'])
def test_evaluation_waits_for_full_batch_and_keeps_evidence_failures_unresolved(tmp_path,scenario):
    project(tmp_path,2)
    for n in NAMES:shutil.copy(ROOT/f'prompts/{n}.txt',tmp_path/'prompts')
    (tmp_path/'configs').mkdir();shutil.copy(ROOT/'configs/necessity_assessment_v2.json',tmp_path/'configs')
    init_simple_baseline(run_id='source',materials_path='materials.jsonl',project_root=tmp_path,concurrency=2)
    args=dict(run_id='evaluation',source_run='source',project_root=tmp_path,concurrency=2)
    with pytest.raises(ValueError,match='must finish'):init_evaluation(**args)
    mock=Mock();run_simple_baseline(run_id='source',project_root=tmp_path,provider_factory=lambda p,t:mock)
    source=tmp_path/'outputs/source/records.jsonl';rows=read_jsonl(source)
    for row in rows:row['responses']['TI']['answer']='BASELINE_ANSWER_SENTINEL'
    write_jsonl(source,rows);before=sha256_file(source)
    init_evaluation(**args);calls=[]
    class Judge:
        model='offline';identity={'model':'offline'}
        def complete(self,r):
            calls.append(r);role=r.metadata['role']
            if role=='faithfulness':
                if scenario=='malformed':return Completion('{bad',self.model,{'network_requests':1,'total_tokens':1})
                d={'question_checks':{k:{'value':1,'reason':'fixture'} for k in ('presupposition_supported','answer_supported','scope_clear')},
                   'reference_supported':{'value':0,'reason':'fixture answer wrong'},'reason':'fixture Q supported'}
            elif role.startswith('analysis_'):d={'kind':'unique','answer':'fixture','reason':'fixture sufficient'}
            else:
                fail=scenario=='reject_first' and role=='verification_I'
                d={**{k:'fail' if fail else 'pass' for k in ('question_preserved','retained_preserved','well_posed','premises_supported','conclusion_checked')},
                   'recomputed_answers':[] if fail else ['fixture'],'reason':'fixture gate'}
            return Completion(json.dumps(d),self.model,{'network_requests':1,'total_tokens':1})
    judge=Judge();runargs=dict(run_id='evaluation',project_root=tmp_path,provider_factory=lambda p,t:judge)
    result=run_evaluation(**runargs)
    expected=2 if scenario=='malformed' else 10 if scenario=='reject_first' else 6
    assert len(calls)==expected and result['completed']==2 and result['n_planned']==2
    assessed=read_jsonl(tmp_path/'outputs/evaluation/records.jsonl')
    for r in assessed:
        if scenario=='malformed':assert r['both_required'] is None and r['image_necessary'] is None
        else:
            assert r['question_answerability']==1 and r['TI_answer_support']==0 and r['both_required']==0
            assert (r['image_necessary'],r['text_necessary'])==((None,0) if scenario=='reject_first' else (0,None))
    for r in calls:
        if r.metadata['role']!='faithfulness':
            assert 'BASELINE_ANSWER_SENTINEL' not in r.prompt
            if r.condition=='I':assert r.text is None and 'TEXT_SENTINEL' not in r.prompt.replace(r.question,'') and r.image_path
            if r.condition=='T':assert r.image_path is None and r.text
    count=len(calls);run_evaluation(**runargs)
    assert len(calls)==count and sha256_file(source)==before


@pytest.mark.parametrize('screen_cuts',[True,False])
def test_cascade_screens_first_and_proposes_witnesses_only_for_survivors(tmp_path,screen_cuts):
    project(tmp_path,2)
    for n in NAMES+('ensemble-sufficiency-v1',):shutil.copy(ROOT/f'prompts/{n}.txt',tmp_path/'prompts')
    (tmp_path/'configs').mkdir();shutil.copy(ROOT/'configs/necessity_assessment_v2.json',tmp_path/'configs')
    init_simple_baseline(run_id='source',materials_path='materials.jsonl',project_root=tmp_path,concurrency=2)
    mock=Mock();run_simple_baseline(run_id='source',project_root=tmp_path,provider_factory=lambda p,t:mock)
    init_evaluation(run_id='evaluation',source_run='source',project_root=tmp_path,concurrency=2,cascade=True)
    calls=[]
    class Judge:
        model='offline';identity={'model':'offline'}
        def complete(self,r):
            calls.append(r);role=r.metadata['role']
            if role=='faithfulness':
                d={'question_checks':{k:{'value':1,'reason':'fixture'} for k in ('presupposition_supported','answer_supported','scope_clear')},
                   'reference_supported':{'value':1,'reason':'fixture'},'reason':'fixture Q supported'}
            elif role.startswith('screen_'):d={'determinable':screen_cuts,'answer':'fixture' if screen_cuts else None,'reason':'fixture'}
            else:d={'kind':'ambiguous','scenario_a':'A','scenario_b':'B','answer_a':'x','answer_b':'y','reason':'fixture'}
            return Completion(json.dumps(d),self.model,{'network_requests':1,'total_tokens':1})
    judge=Judge();run_evaluation(run_id='evaluation',project_root=tmp_path,provider_factory=lambda p,t:judge)
    roles=[r.metadata['role'] for r in calls]
    if screen_cuts:
        assert len(calls)==6 and set(roles)=={'faithfulness','screen_text_only','screen_image_only'}      # 3 calls per item, no proposals
    else:
        assert len(calls)==10 and set(roles)=={'faithfulness','screen_text_only','screen_image_only','analysis_I','analysis_T'}
        assert not any(x.startswith('verification') for x in roles)                                          # verification is left to necessity-ensemble
    for r in calls:
        if r.metadata['role']=='screen_text_only':assert r.condition=='T' and r.image_path is None and r.text
        if r.metadata['role']=='screen_image_only':assert r.condition=='I' and r.text is None and r.image_path
    rows=read_jsonl(tmp_path/'outputs/evaluation/records.jsonl')
    for row in rows:
        assert row['question_answerability']==1 and bool(row.get('screen_cut'))==screen_cuts
        assert row['both_required']==(0 if screen_cuts else None)
    assert json.loads((tmp_path/'outputs/evaluation/status.json').read_text())['screen_cut_single_modality_sufficient']==(2 if screen_cuts else 0)
