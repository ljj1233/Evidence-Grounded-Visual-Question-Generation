"""Targeted audit: fixed selection, isolated inputs, failure propagation and resume."""
import json
import shutil

import pytest

from egvqg.baseline_analysis import init_analysis,run_analysis
from egvqg.faithfulness import apply_review,report_targeted_analysis
from egvqg.providers import Completion,ProviderError
from egvqg.records import canonical_json,read_jsonl,sha256_file,sha256_text,write_json,write_jsonl
from test_faithfulness import ROOT,fixture_record,fixture_review


@pytest.mark.parametrize('fail_analysis',[False,True])
def test_bounded_audit_isolation_no_retries_and_resume(tmp_path,fail_analysis):
    (tmp_path/'configs').mkdir();shutil.copy(ROOT/'configs/necessity_assessment_v2.json',tmp_path/'configs')
    (tmp_path/'prompts').mkdir()
    for name in ('open-necessity-v2','verify-necessity-evidence-v2','review-question-faithfulness-v1'):
        shutil.copy(ROOT/f'prompts/{name}.txt',tmp_path/'prompts')
    rows=[fixture_record(tmp_path,i,v=0 if i==0 else 1) for i in range(3)]
    rows[1]['labels']['adjudicated']['text_necessary'].update(value=None,status='undecided')
    rows[2]['labels']['adjudicated']['image_necessary'].update(value=0,status='decided')
    rows=[apply_review(r,fixture_review(r,gate=0 if i==0 else 1)) for i,r in enumerate(rows)]
    for r in rows:
        r['reference']['candidate_answer']='REFERENCE_SENTINEL'
        r['labels']['adjudicated']['validity']['reason']='OLD_LABEL_SENTINEL'
    source=tmp_path/'outputs/source';source.mkdir(parents=True);write_jsonl(source/'records.jsonl',rows)
    before=sha256_file(source/'records.jsonl')
    args=dict(run_id='audit',project_root=tmp_path)
    result=init_analysis(source_run='source',**args)
    assert result['n_planned_calls']==result['hard_http_cap']==5
    out=tmp_path/'outputs/audit';calls=[]
    class Mock:
        model='offline';identity={'model':'offline'}
        def complete(self,request):
            calls.append(request)
            if fail_analysis and request.metadata['role']=='analysis':raise ProviderError('fixture failure')
            if request.metadata['role']=='analysis':output={'kind':'unique','answer':'fixture','reason':'fixture sufficient'}
            elif request.metadata['role']=='verification':output={**{k:'pass' for k in ('question_preserved','retained_preserved','well_posed','premises_supported','conclusion_checked')},'reason':'fixture','recomputed_answers':['fixture']}
            else:output={'reason':'fixture full material audit'}
            return Completion(json.dumps(output),self.model,{'network_requests':1,'total_tokens':3})
    mock=Mock();run_analysis(**args,provider=mock,max_steps=1)
    run_analysis(**args,provider=mock)
    assert len(calls)==(3 if fail_analysis else 5)
    for r in calls:
        assert 'REFERENCE_SENTINEL' not in r.prompt and 'OLD_LABEL_SENTINEL' not in r.prompt
        if r.condition=='I':assert r.text is None and 'fixture text' not in r.prompt and r.image_path
        if r.condition=='T':assert r.text=='fixture text' and r.image_path is None
    assert len(read_jsonl(out/'records.jsonl'))==5
    if fail_analysis:
        assert sum(r['status']=='skipped_no_complete_proof' for r in read_jsonl(out/'records.jsonl'))==2
    count=len(calls);hashes={p.name:sha256_file(p) for p in (out/'records.jsonl',out/'calls.jsonl')}
    assert run_analysis(**args,provider=mock)['new_steps']==0 and len(calls)==count
    assert hashes=={p.name:sha256_file(p) for p in (out/'records.jsonl',out/'calls.jsonl')}
    assert sha256_file(source/'records.jsonl')==before
    tasks=read_jsonl(out/'records.jsonl');reviews=[]
    for a in (r for r in tasks if r['stage']=='analysis'):
        v=next(r for r in tasks if r['stage']=='verification' and r['removed']==a['removed'])
        reviews.append({'item_id':a['item_id'],'removed':a['removed'],
            'analysis_sha256':sha256_text(canonical_json(a)),
            'verification_sha256':sha256_text(canonical_json(v)),
            'source':'user_delegated_ai','reviewer':'fixture','reason':'fixture review',
            'result':'uncertain' if fail_analysis else 'supported'})
    write_jsonl(out/'evidence_reviews.jsonl',reviews)
    api=next(r for r in tasks if r['stage']=='faithfulness')
    k=json.loads((out/'manifest.json').read_text())['config']['knowledge']
    write_jsonl(out/'validity_reviews.jsonl',[{'item_id':api['item_id'],'value':None,
        'reason':'fixture explicit assumption boundary review',
        'api_record_sha256':sha256_text(canonical_json(api)),
        'question_sha256':rows[0]['question']['sha256'],'knowledge_sha256':sha256_text(k),
        'source':'user_delegated_ai','reviewer':'fixture'}])
    result=report_targeted_analysis(**args)
    assert result['answerability']=={'pass':2,'fail':0,'unresolved':1}
    assessed=read_jsonl(out/'assessed_questions.jsonl')
    assert assessed[0]['labels']['adjudicated']['text_necessary']['value'] is None
    assert assessed[1]['labels']['adjudicated']['text_necessary']['value']==(None if fail_analysis else 0)
    assert len(calls)==count and sha256_file(source/'records.jsonl')==before
    # Real adapter's ledger cap is cumulative and checked before any network send.
    from egvqg.providers import JudgeProvider,RequestBudgetExceeded,CompletionRequest
    write_jsonl(out/'requests.jsonl',[{'event':'started'} for _ in range(5)])
    p=JudgeProvider(model='fixture',base_url='https://example.invalid',api_key='offline-placeholder',
                    cache_dir=tmp_path/'cache',ledger_path=out/'requests.jsonl',max_requests=5,visual=True)
    with pytest.raises(RequestBudgetExceeded):p.complete(CompletionRequest(task='assess_quality',prompt='new'))
