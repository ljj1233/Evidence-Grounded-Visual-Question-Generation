"""Sampling denominators, uncertainty, malformed decisions and interruption recovery."""
import json
import zipfile
from pathlib import Path

import pytest

from egvqg.dataset import REVISION, import_scienceqa
from egvqg.pipeline import RunConfigMismatch
from egvqg.providers import Completion, RequestBudgetExceeded
from egvqg.records import read_jsonl
from egvqg.screening import CHECKS, combine_screen, estimate_yield, estimate_reviewed_yield, screen_sample, validate_screen, wilson_interval

ROOT=Path(__file__).resolve().parents[1]


def test_yield_denominator_and_unresolved_are_not_dropped():
    result=estimate_yield([{'decision':s} for s in ('pass','fail','unresolved')],100,{'train':100,'val':40,'test':30})
    assert result['strict_automatic_acceptance_rate']==1/3
    assert result['unresolved_acceptance_bounds']==[1/3,2/3]
    assert result['training_projection']['point_strict_rows']==33
    assert result['training_projection']['point_if_all_unresolved_pass_rows']==67
    assert result['other_split_scenarios_same_retention_assumption']['test']['point_strict_rows']==10
    assert result['sampling_and_unresolved_envelope'][0] < 1/3
    assert result['sampling_and_unresolved_envelope'][1] > 2/3
    assert wilson_interval(0,30)==pytest.approx([0,.1135133931739688])
    assert wilson_interval(30,30)==pytest.approx([.8864866068260312,1])
    assert wilson_interval(5,10)==pytest.approx([.236593090512564,.763406909487436])
    census=estimate_yield([{'decision':'pass'},{'decision':'fail'}],2,{'train':2})
    assert census['wilson95_strict_acceptance']==[.5,.5]


def test_full_sample_review_keeps_original_labels_and_pending_denominator():
    original=[{'material_id':str(i),'decision':'pass'} for i in range(3)]
    reviews=[{'material_id':str(i),'decision':decision,'reason':'evidence',
              'witness':{'question':'a question','answer':'an answer'} if decision=='pass' else None,
              'review_source':'codex_visual_review_not_human_gold','human_review_status':'pending'}
             for i,decision in enumerate(('pass','fail','unresolved'))]
    stats=estimate_reviewed_yield(original,reviews,100,{'train':100})
    assert stats['n_pass']==1 and stats['n_unresolved']==1 and stats['n_changed_from_endpoint']==2
    assert all(r['decision']=='pass' for r in original)
    for invalid in (reviews[:2],[reviews[0],reviews[0],reviews[2]]):
        with pytest.raises(ValueError,match='every sampled material'):
            estimate_reviewed_yield(original,invalid,100,{'train':100})


@pytest.mark.parametrize('first,audit,expected',[
    ('suitable','suitable','pass'),('unsuitable','unsuitable','fail'),
    ('suitable','unsuitable','unresolved'),('unsuitable','suitable','unresolved'),
    ('suitable','unresolved','unresolved'),('unresolved','unresolved','unresolved')])
def test_disagreement_stays_unresolved(first,audit,expected):
    assert combine_screen({'status':first},{'status':audit})==expected
    with pytest.raises(ValueError):
        validate_screen({'status':'suitable','reason':'x','exclusion_code':'none',
                         'checks':{k:True for k in CHECKS},'witness':None})


def test_random_screen_resume_no_resampling_and_no_original_answers(tmp_path):
    raw=tmp_path/'data/raw/scienceqa'/REVISION;raw.mkdir(parents=True)
    problems={str(i):dict(split='train',hint=f'material {i}',image='image.png',subject='natural science',
                          question='ORIGINAL_SECRET',answer='ORIGINAL_SECRET',solution='ORIGINAL_SECRET') for i in range(1,5)}
    (raw/'problems.json').write_text(json.dumps(problems))
    (raw/'pid_splits.json').write_text(json.dumps({'train':list(problems),'val':[],'test':[]}))
    (raw/'LICENSE-DATA').write_text('offline fixture')
    image=(ROOT/'tests/fixtures/stage_a/images/m_thermo.png').read_bytes()
    with zipfile.ZipFile(raw/'train.zip','w') as z:
        for pid in problems:z.writestr(f'train/{pid}/image.png',image)
    (tmp_path/'prompts').mkdir()
    (tmp_path/'prompts/screen_material.txt').write_text((ROOT/'prompts/screen_material.txt').read_text())
    selected=import_scienceqa(tmp_path,limit=3,seed=42,offline=True)
    ids=[r['source_id'] for r in read_jsonl(Path(selected['materials']))]
    class Fake:
        model='offline';identity={'adapter':'offline'}
        def __init__(self,cap=100):self.requests=[];self.cap=cap
        def complete(self,request):
            if len(self.requests)>=self.cap:raise RequestBudgetExceeded('test budget')
            self.requests.append(request)
            assert 'ORIGINAL_SECRET' not in request.prompt
            assert 'FIRST_REASON_CANARY' not in request.prompt
            pid=request.text.rsplit(' ',1)[1]
            status='suitable' if pid==ids[0] or (pid==ids[2] and request.task=='audit_material') else 'unsuitable'
            checks={k:True for k in CHECKS}
            witness={k:'independent witness' for k in ('question','answer','rubric','text_evidence','image_evidence')}
            if status=='unsuitable':checks['integrated']=False;witness=None
            data={'status':status,'reason':'FIRST_REASON_CANARY','checks':checks,'witness':witness,
                  'exclusion_code':'none' if status=='suitable' else 'no_joint_goal'}
            return Completion(json.dumps(data),self.model)
    kwargs=dict(run_id='screen',project_root=tmp_path,limit=3,seed=42)
    with pytest.raises(RequestBudgetExceeded):screen_sample(**kwargs,provider=Fake(1))
    second=Fake();result=screen_sample(**kwargs,provider=second)
    assert second.requests[0].task=='audit_material'
    assert result['estimate']['n_sampled']==3
    assert (result['estimate']['n_pass'],result['estimate']['n_fail'],result['estimate']['n_unresolved'])==(1,1,1)
    sample_before=(tmp_path/'outputs/screen/sampling.json').read_bytes()
    again=Fake();resumed=screen_sample(**kwargs,provider=again)
    assert resumed['new_calls']==resumed['new_records']==0 and not again.requests
    assert sample_before==(tmp_path/'outputs/screen/sampling.json').read_bytes()
    with pytest.raises(RunConfigMismatch):screen_sample(**{**kwargs,'seed':43},provider=Fake())
