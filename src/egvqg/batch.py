"""Frozen-input stage-two batches and auditable evaluation-only retries."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from .experiments import TEMPLATES, STEPS, run_experiment
from .metrics import compute_record_metrics
from .prompts import PromptSet
from .providers import JudgeProvider
from .records import (ExperimentStore, canonical_json, read_jsonl, sha256_file, sha256_text,
                      utc_now, write_json, write_jsonl)

FROZEN = ('frozen_human_reviewed','frozen_delegated_reviewed')


def bind_batch_contract(*, run_id, evaluation_run, materials_path, repeatability_run,
                        project_root='.', output_root='outputs', provider=None):
    root=Path(project_root).resolve();out=root/output_root/run_id;store=ExperimentStore(out)
    evaluation=root/output_root/evaluation_run; em=json.loads((evaluation/'manifest.json').read_text())
    if em.get('evaluation_status') not in FROZEN or sha256_file(evaluation/'records.jsonl')!=em.get('frozen_records_sha256'):
        raise ValueError('evaluation protocol and reviewed controls must be frozen first')
    protocol=Path(em['evaluation_protocol_path'])
    if sha256_file(protocol)!=em['evaluation_protocol_sha256']:
        raise ValueError('frozen evaluation protocol changed')
    material_file=root/materials_path; mm=json.loads(material_file.with_name('manifest.json').read_text())
    if mm.get('status') not in FROZEN or sha256_file(material_file)!=mm.get('materials_sha256'):
        raise ValueError('the 30 material selection must be frozen first')
    material_rows=read_jsonl(material_file)
    if len(material_rows)!=30 or len({r['image_sha256'] for r in material_rows})!=30:
        raise ValueError('expected 30 unique frozen source images')
    if read_jsonl(out/'materials.jsonl')!=material_rows:
        raise ValueError('plan input does not match the frozen 30 materials')
    if len(store.plan)!=120 or any(sum(s['method']==m for s in store.plan)!=30
                                  for m in ('plain','strong','anchored','no_image_anchor')):
        raise ValueError('expected the unchanged four-method 120-slot plan')
    prompts=PromptSet(root/'prompts',names=TEMPLATES)
    if prompts.snapshot_sha256()!=em['frozen_prompt_sha256'] or prompts.snapshot_sha256()!=store.manifest['prompt_sha256']:
        raise ValueError('generation/evaluation prompt version changed')
    repeated=root/output_root/repeatability_run/'reproducibility.json'
    rp=json.loads(repeated.read_text())
    if not rp.get('passed') or len(rp.get('groups',[]))!=3:
        raise ValueError('3x5 fresh-request repeatability validation must pass')
    p=provider or JudgeProvider.from_env(root=root,run_dir=out,visual=True,max_requests=1000,
                                       max_tokens=store.manifest['config']['max_output_tokens'])
    identity=getattr(p,'identity',{'model':p.model})
    if identity!=em['model_identity'] or identity!=rp['model_identity']:
        raise ValueError('model/effective configuration differs from calibration or repeatability test')
    contract={'protocol_path':str(protocol),'protocol_sha256':sha256_file(protocol),
        'evaluation_manifest':str(evaluation/'manifest.json'),'evaluation_manifest_sha256':sha256_file(evaluation/'manifest.json'),
        'evaluation_records':str(evaluation/'records.jsonl'),'evaluation_records_sha256':sha256_file(evaluation/'records.jsonl'),
        'selection_manifest':str(material_file.with_name('manifest.json')),
        'selection_manifest_sha256':sha256_file(material_file.with_name('manifest.json')),
        'materials_path':str(material_file),'materials_sha256':sha256_file(material_file),
        'material_snapshot_sha256':sha256_file(out/'materials.jsonl'),
        'plan_sha256':store.manifest['plan_sha256'],'prompt_sha256':prompts.snapshot_sha256(),
        'config_path':str(root/'configs/experiments.json'),
        'configuration_sha256':sha256_text(canonical_json(store.manifest['config'])),
        'model_identity':identity,'repeatability_path':str(repeated),'repeatability_sha256':sha256_file(repeated),
        'review_sampling':{'seed':20261111,'per_method':8,'policy':'fixed before generation; 26.7% per method'}}
    old=store.manifest.get('batch_contract')
    if old is not None and old!=contract:
        raise ValueError('batch contract changed; refuse to overwrite the original run')
    manifest=store.manifest; manifest['batch_contract']=contract;manifest['model_identity']=identity
    write_json(out/'manifest.json',manifest)
    if not (out/'stage_two_review_plan.jsonl').exists():
        rng=random.Random(20261111); planned=[]
        for method in ('plain','strong'):
            slots=[s for s in store.plan if s['method']==method]
            planned.extend(rng.sample(slots,8))
        write_jsonl(out/'stage_two_review_plan.jsonl',planned)
    return contract


def validate_batch_contract(out, root, *, identity=None):
    manifest=json.loads((out/'manifest.json').read_text()); c=manifest.get('batch_contract')
    if not c:raise ValueError('pilot batch requires a bound frozen evaluation/material/model contract')
    checks=(('protocol_path','protocol_sha256'),('evaluation_manifest','evaluation_manifest_sha256'),
            ('evaluation_records','evaluation_records_sha256'),('selection_manifest','selection_manifest_sha256'),
            ('materials_path','materials_sha256'),('repeatability_path','repeatability_sha256'))
    for path_key,hash_key in checks:
        if sha256_file(Path(c[path_key]))!=c[hash_key]:raise ValueError(f'frozen {path_key} changed')
    if sha256_file(out/'materials.jsonl')!=c['material_snapshot_sha256']:
        raise ValueError('material snapshot changed')
    if sha256_text(canonical_json(read_jsonl(out/'plan.jsonl')))!=c['plan_sha256']:
        raise ValueError('planned denominator changed')
    if PromptSet(root/'prompts',names=TEMPLATES).snapshot_sha256()!=c['prompt_sha256']:
        raise ValueError('current prompt version changed')
    current=json.loads(Path(c['config_path']).read_text())
    if sha256_text(canonical_json(current))!=c['configuration_sha256'] or current!=manifest['config']:
        raise ValueError('frozen model/run configuration changed')
    if identity is not None and identity!=c['model_identity']:
        raise ValueError('actual model configuration changed')
    return c


def run_stage_two(*, run_id, evaluation_run, materials_path, repeatability_run='repeatability-3x5',
                  method='both', max_requests=1000, dry_run=False, project_root='.', output_root='outputs',provider=None):
    root=Path(project_root).resolve();out=root/output_root/run_id
    bind_batch_contract(run_id=run_id,evaluation_run=evaluation_run,materials_path=materials_path,
                        repeatability_run=repeatability_run,project_root=root,output_root=output_root,provider=provider)
    methods=('plain','strong') if method=='both' else (method,)
    if any(m not in ('plain','strong') for m in methods):raise ValueError('stage two only supports plain/strong')
    if dry_run:
        validate_batch_contract(out,root)
        return {'methods':methods,'planned_per_method':30,'full_plan_n':120,'http_budget':max_requests,
                'logical_call_upper_without_retries':sum(30*15 for _ in methods),'cost':None}
    results=[]
    for m in methods:
        if m=='strong' and any(r['status']!='automatic_complete' for r in ExperimentStore(out).rows() if r['method']=='plain'):
            raise ValueError('complete all 30 plain attempts before starting strong')
        results.append(run_experiment(run_id=run_id,methods=(m,),max_requests=max_requests,
                                      project_root=root,output_root=output_root,provider=provider))
        stage_two_report(out)
    return {'results':results,'report':str(out/'stage_two_summary.md')}


def stage_two_report(out):
    rows=[r for r in ExperimentStore(out).rows() if r['method'] in ('plain','strong')]
    by={m:compute_record_metrics([r for r in rows if r['method']==m]) for m in ('plain','strong')}
    write_json(out/'stage_two_metrics.json',{'n_planned':60,'label_source':'automatic',
        'by_method':by,'incomplete':sum(r['status']!='automatic_complete' for r in rows)})
    lines=['# 阶段二自动结果','','全部分母按每方法30，失败/无效/未决均保留。自动标签仍需固定抽查校准。','',
      '| 方法 | 有效率 | BDR | BDR未决 | TI正确 | T正确 | I正确 | 0正确 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    def count(x):return f"{x['n_pass']}/30 [{x['lower']:.3f},{x['upper']:.3f}]"
    for m,s in by.items():
        lines.append('| '+m+' | '+count(s['validity'])+' | '+count(s['bdr'])+f" | {s['bdr']['n_unresolved']}/30 | "+
                     ' | '.join(count(s['conditions'][c]) for c in ('TI','T','I','0'))+' |')
    lines+=['','| 方法 | 仅文本 | 仅图像 | 常识替代 | 题干泄露 | 独立拼接 |','|---|---:|---:|---:|---:|---:|']
    for m,s in by.items():lines.append('| '+m+' | '+' | '.join(count(s['shortcuts'][k]) for k in (
        'text_only','image_only','common_knowledge','question_leakage','independent_stitching'))+' |')
    lines+=['','所有原始结果在主records.jsonl，另60个锚定名额仍为pending，不混入本表。',
            '抽查名额在生成前固定为每方法8题，不按支持预期与否选题。']
    (out/'stage_two_summary.md').write_text('\n'.join(lines))
    # Export the 60 preplanned stage-two rows as a derived cohort with its own exact plan.
    export=out/'stage_two';export.mkdir(exist_ok=True)
    plan=[s for s in ExperimentStore(out).plan if s['method'] in ('plain','strong')]
    write_jsonl(export/'plan.jsonl',plan);write_jsonl(export/'records.jsonl',rows)
    write_json(export/'manifest.json',{'plan_sha256':sha256_text(canonical_json(plan)),
        'fingerprint':sha256_text(canonical_json({'source':str(out),'plan':plan})),
        'source_run':out.name,'n_planned':60,'derived_export':True})
    paired={}
    for r in rows:paired.setdefault(r['material_id'],{})[r['method']]=r
    selected=[]
    predicates=(('plain_shortcut',lambda a,b:a['any_shortcut']==1),
        ('strong_improved',lambda a,b:a['any_shortcut']==1 and b['validity']==1 and all(b['labels'][k]==1 for k in ('text_necessary','image_necessary','integration'))),
        ('strong_still_shortcut',lambda a,b:b['any_shortcut']==1))
    from .metrics import record_states
    for name,predicate in predicates:
        match=next(((mid,pair) for mid,pair in paired.items() if set(pair)=={'plain','strong'} and
                    predicate(record_states(pair['plain']),record_states(pair['strong']))),None)
        selected.append({'type':name,'case':None if match is None else {'material_id':match[0],
                         'plain_attempt_id':match[1]['plain']['attempt_id'],'strong_attempt_id':match[1]['strong']['attempt_id'],
                         'plain_q':match[1]['plain']['question']['normalized_text'],'strong_q':match[1]['strong']['question']['normalized_text']}})
    write_json(out/'stage_two_cases.json',{'source':'automatic','cases':selected})


def retry_failed_evaluation(*, run_id, attempt_ids, steps, project_root='.', output_root='outputs'):
    """Explicit evaluation-only retry; preserve Q, original raw failures and fixed N."""
    root=Path(project_root).resolve();out=root/output_root/run_id;store=ExperimentStore(out)
    if not steps or any(s not in STEPS or s=='generation' for s in steps):
        raise ValueError('only explicit evaluation steps may be retried; generation is never replaced')
    changed=0
    for aid in attempt_ids:
        r=store.get(aid);previous=copy.deepcopy(r)
        if not r['question']['normalized_text']:raise ValueError('cannot regenerate a failed question')
        if any(r['steps'][s] not in ('failed','blocked') for s in steps):
            raise ValueError('only failed/blocked steps may be explicitly retried')
        reset=set(steps)
        for s in steps:
            if s in ('candidate_answer','independent_answer'):
                reset.update(('reference_resolution','necessity','shortcuts',*[f'score_{c}' for c in ('TI','T','I','0')]))
            if s.startswith('answer_'):reset.update(('score_'+s.removeprefix('answer_'),'shortcuts'))
            if s.startswith('score_'):reset.add('shortcuts')
        r['notes'].append({'event':'explicit_evaluation_retry','steps':sorted(reset),
                           'prior_states':{s:r['steps'][s] for s in reset},'created_at':utc_now()})
        for s in reset:r['steps'][s]='pending'
        r['evaluation_revision']=r.get('evaluation_revision',0)+1;r['status']='running'
        store.put(r,previous=previous);changed+=1
    return {'attempts_reset':changed,'generation_replaced':False}
