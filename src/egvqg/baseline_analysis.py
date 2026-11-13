"""Bounded API analysis of existing baseline defects; no new question generation."""
from __future__ import annotations

import json
from pathlib import Path

from .assessment_rules import load_assessment_config
from .evidence_verification import verification_gate
from .experiments import call_usage
from .faithfulness import separated_assessment
from .pipeline import _validate_run_id
from .prompts import PromptSet
from .providers import JudgeProvider, ModelCaller, call_model
from .records import (atomic_upsert, canonical_json, file_lock, read_jsonl, sha256_file,
                      sha256_text, stable_id, utc_now, write_json, write_jsonl)

VERSION = 'baseline-targeted-api-analysis-v1'
FAITH_TEMPLATE = 'review-question-faithfulness-v1'


def proof_shape(proof):
    if not isinstance(proof, dict) or not isinstance(proof.get('reason'), str) or not proof['reason'].strip():
        return False
    keys = (('answer',) if proof.get('kind') == 'unique' else
            ('scenario_a','answer_a','scenario_b','answer_b') if proof.get('kind') == 'ambiguous' else ())
    return bool(keys) and all(isinstance(proof.get(k), str) and proof[k].strip() for k in keys)


def analysis_input(row, stage, removed, config, proof=None):
    """Only Q/T/I/K; proof checker sees one candidate, never old decisions or reference."""
    q = row['question']['normalized_text']; m = row['material']
    if stage == 'faithfulness':
        return {'template': FAITH_TEMPLATE,
                'variables': {'Q': q, 'T': m['text'], 'KNOWLEDGE': config['knowledge'],
                              'REFERENCE': '未提供参考答案。独立检查原Q是否有材料支持；reference_supported请填null。'},
                'text': m['text'], 'image_path': m['image_path'], 'question': q, 'condition': 'TI'}
    retained = 'I' if removed == 'T' else 'T'
    if stage == 'screen':      # the retained modality alone: is the full answer already determined?
        from .necessity_ensemble import NO_IMAGE_TEXT
        return {'template': 'ensemble-sufficiency-v1',
                'variables': {'Q': q, 'MATERIAL': '<T>' + m['text'] + '</T>' if retained == 'T' else NO_IMAGE_TEXT,
                              'KNOWLEDGE': config['knowledge']},
                'question': q, 'condition': retained, 'text': m['text'] if retained == 'T' else None,
                'image_path': m['image_path'] if retained == 'I' else None}
    text = m['text'] if retained == 'T' else None
    variables = {'Q': q, 'REMOVED': '文本' if removed == 'T' else '图像',
                 'VISIBLE': '<T>'+text+'</T>' if text else '仅附上的原图，没有文本材料。',
                 'KNOWLEDGE': config['knowledge']}
    template = config['analysis_template']
    if stage == 'verification':
        template = config['verification_template']
        keys = ('scenario_a','answer_a','scenario_b','answer_b') if proof['kind'] == 'ambiguous' else ('answer','reason')
        variables['EVIDENCE'] = canonical_json({'proof_form': (
            'two_hypothetical_completions' if proof['kind']=='ambiguous' else 'single_sufficient_solution_no_scenarios_required'),
            **{k: proof[k] for k in keys}})
    return {'template': template, 'variables': variables, 'question': q,
            'condition': retained, 'text': text, 'image_path': m['image_path'] if retained == 'I' else None}


def init_analysis(*, source_run, run_id, project_root='.', output_root='outputs'):
    root=Path(project_root).resolve();_validate_run_id(run_id);_validate_run_id(source_run)
    source=root/output_root/source_run;out=root/output_root/run_id
    if source==out:raise ValueError('use a new analysis run')
    rows=read_jsonl(source/'records.jsonl');dims=separated_assessment(rows)
    config=load_assessment_config(root,'configs/necessity_assessment_v2.json')
    prompts=PromptSet(root/'prompts', names=(FAITH_TEMPLATE,config['analysis_template'],config['verification_template']))
    selected={i['item_id']:i for i in dims['items']};plan=[]
    for row in rows:
        item=selected[row['faithfulness']['review']['item_id']]
        stages=(('faithfulness',None),) if item['answerability']=='fail' else (
            (('analysis','T'),('analysis','I'),('verification','T'),('verification','I'))
            if item['modality_necessity']=='unresolved' else ())
        for stage,removed in stages:
            plan.append({'item_id':item['item_id'],'attempt_id':row['attempt_id'],
                         'stage':stage,'removed':removed,
                         'operation_id':stable_id('targeted',run_id,item['item_id'],stage,str(removed))})
    if not plan:raise ValueError('no selected defects or unresolved cases')
    protected={str(p.relative_to(root)):sha256_file(p) for p in source.rglob('*') if p.is_file() and not p.name.endswith('.lock')}
    fixed={'source_run':source_run,'source_files_sha256':protected,'config':config,'plan':plan,
           'prompts':prompts.snapshot(),'code_sha256':sha256_file(Path(__file__))}
    fingerprint=sha256_text(canonical_json(fixed))
    out.mkdir(parents=True,exist_ok=True)
    with file_lock(out/'.init.lock'):
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text())['fingerprint']!=fingerprint:raise ValueError('analysis configuration changed')
            return {'n_planned_calls':len(plan),'resumed':True}
        write_jsonl(out/'parent_records.jsonl',rows);write_jsonl(out/'plan.jsonl',plan)
        (out/'prompts').mkdir(exist_ok=True)
        for name,txt in prompts.snapshot().items():(out/'prompts'/f'{name}.txt').write_text(txt)
        write_json(out/'manifest.json',{'created_at':utc_now(),'protocol_version':VERSION,
            'fingerprint':fingerprint,'source_run':source_run,'n_planned':len(rows),'n_planned_calls':len(plan),
            'hard_http_cap':len(plan),'max_attempts':1,'model':'gpt-6-luna','model_identities':{},
            'config':config,'source_files_sha256':protected,'code_sha256':fixed['code_sha256'],
            'selection':'All previously failed answerability items and both directions of all answerable necessity-unresolved items; targeted, not representative accuracy estimation.',
            'prior_exposure':'Codex has seen old outputs and labels. API does not see labels, old outputs or references. Not independent human gold.',
            'readings':['Faithfulness API disagreement requires explicit evidence review; no automatic reversal.',
                        'New necessity label requires a complete proof, verifier accept and delegated evidence review.',
                        'Proof failure/technical failure preserves unresolved; never infer opposite necessity.',
                        'Keep all 30 original slots; no regeneration, resampling or expansion; report both dimensions separately.'],
            'frozen_files':{'parent_records.jsonl':sha256_file(out/'parent_records.jsonl'),
                            'plan.jsonl':sha256_file(out/'plan.jsonl')},
            'prompt_sha256':prompts.snapshot_sha256()})
    return {'n_planned_calls':len(plan),'hard_http_cap':len(plan),'resumed':False}


def run_analysis(*,run_id,project_root='.',output_root='outputs',dry_run=False,provider=None,max_steps=None):
    root=Path(project_root).resolve();_validate_run_id(run_id);out=root/output_root/run_id
    manifest=json.loads((out/'manifest.json').read_text());config=manifest['config']
    if sha256_file(Path(__file__))!=manifest['code_sha256']:raise ValueError('analysis code changed')
    for path,digest in manifest['source_files_sha256'].items():
        if sha256_file(root/path)!=digest:raise ValueError('source analysis changed')
    for path,digest in manifest['frozen_files'].items():
        if sha256_file(out/path)!=digest:raise ValueError('frozen analysis input changed')
    prompts=PromptSet(out/'prompts',names=(FAITH_TEMPLATE,config['analysis_template'],config['verification_template']))
    if prompts.snapshot_sha256()!=manifest['prompt_sha256']:raise ValueError('analysis prompts changed')
    plan=read_jsonl(out/'plan.jsonl');rows={r['attempt_id']:r for r in read_jsonl(out/'parent_records.jsonl')}
    done={r['operation_id']:r for r in read_jsonl(out/'records.jsonl')}
    if dry_run:return {'n_planned_calls':len(plan),'pending':len(plan)-len(done),'hard_http_cap':manifest['hard_http_cap']}
    steps=0
    with file_lock(out/'.pipeline.lock'):
        for task in plan:
            if task['operation_id'] in done:continue
            if max_steps is not None and steps>=max_steps:break
            proof=None
            if task['stage']=='verification':
                a=next((r for r in done.values() if r['stage']=='analysis' and r['item_id']==task['item_id'] and r['removed']==task['removed']),None)
                proof=a.get('output') if a else None
                if not a or a['status']!='ok' or not proof_shape(proof):
                    result={**task,'status':'skipped_no_complete_proof','output':None,'call_id':None}
                    atomic_upsert(out/'records.jsonl','operation_id',result);done[task['operation_id']]=result
                    continue
            tokens=config['verification_max_tokens'] if task['stage']=='verification' else config['analysis_max_tokens']
            p=provider or JudgeProvider.from_env(root=root,run_dir=out,max_requests=manifest['hard_http_cap'],
                    max_tokens=tokens,visual=True,model=manifest['model'])
            if isinstance(p,JudgeProvider):p.max_attempts=1
            identity=getattr(p,'identity',{'model':p.model});key='verification' if task['stage']=='verification' else 'analysis'
            if manifest['model_identities'].get(key,identity)!=identity:raise ValueError('analysis model changed')
            manifest['model_identities'][key]=identity;write_json(out/'manifest.json',manifest)
            caller=ModelCaller(provider=p,prompts=prompts,run_dir=out,protocol_version=VERSION,max_attempts=1)
            payload=analysis_input(rows[task['attempt_id']],task['stage'],task['removed'],config,proof)
            if payload['image_path']:payload['image_path']=str(root/payload['image_path'])
            call=call_model(caller,operation_id=task['operation_id'],attempt_id=task['attempt_id'],
                role=task['stage'],task='assess_quality',**payload)
            result={**task,'status':call['status'],'output':call.get('parsed_output'),'call_id':call['call_id']}
            if task['stage']=='verification':result['gate']=verification_gate(call.get('parsed_output'),strict=True) if call['status']=='ok' else 'unresolved'
            atomic_upsert(out/'records.jsonl','operation_id',result);done[task['operation_id']]=result;steps+=1
            print(json.dumps({'item':task['item_id'],'stage':task['stage'],'removed':task['removed'],'status':call['status']}),flush=True)
    usage=call_usage(read_jsonl(out/'calls.jsonl'));write_json(out/'usage.json',usage)
    return {'new_steps':steps,'completed_slots':len(done),'planned_slots':len(plan),'usage':usage}
