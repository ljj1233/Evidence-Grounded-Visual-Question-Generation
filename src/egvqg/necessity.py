"""Evaluation-only evidence probe: fixed slots, bounded calls and reviewed evidence.

Expected labels are held outside model records and are never rendered into prompts.
References must already have a named review. No generation or BDR runs here.
"""
from __future__ import annotations

import json
from pathlib import Path

from .experiments import call_usage
from .metrics import decide_necessity
from .pipeline import _validate_run_id, load_materials
from .prompts import PromptSet
from .providers import JudgeProvider, ModelCaller, call_model
from .records import (atomic_upsert, canonical_json, file_lock, read_jsonl, review_source,
                      sha256_file, sha256_text, stable_id, utc_now, write_json, write_jsonl)


def _config(path):
    """One optional base config, with every source pinned for resume."""
    config=json.loads(path.read_text());sources={str(path):sha256_file(path)}
    base=config.pop('extends',None)
    if base:
        source=path.parent/base;defaults=json.loads(source.read_text())
        if 'extends' in defaults:raise ValueError('only one base evaluation config is supported')
        config={**defaults,**config};sources[str(source)]=sha256_file(source)
    return config,sources


def init_necessity(*, run_id, materials_path, cases_path, expected_path=None,
                   config_path='configs/evaluation_v2.json', project_root='.', output_root='outputs'):
    root = Path(project_root).resolve(); _validate_run_id(run_id); out = root/output_root/run_id
    config,config_sources = _config(root/config_path)
    prompts = PromptSet(root/'prompts', names=(config['analysis_template'], config['solve_template']))
    materials = load_materials(root, root/materials_path); by_id = {m['material_id']:m for m in materials}
    cases = read_jsonl(root/cases_path)
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('cases must be unique and nonempty')
    plan, rows = [], []
    for c in cases:
        ref = c['reference']; review_source(ref['review'])
        if not ref['review'].get('reviewer') or c['material_id'] not in by_id or not c['question'].strip():
            raise ValueError('review identity, source material and Q required')
        ids = [x['id'] for x in ref['components']]
        if not ids or len(set(ids)) != len(ids) or ref['validity'] not in (None, 0, 1):
            raise ValueError('reference components and validity required')
        if ref['validity'] == 1 and (ref['status'] != 'verified' or any(x['value'] is None for x in ref['components'])):
            raise ValueError('valid reference must be verified with complete values')
        aid = stable_id('attempt',run_id,c['case_id'])
        slot = {'attempt_id':aid,'case_id':c['case_id'],'material_id':c['material_id']}
        plan.append(slot)
        injected=c.get('injected_isolated',{})
        if injected and (not config.get('allow_synthetic_isolated') or any(
                m not in ('T','I') or v.get('source')!='synthetic' for m,v in injected.items())):
            raise ValueError('synthetic solver records require explicit dev configuration')
        rows.append({**slot, 'run_id':run_id, 'protocol_version':config['protocol_version'],
            'knowledge_version':config['knowledge_version'], 'question':c['question'],
            'question_sha256':sha256_text(c['question']), 'reference':ref,
            'analysis':{'T':[],'I':[]},'isolated':injected,'reviews':[]})
    expected = read_jsonl(root/expected_path) if expected_path else []
    if expected_path and ({e['case_id'] for e in expected} != {c['case_id'] for c in cases} or len(expected)!=len(cases)):
        raise ValueError('expected labels must cover all fixed slots')
    identity = {'config':config,'cases':cases,'materials':materials,'plan':plan,
                'prompts':prompts.snapshot(),'expected_sha256':sha256_text(canonical_json(expected))}
    fingerprint = sha256_text(canonical_json(identity))
    out.mkdir(parents=True,exist_ok=True)
    with file_lock(out/'.init.lock'):
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text())['fingerprint'] != fingerprint:
                raise ValueError('frozen evaluation changed; use a new run_id')
            _load(out,root); return {'n_planned':len(plan),'resumed':True}
        write_jsonl(out/'plan.jsonl',plan); write_jsonl(out/'records.jsonl',rows)
        write_jsonl(out/'materials.jsonl',materials);write_jsonl(out/'cases.jsonl',cases)
        if expected_path: write_jsonl(out/'expected.jsonl',expected)
        for name,text in prompts.snapshot().items():
            (out/'prompts').mkdir(exist_ok=True);(out/'prompts'/f'{name}.txt').write_text(text)
        frozen = {name:sha256_file(out/name) for name in ('plan.jsonl','materials.jsonl','cases.jsonl')}
        if expected_path:frozen['expected.jsonl']=sha256_file(out/'expected.jsonl')
        write_json(out/'manifest.json',{'fingerprint':fingerprint,'config':config,'n_planned':len(plan),
            'model_identity':None,'created_at':utc_now(),'frozen_files':frozen,
            'prompt_sha256':prompts.snapshot_sha256(),'config_path':str(root/config_path),
            'config_sha256':sha256_file(root/config_path),'config_sources':config_sources})
    return {'n_planned':len(plan),'resumed':False}


def _load(out,root):
    manifest=json.loads((out/'manifest.json').read_text());cfg=manifest['config']
    for name,digest in manifest['frozen_files'].items():
        if sha256_file(out/name)!=digest:raise ValueError('frozen evaluation input changed: '+name)
    if sha256_file(Path(manifest['config_path']))!=manifest['config_sha256']:
        raise ValueError('evaluation configuration changed')
    for path,digest in {**manifest.get('config_sources',{}),**manifest.get('derived_source_hashes',{})}.items():
        if sha256_file(Path(path))!=digest:raise ValueError('frozen evaluation source changed')
    prompts=PromptSet(out/'prompts',names=(cfg['analysis_template'],cfg['solve_template']))
    if prompts.snapshot_sha256()!=manifest['prompt_sha256']:raise ValueError('prompt snapshot changed')
    rows=read_jsonl(out/'records.jsonl');plan=read_jsonl(out/'plan.jsonl');cases=read_jsonl(out/'cases.jsonl')
    if len(rows)!=len(plan) or {r['attempt_id'] for r in rows}!={s['attempt_id'] for s in plan}:
        raise ValueError('records must contain exactly the preplanned slots')
    by_case={c['case_id']:c for c in cases};by_slot={s['attempt_id']:s for s in plan}
    for r in rows:
        c=by_case[r['case_id']]
        if (any(r[k]!=by_slot[r['attempt_id']][k] for k in ('case_id','material_id'))
                or r['question']!=c['question'] or r['question_sha256']!=sha256_text(c['question'])
                or r['reference']!=c['reference'] or r['knowledge_version']!=cfg['knowledge_version']):
            raise ValueError('Q, reference or knowledge binding changed')
    mats={m['material_id']:m for m in load_materials(root,out/'materials.jsonl')}
    return manifest,prompts,rows,mats


def _analysis(caller, r, material, root, cfg, removed, round_index):
    isolated=r['isolated'].get(removed)
    if isolated is not None and cfg.get('evidence_contract')=='v3':isolated=isolated.get('output')
    variables={'REMOVED':removed,'RETAINED':'I' if removed=='T' else 'T',
        'T':material['text'],'Q':r['question'], 'KNOWLEDGE':canonical_json(cfg['knowledge']),
        'COMPLETION_SPACE':cfg['completion_space'],
        'REFERENCE':canonical_json(r['reference']),
        'ISOLATED_RECORD':'' if isolated is None else '隔离作答仅供核对前提，不是必要性结论：'+canonical_json(isolated)}
    call=call_model(caller,operation_id=stable_id('analysis',r['attempt_id'],removed,round_index),
        attempt_id=r['attempt_id'],role=f'analysis_{removed}_{round_index}',template=cfg['analysis_template'],
        variables=variables,task='assess_quality',text=material['text'],image_path=str(root/material['image_path']),
        question=r['question'])
    result={'round':round_index,'call_id':call['call_id'],'status':call['status'],
            'output':call.get('parsed_output'),'error':call.get('error')}
    if cfg.get('evidence_contract')=='v3' and call.get('parsed_output') is not None:
        from .evidence_contract import decode_v3
        result['raw_contract']=call['parsed_output']
        result['output'],result['contract_binding']=decode_v3(call['parsed_output'],removed,r['reference'],material,cfg['knowledge'])
    return result


def normalize_necessity(*,run_id,source_run,project_root='.',output_root='outputs'):
    """Zero-call derived evaluation of baseline rounds, never latest-round selection."""
    from .evidence_contract import normalize_v1
    root=Path(project_root).resolve();source=root/output_root/source_run;out=root/output_root/run_id
    manifest,_,originals,_=_load(source,root)
    if source.resolve()==out.resolve():raise ValueError('normalization requires a separate run')
    if (out/'manifest.json').exists():
        loaded=json.loads((out/'manifest.json').read_text())
        if loaded.get('normalization_policy')!='norm-v1' or loaded.get('source_run')!=source_run:
            raise ValueError('derived run identity changed')
        _load(out,root);return {'n_planned':len(originals),'new_calls':0,'resumed':True}
    init_necessity(run_id=run_id,materials_path=str(source/'materials.jsonl'),cases_path=str(source/'cases.jsonl'),
        expected_path=str(source/'expected.jsonl') if (source/'expected.jsonl').exists() else None,
        config_path=manifest['config_path'],project_root=root,output_root=output_root)
    rows=read_jsonl(out/'records.jsonl');by_case={r['case_id']:r for r in originals};logs=[]
    for r in rows:
        orig=by_case[r['case_id']];r.update(source_run_id=source_run,source_attempt_id=orig['attempt_id'],evaluation_variant='norm-v1')
        for removed,rounds in orig['analysis'].items():
            if not rounds:continue
            first=rounds[0];raw=first.get('output') or {}
            normalized,changes=normalize_v1(raw,removed,manifest['config']['knowledge'],[c['id'] for c in r['reference']['components']])
            r['analysis'][removed]=[{'round':0,'status':'derived','output':normalized,'original_output':raw,
                'source_call_id':first['call_id'],'source_output_sha256':sha256_text(canonical_json(raw))}]
            logs.append({'case_id':r['case_id'],'removed':removed,'source_call_id':first['call_id'],
                         'raw_sha256':sha256_text(canonical_json(raw)),'normalized_sha256':sha256_text(canonical_json(normalized)),
                         'changes':changes})
    write_jsonl(out/'records.jsonl',rows);write_jsonl(out/'normalization_log.jsonl',logs)
    current=json.loads((out/'manifest.json').read_text());current.update(offline_derived=True,
        normalization_policy='norm-v1',source_run=source_run,derived_source_hashes={str(source/f):sha256_file(source/f)
        for f in ('records.jsonl','calls.jsonl','manifest.json','cases.jsonl','expected.jsonl') if (source/f).exists()})
    write_json(out/'manifest.json',current);report_necessity(run_id=run_id,project_root=root,output_root=output_root)
    return {'n_planned':len(rows),'new_calls':0,'resumed':False}


def run_necessity(*, run_id, follow_up=False, max_requests=24, dry_run=False,
                  project_root='.',output_root='outputs',provider=None):
    root=Path(project_root).resolve();_validate_run_id(run_id);out=root/output_root/run_id
    manifest,prompts,rows,mats=_load(out,root);cfg=manifest['config']
    if manifest.get('offline_derived'):raise ValueError('derived normalization never sends model requests')
    budget=min(max_requests,cfg['max_http_requests'])
    if dry_run:
        return {'n_planned':len(rows),'baseline_pending':sum(not r['analysis'][m] for r in rows
                if r['reference']['validity']==1 for m in ('T','I')),'hard_http_cap':budget}
    provider=provider or JudgeProvider.from_env(root=root,run_dir=out,max_requests=budget,visual=True,
                max_tokens=cfg['max_output_tokens'],env_prefix=cfg['env_prefix'])
    if isinstance(provider,JudgeProvider):provider.max_attempts=1
    identity=getattr(provider,'identity',{'model':provider.model})
    if manifest['model_identity'] not in (None,identity):raise ValueError('actual model identity changed')
    if manifest['model_identity'] is None:
        manifest['model_identity']=identity;write_json(out/'manifest.json',manifest)
    caller=ModelCaller(provider=provider,prompts=prompts,run_dir=out,protocol_version=cfg['protocol_version'],
            max_attempts=cfg['max_technical_attempts'],json_repair=cfg['json_repair'])
    steps=0
    with file_lock(out/'.pipeline.lock'):
        for r in read_jsonl(out/'records.jsonl'):
            if r['reference']['validity']!=1:continue
            material=mats[r['material_id']]
            for removed in ('T','I'):
                rounds=r['analysis'][removed]
                if not follow_up and rounds:continue
                if follow_up:
                    if len(rounds)!=1:continue  # at most one targeted follow-up per modality
                    proposed=decide_necessity(r)
                    rejected=any(x['removed']==removed and x['evidence_key'].startswith('component:')
                                 and x['result']!='supported' for x in r['reviews'])
                    unresolved=any(v[removed] is None for v in proposed['components'].values())
                    if cfg.get('followup_policy')=='substantive_only':
                        output=rounds[0].get('output') or {}
                        unresolved=not proposed['flags'] and any(i.get('proposed') is None for i in output.get('components',{}).values())
                        rejected=any(x['removed']==removed and x['evidence_key'].startswith('component:')
                                     and x.get('failure_class')=='semantic' and x['result']!='supported' for x in r['reviews'])
                    if not rejected and not unresolved:continue
                used = sum(x.get('event')=='started' for x in read_jsonl(out/'requests.jsonl')) if isinstance(
                    provider,JudgeProvider) else call_usage(read_jsonl(out/'calls.jsonl'))['remote_requests']
                needed = (1 if removed in r['isolated'] else 2) if follow_up else 1
                if used+needed > budget:
                    report_necessity(run_id=run_id,project_root=root,output_root=output_root)
                    return {'n_planned':len(rows),'new_steps':steps,'status':'budget_exhausted',
                            'http_used':used,'hard_http_cap':budget}
                if follow_up:
                    if removed not in r['isolated']:
                        text=material['text'] if removed=='I' else None
                        call=call_model(caller,operation_id=stable_id('isolated',r['attempt_id'],removed),
                            attempt_id=r['attempt_id'],role='isolated_without_'+removed,template=cfg['solve_template'],
                            variables={'Q':r['question'],'KNOWLEDGE':canonical_json(cfg['knowledge']),
                                'VISIBLE_MATERIALS':'<T>'+text+'</T>' if text else ''},
                            task='answer_question',question=r['question'],condition='T' if text else 'I',
                            text=text,image_path=None if text else str(root/material['image_path']))
                        r['isolated'][removed]={'call_id':call['call_id'],'status':call['status'],
                                               'output':call.get('parsed_output')}
                        atomic_upsert(out/'records.jsonl','attempt_id',r,allow_insert=False);steps+=1
                rounds.append(_analysis(caller,r,material,root,cfg,removed,1 if follow_up else 0))
                atomic_upsert(out/'records.jsonl','attempt_id',r,allow_insert=False);steps+=1
    report_necessity(run_id=run_id,project_root=root,output_root=output_root)
    return {'n_planned':len(rows),'new_steps':steps,'hard_http_cap':budget}


def export_evidence_reviews(out, rows):
    existing=read_jsonl(out/'evidence_review_template.jsonl');seen={r['review_id'] for r in existing}
    for r in rows:
        for removed,rounds in r['analysis'].items():
            if not rounds:continue
            a=rounds[-1];output=a.get('output') or {};digest=sha256_text(canonical_json(output))
            ids={c['id'] for c in r['reference']['components']}
            items=[('component:'+c,ev) for c,ev in output.get('components',{}).items()
                   if c in ids and isinstance(ev,dict)]
            items += [('fact:'+f['id'],f) for f in output.get('decisive_facts',[]) if isinstance(f,dict) and f.get('id')]
            for key,evidence in items:
                rid=stable_id('evidence_review',r['attempt_id'],removed,digest,key)
                if rid in seen:continue
                existing.append({'review_id':rid,'attempt_id':r['attempt_id'],'case_id':r['case_id'],
                    'question_sha256':r['question_sha256'],'knowledge_version':r['knowledge_version'],
                    'removed':removed,'analysis_sha256':digest,'evidence_key':key,'evidence':evidence,
                    'result':None,'reason':None,'source':None,'reviewer':None})
                seen.add(rid)
    write_jsonl(out/'evidence_review_template.jsonl',existing)


def review_necessity(*,run_id,reviews_path,project_root='.',output_root='outputs'):
    root=Path(project_root).resolve();out=root/output_root/run_id
    _,_,rows,_=_load(out,root);by_id={r['attempt_id']:r for r in rows}
    reviews=[r for r in read_jsonl(root/reviews_path) if r.get('result') is not None]
    if len({r['review_id'] for r in reviews})!=len(reviews):raise ValueError('duplicate evidence review')
    for review in reviews:
        review_source(review);r=by_id[review['attempt_id']]
        if (not review.get('reviewer') or not review.get('reason')
                or review['result'] not in ('supported','not_supported','uncertain')
                or review['question_sha256']!=r['question_sha256']
                or review['knowledge_version']!=r['knowledge_version']):raise ValueError('review binding invalid')
        rounds=r['analysis'][review['removed']]
        matched=next((a for a in rounds if sha256_text(canonical_json(a.get('output') or {}))==review['analysis_sha256']),None)
        if matched is None:
            raise ValueError('review belongs to another analysis')
        output=matched.get('output') or {};kind,_,key=review['evidence_key'].partition(':')
        if kind not in ('component','fact'):raise ValueError('unknown evidence kind')
        evidence=output.get('components',{}).get(key) if kind=='component' else next(
            (f for f in output.get('decisive_facts',[]) if f.get('id')==key),None)
        if (evidence is None or review.get('evidence')!=evidence or review['review_id']!=stable_id(
                'evidence_review',r['attempt_id'],review['removed'],review['analysis_sha256'],review['evidence_key'])):
            raise ValueError('review evidence identity changed')
        previous=next((x for x in r['reviews'] if x['review_id']==review['review_id']),None)
        if previous and previous!=review:raise ValueError('review already sealed; append a new evaluation instead')
    changed=0
    with file_lock(out/'.pipeline.lock'):
        for review in reviews:
            r=next(r for r in read_jsonl(out/'records.jsonl') if r['attempt_id']==review['attempt_id'])
            previous=next((x for x in r['reviews'] if x['review_id']==review['review_id']),None)
            if previous:
                if previous!=review:raise ValueError('review already sealed; append a new evaluation instead')
                continue
            r['reviews'].append(review);atomic_upsert(out/'records.jsonl','attempt_id',r,allow_insert=False);changed+=1
    report_necessity(run_id=run_id,project_root=root,output_root=output_root)
    return {'updated_evidence_items':changed}


def report_necessity(*,run_id,project_root='.',output_root='outputs'):
    root=Path(project_root).resolve();out=root/output_root/run_id
    manifest,_,rows,_=_load(out,root);expected={e['case_id']:e for e in read_jsonl(out/'expected.jsonl')}
    results=[]
    for r in rows:
        results.append({'case_id':r['case_id'],'attempt_id':r['attempt_id'],'question':r['question'],
            'proposed':decide_necessity(r),'confirmed':decide_necessity(r,confirmed=True),
            'expected':expected.get(r['case_id'])})
    def summary(source,key):
        values=[r[source][key] for r in results];n=len(values);p=values.count(1)
        na=sum(r[source]['integration_status']=='not_applicable' for r in results) if key=='integration' else 0
        u=values.count(None)-na
        return {'n_planned':n,'n_pass':p,'n_fail':values.count(0),'n_unresolved':u,
                'n_not_applicable':na,'lower':p/n if n else None,'upper':(p+u)/n if n else None}
    metrics={s:{k:summary(s,k) for k in ('validity','text_necessary','image_necessary','integration')}
             for s in ('proposed','confirmed')}
    calls=read_jsonl(out/'calls.jsonl');write_json(out/'results.json',{'n_planned':len(rows),'cases':results,'metrics':metrics})
    write_json(out/'usage.json',call_usage(calls));export_evidence_reviews(out,rows)
    lines=['# Evidence-based necessity probe','','Evidence judgments are conditional on the frozen K/completion space. Proposed and reviewed results are separate; reviewer provenance is retained. No generation, four-condition scoring or BDR.','',
        '|Case|Expected T/I/S|Proposed T/I/S|Reviewed T/I/S|','|---|---|---|---|']
    def triple(value):return '/'.join('?' if value.get(k) is None else str(value[k]) for k in ('text_necessary','image_necessary','integration'))
    for r in results:lines.append(f"|{r['case_id']}|{triple((r['expected'] or {}).get('verdict',{}))}|{triple(r['proposed'])}|{triple(r['confirmed'])}|")
    lines+=['',f"Planned N={len(rows)}; logical calls={len(calls)}; HTTP hard cap={manifest['config']['max_http_requests']}.",
            'Integration is not applicable for confirmed non-bimodal items; this is distinct from a confirmed stitching failure. See results.json for components, unresolved flags and fixed-N bounds.']
    (out/'summary.md').write_text('\n'.join(lines))
    return {'n_planned':len(rows),'calls':len(calls),'report':str(out/'summary.md')}
