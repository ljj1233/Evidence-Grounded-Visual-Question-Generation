"""Open completion witnesses with isolated inputs and explicit AI evidence review."""
from __future__ import annotations

import json
from pathlib import Path

from .assessment_rules import check_frozen_sources, choice_conflicts, load_assessment_config
from .experiments import call_usage
from .pipeline import _validate_run_id
from .prompts import PromptSet
from .providers import JudgeProvider, ModelCaller, call_model
from .records import (atomic_upsert, canonical_json, file_lock, read_jsonl,
                      sha256_file, sha256_text, stable_id, utc_now, write_json, write_jsonl)

TEMPLATE = 'open-necessity-v1'


def init_open_probe(*, run_id, cases_path, key_path, project_root='.', output_root='outputs', config_path=None):
    root=Path(project_root).resolve();_validate_run_id(run_id);out=root/output_root/run_id
    cases=read_jsonl(root/cases_path);key=read_jsonl(root/key_path)
    ids=[c['case_id'] for c in cases]
    if not ids or len(ids)!=len(set(ids)) or set(ids)!={k['case_id'] for k in key} or len(ids)!=len(key):
        raise ValueError('unique cases and complete review key required')
    config=load_assessment_config(root,config_path)
    template=config['analysis_template'] if config else TEMPLATE
    prompts=PromptSet(root/'prompts',names=(template,))
    sources={str(root/p):sha256_file(root/p) for p in (cases_path,key_path)}
    sources[str(Path(__file__).resolve())]=sha256_file(Path(__file__))
    if config:
        for path in (root/config_path,Path(__file__).with_name('assessment_rules.py')):
            sources[str(path.resolve())]=sha256_file(path)
    for c in cases:
        if not c['question'].strip() or sha256_file(root/c['image_path'])!=c['image_sha256']:
            raise ValueError('invalid Q or image hash')
        sources[str(root/c['image_path'])]=c['image_sha256']
    fingerprint=sha256_text(canonical_json({'cases':cases,'key':key,'prompts':prompts.snapshot(),'sources':sources}))
    out.mkdir(parents=True,exist_ok=True)
    with file_lock(out/'.init.lock'):
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text())['fingerprint']!=fingerprint:
                raise ValueError('frozen open probe changed; use a new run')
            return {'n_planned':len(ids),'resumed':True}
        plan=[{'case_id':c['case_id'],'attempt_id':stable_id('attempt',run_id,c['case_id'])} for c in cases]
        write_jsonl(out/'cases.jsonl',cases);write_jsonl(out/'review_key.jsonl',key)
        write_jsonl(out/'plan.jsonl',plan)
        write_jsonl(out/'records.jsonl',[{**p,'analysis':{'T':None,'I':None}} for p in plan])
        (out/'prompts').mkdir(exist_ok=True);(out/'prompts'/f'{template}.txt').write_text(prompts.snapshot()[template])
        write_json(out/'manifest.json',{'fingerprint':fingerprint,'created_at':utc_now(),
            'protocol_version':config['protocol_version'] if config else 'open-answer-identifiability-dev-v1','partition':'dev',
            'n_planned':len(ids),'max_http_requests':2*len(ids)+4,'max_output_tokens':config['analysis_max_tokens'] if config else 1200,
            **({'assessment':config,'template':template} if config else {}),
            'sources':sources,'model_identity':None,'prompt_sha256':prompts.snapshot_sha256(),
            'frozen_files':{f:sha256_file(out/f) for f in ('cases.jsonl','review_key.jsonl','plan.jsonl')},
            'provenance':'Codex-authored controls and user-delegated AI review; not blind or natural generation'})
    return {'n_planned':len(ids),'resumed':False}


def _load(out):
    manifest=json.loads((out/'manifest.json').read_text())
    rows=read_jsonl(out/'records.jsonl')
    for name,h in manifest['frozen_files'].items():
        if sha256_file(out/name)!=h:raise ValueError('frozen file changed: '+name)
    check_frozen_sources(manifest['sources'],out,completed=all(r['analysis'][m] is not None for r in rows for m in ('T','I')))
    prompts=PromptSet(out/'prompts',names=(manifest.get('template',TEMPLATE),))
    if prompts.snapshot_sha256()!=manifest['prompt_sha256']:raise ValueError('prompt changed')
    rows=read_jsonl(out/'records.jsonl');plan=read_jsonl(out/'plan.jsonl')
    if (len(rows)!=len(plan) or len({r['attempt_id'] for r in rows})!=len(plan)
            or {(r['attempt_id'],r['case_id']) for r in rows}!={(p['attempt_id'],p['case_id']) for p in plan}):
        raise ValueError('planned slots changed')
    return manifest,prompts,rows,{c['case_id']:c for c in read_jsonl(out/'cases.jsonl')}


def run_open_probe(*,run_id,max_requests=20,dry_run=False,project_root='.',output_root='outputs',provider=None):
    root=Path(project_root).resolve();_validate_run_id(run_id);out=root/output_root/run_id
    manifest,prompts,rows,cases=_load(out);budget=min(max_requests,manifest['max_http_requests'])
    pending=sum(r['analysis'][m] is None for r in rows for m in ('T','I'))
    if dry_run:return {'n_planned':len(rows),'pending':pending,'hard_http_cap':budget}
    if not pending:return {'n_planned':len(rows),'new_steps':0}
    provider=provider or JudgeProvider.from_env(root=root,run_dir=out,max_requests=budget,
                max_tokens=manifest['max_output_tokens'],visual=True)
    if isinstance(provider,JudgeProvider):provider.max_attempts=1
    identity=getattr(provider,'identity',{'model':provider.model})
    if manifest['model_identity'] not in (None,identity):raise ValueError('model changed')
    manifest['model_identity']=identity;write_json(out/'manifest.json',manifest)
    caller=ModelCaller(provider=provider,prompts=prompts,run_dir=out,
        protocol_version=manifest['protocol_version'],max_attempts=2)
    steps=0
    with file_lock(out/'.pipeline.lock'):
        for r in read_jsonl(out/'records.jsonl'):
            c=cases[r['case_id']]
            for removed,retained in (('T','I'),('I','T')):
                if r['analysis'][removed] is not None:continue
                text=c['text'] if retained=='T' else None
                call=call_model(caller,operation_id=stable_id('open',r['attempt_id'],removed),
                    attempt_id=r['attempt_id'],role='open_without_'+removed,template=manifest.get('template',TEMPLATE),
                    variables={'Q':c['question'],'REMOVED':'文本' if removed=='T' else '图像',
                        'VISIBLE':'<T>'+text+'</T>' if text else '仅附上的原图，没有文本材料。',
                        'KNOWLEDGE':manifest.get('assessment',{}).get('knowledge','')},
                    task='assess_quality',question=c['question'],condition=retained,text=text,
                    image_path=str(root/c['image_path']) if retained=='I' else None)
                r['analysis'][removed]={'call_id':call['call_id'],'status':call['status'],
                    'output':call.get('parsed_output'),'error':call.get('error')}
                atomic_upsert(out/'records.jsonl','attempt_id',r,allow_insert=False);steps+=1
                print(json.dumps({'case_id':r['case_id'],'removed':removed,'status':call['status']}),flush=True)
    report_open_probe(run_id=run_id,project_root=root,output_root=output_root)
    return {'n_planned':len(rows),'new_steps':steps}


def report_open_probe(*,run_id,project_root='.',output_root='outputs',verification_run=None):
    root=Path(project_root).resolve();_validate_run_id(run_id);out=root/output_root/run_id
    manifest,_,rows,cases=_load(out);reviews=read_jsonl(out/'evidence_reviews.jsonl');results=[]
    key={k['case_id']:k for k in read_jsonl(out/'review_key.jsonl')}
    seen=set()
    for r in reviews:
        ident=(r['attempt_id'],r['removed'])
        if ident in seen:raise ValueError('duplicate evidence review')
        seen.add(ident)
        row=next((x for x in rows if x['attempt_id']==r['attempt_id']),None)
        analysis=row['analysis'].get(r['removed']) if row else None
        if (not analysis or r['case_id']!=row['case_id'] or r['analysis_sha256']!=sha256_text(canonical_json(analysis))
                or r['source']!='user_delegated_ai' or not r['reviewer'] or not r['reason']
                or r['result'] not in ('supported','not_supported','uncertain')):
            raise ValueError('review binding or provenance invalid')
    for row in rows:
        values={}
        for removed in ('T','I'):
            a=row['analysis'][removed];output=a.get('output') if a else None
            kind=output.get('kind') if isinstance(output,dict) else None
            shape=(isinstance(output,dict) and isinstance(output.get('reason'),str) and bool(output['reason'].strip()))
            if kind=='ambiguous':
                shape=shape and all(isinstance(output.get(k),str) and output[k].strip() for k in ('scenario_a','scenario_b','answer_a','answer_b'))
            elif kind=='unique':shape=shape and isinstance(output.get('answer'),str) and bool(output['answer'].strip())
            elif kind!='unresolved':shape=False
            proposed=(1 if kind=='ambiguous' else 0 if kind=='unique' else None) if shape and a['status']=='ok' else None
            conflicts=choice_conflicts(output,cases[row['case_id']].get('choices')) if manifest.get('assessment') else []
            if conflicts:proposed=None
            review=next((r for r in reviews if r['attempt_id']==row['attempt_id'] and r['removed']==removed),None)
            values[removed]={'kind':kind,'proposed':proposed,
                'confirmed':proposed if review and review['result']=='supported' else None,
                'review_result':review['result'] if review else None,
                **({'choice_conflicts':conflicts} if manifest.get('assessment') else {})}
        results.append({'case_id':row['case_id'],'attempt_id':row['attempt_id'],'analysis':values,
                        'expected':key[row['case_id']]['expected']})
    counts={}
    for m in ('T','I'):
        vals=[r['analysis'][m]['confirmed'] for r in results];n=len(rows);p=vals.count(1);u=vals.count(None)
        counts[m]={'n_planned':n,'pass':p,'fail':vals.count(0),'unresolved':u,'lower':p/n,'upper':(p+u)/n}
    write_json(out/'results.json',{'n_planned':len(rows),'cases':results,'confirmed_necessity':counts})
    if verification_run:
        from .evidence_verification import verification_gate
        _validate_run_id(verification_run);side=root/output_root/verification_run
        vm=json.loads((side/'manifest.json').read_text());vr=read_jsonl(side/'records.jsonl')
        if vm['source_run']!=run_id or vm.get('assessment')!=manifest.get('assessment'):
            raise ValueError('verification source/config mismatch')
        check_frozen_sources(vm['sources'],side,completed=all(r['verification'] is not None for r in vr))
        if sha256_file(side/'items.jsonl')!=vm['items_sha256']:raise ValueError('verification items changed')
        vp=PromptSet(side/'prompts',names=(vm.get('template','verify-necessity-evidence-v1'),))
        if vp.snapshot_sha256()!=vm['prompt_sha256']:raise ValueError('verification prompt changed')
        items={i['item_id']:i for i in read_jsonl(side/'items.jsonl')}
        if len(items)!=len(vr) or set(items)!={r['item_id'] for r in vr}:raise ValueError('verification slots changed')
        gates={}
        by_case={r['case_id']:r for r in rows}
        for v in vr:
            item=items[v['item_id']];ident=(item['case_id'],item['removed'])
            if ident in gates:raise ValueError('duplicate source verification')
            a=by_case[item['case_id']]['analysis'][item['removed']]
            if item['source_analysis_sha256']!=sha256_text(canonical_json(a)) or item['candidate']!=a.get('output'):
                raise ValueError('verification proof binding changed')
            value=v['verification']
            gate=verification_gate(value['output'],strict=bool(manifest.get('assessment'))) if value and value['status']=='ok' else 'unresolved'
            choices=cases[item['case_id']].get('choices')
            if manifest.get('assessment') and (choice_conflicts(item['candidate'],choices) or choice_conflicts(value.get('output') if value else None,choices)):
                gate='reject'
            gates[ident]=gate
        validity_reviews={}
        for review in read_jsonl(out/'validity_reviews.jsonl'):
            cid=review['case_id']
            if (cid in validity_reviews or cid not in cases or review.get('validity') not in (0,1,None)
                    or review.get('case_sha256')!=sha256_text(canonical_json(cases[cid]))
                    or review.get('knowledge_sha256')!=sha256_text(manifest.get('assessment',{}).get('knowledge',''))
                    or review.get('source')!='user_delegated_ai' or not review.get('reviewer') or not review.get('reason')):
                raise ValueError('validity review binding or provenance invalid')
            validity_reviews[cid]=review
        final=[]
        for result in results:
            directions={}
            for m in ('T','I'):
                d=result['analysis'][m];gate=gates.get((result['case_id'],m),'unresolved')
                directions[m]={**d,'verify_gate':gate,'confirmed':d['confirmed'] if gate=='accept' else None}
            reviewed_key=key[result['case_id']]
            validity=reviewed_key.get('validity')
            if (validity is None and reviewed_key.get('reference_status')=='verified_by_delegated_ai'
                    and reviewed_key.get('review_source')=='user_delegated_ai'):
                validity=1
            if result['case_id'] in validity_reviews:validity=validity_reviews[result['case_id']]['validity']
            if validity not in (0,1,None):raise ValueError('invalid reviewed validity')
            values=[validity]+[directions[m]['confirmed'] for m in ('T','I')]
            both=0 if 0 in values else 1 if values==[1,1,1] else None
            final.append({'case_id':result['case_id'],'attempt_id':result['attempt_id'],
                          'validity':validity,'directions':directions,'both_required':both})
        vals=[r['both_required'] for r in final];n=len(rows);p=vals.count(1);u=vals.count(None)
        verified_counts={}
        for m in ('T','I'):
            values=[r['directions'][m]['confirmed'] for r in final];mp=values.count(1);mu=values.count(None)
            verified_counts[m]={'n_planned':n,'pass':mp,'fail':values.count(0),'unresolved':mu,'lower':mp/n,'upper':(mp+mu)/n}
        verified={'n_planned':n,'cases':final,'both_required':{'pass':p,'fail':vals.count(0),'unresolved':u,'lower':p/n,'upper':(p+u)/n},
                  'confirmed_necessity':verified_counts,
                  'provenance':'user_delegated_ai + same-backbone verifier; not independent human gold',
                  'verification_run':verification_run}
        write_json(out/'verified_results.json',verified)
    usage=call_usage(read_jsonl(out/'calls.jsonl'));write_json(out/'usage.json',usage)
    return {'n_planned':len(rows),'confirmed_necessity':counts,'usage':usage,
            **({'verified_both_required':verified['both_required'],'verified_necessity':verified_counts} if verification_run else {})}
