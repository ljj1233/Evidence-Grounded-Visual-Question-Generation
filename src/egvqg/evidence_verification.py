"""A frozen sidecar verifier; it never changes source necessity labels or proofs."""
from __future__ import annotations

import json
from pathlib import Path

from .assessment_rules import check_frozen_sources, choice_conflicts, load_assessment_config
from .experiments import call_usage
from .pipeline import _validate_run_id
from .prompts import PromptSet
from .providers import JudgeProvider, ModelCaller, call_model
from .records import (atomic_upsert, canonical_json, file_lock, read_jsonl, sha256_file,
                      sha256_text, stable_id, utc_now, write_json, write_jsonl)

TEMPLATE='verify-necessity-evidence-v1'
CHECKS=('question_preserved','retained_preserved','well_posed','premises_supported','conclusion_checked')


def verification_gate(output, *, strict=False):
    if not isinstance(output,dict) or any(output.get(k) not in ('pass','fail','uncertain') for k in CHECKS):
        return 'unresolved'
    if not isinstance(output.get('reason'),str) or not output['reason'].strip():return 'unresolved'
    if strict and any(output[k]=='fail' for k in CHECKS):return 'reject'
    if not isinstance(output.get('recomputed_answers'),list) or not output['recomputed_answers']:return 'unresolved'
    if any(output[k]=='fail' for k in CHECKS):return 'reject'
    return 'accept' if all(output[k]=='pass' for k in CHECKS) else 'unresolved'


def verify_open(*,run_id,source_run,items_path,max_requests=16,dry_run=False,report_only=False,
                project_root='.',output_root='outputs',provider=None,config_path=None):
    root=Path(project_root).resolve();_validate_run_id(run_id);_validate_run_id(source_run)
    source=root/output_root/source_run;out=root/output_root/run_id
    if source.resolve()==out.resolve():raise ValueError('verification must use a separate run')
    items=read_jsonl(root/items_path);cases={c['case_id']:c for c in read_jsonl(source/'cases.jsonl')}
    rows={r['case_id']:r for r in read_jsonl(source/'records.jsonl')}
    if not items or len({i['item_id'] for i in items})!=len(items):raise ValueError('unique fixed verification items required')
    config=load_assessment_config(root,config_path)
    existing=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else None
    if existing and config_path is None:config=existing.get('assessment')
    if existing and config!=existing.get('assessment'):raise ValueError('frozen assessment config changed')
    source_config=json.loads((source/'manifest.json').read_text()).get('assessment')
    if source_config and config!=source_config:raise ValueError('analysis and verifier must share the same knowledge/config')
    template=config['verification_template'] if config else TEMPLATE
    prompts=PromptSet(root/'prompts',names=(template,));out.mkdir(parents=True,exist_ok=True)
    sources={str(p):sha256_file(p) for p in [root/items_path,Path(__file__).resolve(),
              source/'cases.jsonl',source/'records.jsonl',source/'manifest.json']}
    if config:
        shared=Path(__file__).with_name('assessment_rules.py')
        sources[str(shared.resolve())]=sha256_file(shared)
        if config_path:sources[str((root/config_path).resolve())]=sha256_file(root/config_path)
        elif existing:
            for path,h in existing['sources'].items():
                if Path(path).suffix=='.json' and path not in sources:sources[path]=h
    for item in items:
        row=rows[item['case_id']];a=row['analysis'][item['removed']]
        if item['source_analysis_sha256']!=sha256_text(canonical_json(a)):raise ValueError('source proof binding changed')
        c=cases[item['case_id']];image=root/c['image_path']
        if sha256_file(image)!=c['image_sha256']:raise ValueError('source image changed')
        sources[str(image)]=c['image_sha256']
    if existing:
        completed=all(r['verification'] is not None for r in read_jsonl(out/'records.jsonl'))
        check_frozen_sources(existing['sources'],out,completed=completed)
        for path,h in existing['sources'].items():
            if sources.get(path)!=h and Path(path).suffix=='.py' and completed:sources[path]=h
    fingerprint=sha256_text(canonical_json({'sources':sources,'items':items,'prompts':prompts.snapshot()}))
    with file_lock(out/'.init.lock'):
        if (out/'manifest.json').exists():
            manifest=json.loads((out/'manifest.json').read_text())
            if manifest['fingerprint']!=fingerprint:raise ValueError('frozen verification changed; use a new run')
            if sha256_file(out/'items.jsonl')!=manifest['items_sha256']:raise ValueError('frozen items changed')
        else:
            write_jsonl(out/'items.jsonl',items)
            write_jsonl(out/'records.jsonl',[{'item_id':i['item_id'],'attempt_id':stable_id('verify',run_id,i['item_id']),'verification':None} for i in items])
            (out/'prompts').mkdir(exist_ok=True);(out/'prompts'/f'{template}.txt').write_text(prompts.snapshot()[template])
            manifest={'fingerprint':fingerprint,'created_at':utc_now(),'source_run':source_run,'n_planned':len(items),
                'sources':sources,'items_sha256':sha256_file(out/'items.jsonl'),'prompt_sha256':prompts.snapshot_sha256(),
                'model_identity':None,'max_http_requests':len(items)+4,'max_output_tokens':config['verification_max_tokens'] if config else 1400,
                **({'assessment':config,'template':template} if config else {}),
                'protocol_version':config['protocol_version']+'-verify' if config else 'necessity-proof-verification-dev-v1'}
            write_json(out/'manifest.json',manifest)
    snapshot=PromptSet(out/'prompts',names=(template,))
    if snapshot.snapshot_sha256()!=manifest['prompt_sha256']:raise ValueError('verification prompt snapshot changed')
    records=read_jsonl(out/'records.jsonl')
    if len(records)!=len(items) or {r['item_id'] for r in records}!={i['item_id'] for i in items}:raise ValueError('fixed verification slots changed')
    pending=sum(r['verification'] is None for r in records);budget=min(max_requests,manifest['max_http_requests'])
    if dry_run:return {'n_planned':len(items),'pending':pending,'hard_http_cap':budget}
    steps=0
    if pending and not report_only:
        provider=provider or JudgeProvider.from_env(root=root,run_dir=out,max_requests=budget,
                    max_tokens=manifest['max_output_tokens'],visual=True)
        if isinstance(provider,JudgeProvider):provider.max_attempts=1
        identity=getattr(provider,'identity',{'model':provider.model})
        if manifest['model_identity'] not in (None,identity):raise ValueError('verification model changed')
        manifest['model_identity']=identity;write_json(out/'manifest.json',manifest)
        caller=ModelCaller(provider=provider,prompts=snapshot,run_dir=out,
            protocol_version=manifest['protocol_version'],max_attempts=2)
        by_id={i['item_id']:i for i in items}
        with file_lock(out/'.pipeline.lock'):
            for r in read_jsonl(out/'records.jsonl'):
                if r['verification'] is not None:continue
                item=by_id[r['item_id']];c=cases[item['case_id']];retained='I' if item['removed']=='T' else 'T'
                proof=item['candidate']
                evidence=({k:proof.get(k) for k in ('scenario_a','answer_a','scenario_b','answer_b')}
                          if proof.get('kind')=='ambiguous' else {k:proof.get(k) for k in ('answer','reason')})
                if config:
                    # This is an input schema, not a necessity verdict. Without it
                    # some verifiers wrongly require two scenarios for a single
                    # sufficient solution. Never expose source decisions/references.
                    evidence={'proof_form':('two_hypothetical_completions' if proof.get('kind')=='ambiguous'
                                            else 'single_sufficient_solution_no_scenarios_required'),
                              **evidence}
                text=c['text'] if retained=='T' else None
                call=call_model(caller,operation_id=stable_id('verify_operation',run_id,item['item_id']),
                    attempt_id=r['attempt_id'],role='verify_proof',template=template,
                    variables={'Q':c['question'],'REMOVED':'文本' if item['removed']=='T' else '图像',
                        'VISIBLE':'<T>'+text+'</T>' if text else '仅附上的原图，没有原始文本。',
                        'EVIDENCE':canonical_json(evidence),'KNOWLEDGE':config['knowledge'] if config else ''},task='assess_quality',question=c['question'],
                    condition=retained,text=text,image_path=str(root/c['image_path']) if retained=='I' else None)
                r['verification']={'call_id':call['call_id'],'status':call['status'],'output':call.get('parsed_output')}
                atomic_upsert(out/'records.jsonl','item_id',r,allow_insert=False);steps+=1
                print(json.dumps({'item_id':item['item_id'],'status':call['status']}),flush=True)
    results=[]
    for r in read_jsonl(out/'records.jsonl'):
        v=r['verification'];item=next(i for i in items if i['item_id']==r['item_id'])
        gate=verification_gate(v['output'],strict=bool(config)) if v and v['status']=='ok' else 'unresolved'
        conflicts=(choice_conflicts(item['candidate'],cases[item['case_id']].get('choices'))
                   +choice_conflicts(v.get('output') if v else None,cases[item['case_id']].get('choices'))) if config else []
        if conflicts:gate='reject'
        results.append({'item_id':r['item_id'],'case_id':item['case_id'],'removed':item['removed'],
                        'gate':gate,'expected_review':item.get('expected_review'),
                        **({'choice_conflicts':conflicts} if config else {})})
    write_json(out/'results.json',{'n_planned':len(items),'cases':results,
        'warning':'Model verification proposals; no source labels changed. Same backbone, not independent blind gold.'})
    usage=call_usage(read_jsonl(out/'calls.jsonl'));write_json(out/'usage.json',usage)
    return {'n_planned':len(items),'new_steps':steps,'counts':{k:sum(r['gate']==k for r in results) for k in ('accept','reject','unresolved')},'usage':usage}
