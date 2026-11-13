"""Evaluate completed simple baselines; question quality, answer and necessity stay separate."""
from __future__ import annotations

import asyncio
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .assessment_rules import load_assessment_config
from .baseline_analysis import analysis_input,proof_shape
from .evidence_verification import verification_gate
from .experiments import call_usage,label
from .faithfulness import CHECKS,_decision
from .metrics import _three_and
from .pipeline import _validate_run_id
from .prompts import PromptSet
from .providers import JudgeProvider,ModelCaller,call_model
from .records import canonical_json,file_lock,read_jsonl,sha256_file,sha256_text,stable_id,utc_now,write_json,write_jsonl

VERSION='simple-baseline-evaluation-v1'
NAMES=('review-question-faithfulness-v1','open-necessity-v2','verify-necessity-evidence-v2')
SCREEN='ensemble-sufficiency-v1'


def names_for(settings):
    return NAMES+((SCREEN,) if settings.get('cascade') else ())


def init_evaluation(*,run_id,source_run,project_root='.',output_root='outputs',model='gpt-6-luna',concurrency=16,cascade=False):
    root=Path(project_root).resolve();_validate_run_id(run_id);_validate_run_id(source_run)
    out=root/output_root/run_id;source=root/output_root/source_run
    if out==source or not 1<=concurrency<=32:raise ValueError('invalid evaluation run/concurrency')
    status=json.loads((source/'status.json').read_text());rows=read_jsonl(source/'records.jsonl')
    if status['completed']!=status['n_planned'] or len(rows)!=status['n_planned'] or any(r['status']!='generation_answer_complete' for r in rows):
        raise ValueError('generation and TI answering must finish for every slot before evaluation')
    for row in rows:
        q=row['question'];response=row['responses']['TI']
        if q['normalized_text'] and (q['sha256']!=sha256_text(q['normalized_text']) or
                response and (response['question_sha256']!=q['sha256'] or response['question_id']!=q['question_id'])):
            raise ValueError('source answer/question binding changed')
    cfg=load_assessment_config(root,'configs/necessity_assessment_v2.json');prompts=PromptSet(root/'prompts',names=names_for({'cascade':cascade}))
    sources={str((source/n).relative_to(root)):sha256_file(source/n) for n in ('records.jsonl','manifest.json','materials.jsonl','plan.jsonl')}
    code={n:sha256_file(Path(__file__).with_name(n)) for n in ('simple_baseline_evaluation.py','baseline_analysis.py','providers.py','records.py','faithfulness.py','evidence_verification.py')}
    settings={'model':model,'concurrency':concurrency,'assessment':cfg,'per_item_http_cap':5,
        'hard_http_cap':5*len(rows),'technical_attempts':1,'direction_order':['I','T'],'cascade':cascade,
        'short_circuit':'Accepted sufficient solution for either modality rejects both-required; do not infer the unchecked direction.'}
    fingerprint=sha256_text(canonical_json({'sources':sources,'code':code,'settings':settings,'prompts':prompts.snapshot()}))
    out.mkdir(parents=True,exist_ok=True)
    with file_lock(out/'.init.lock'):
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text())['fingerprint']!=fingerprint:raise ValueError('evaluation config changed')
            return {'n_planned':len(rows),'resumed':True}
        write_jsonl(out/'source_questions.jsonl',rows)
        (out/'prompts').mkdir(exist_ok=True);(out/'code_snapshots').mkdir(exist_ok=True)
        for n,t in prompts.snapshot().items():(out/'prompts'/f'{n}.txt').write_text(t)
        for n in code:(out/'code_snapshots'/n).write_bytes(Path(__file__).with_name(n).read_bytes())
        write_json(out/'manifest.json',{'created_at':utc_now(),'protocol_version':VERSION,'fingerprint':fingerprint,
            'source_run':source_run,'n_planned':len(rows),'settings':settings,'source_sha256':sources,'code_sha256':code,
            'source_questions_sha256':sha256_file(out/'source_questions.jsonl'),'prompt_sha256':prompts.snapshot_sha256(),
            'label_scope':'Automatic provisional model evaluation; no human/adjudicated labels. Same API model across roles, correlated errors disclosed.',
            'scope':'Only after whole generation+TI batch completes. Separate question faithfulness, original TI answer support, and both-required proof checks. No four-condition answers or BDR.'})
    return report_evaluation(run_id=run_id,project_root=root,output_root=output_root)


def _empty(source):
    return {'attempt_id':source['attempt_id'],'material_id':source['material_id'],'question':source['question'],
        'status':'pending','question_answerability':None,'TI_answer_support':None,'text_necessary':None,
        'image_necessary':None,'both_required':None,'steps':{},'outputs':{},'errors':[]}


def _evaluate(root,out,m,prompts,source,provider_factory=None):
    item=out/'items'/source['attempt_id'];item.mkdir(parents=True,exist_ok=True);path=item/'record.json'
    row=json.loads(path.read_text()) if path.exists() else _empty(source)
    if row['status']=='complete':return row
    if not source['question']['normalized_text']:
        row.update(status='complete',question_answerability=0,both_required=0,reason='generation failed; original slot retained')
        write_json(path,row);return row
    material=source['material']
    if sha256_file(root/material['image_path'])!=material['image_sha256']:raise ValueError('evaluation image changed')
    cfg=m['settings']['assessment']
    def invoke(step,stage,removed=None,proof=None):
        if step in row['outputs']:return row['outputs'][step]
        payload=analysis_input(source,stage,removed,cfg,proof)
        if stage=='faithfulness':
            response=source['responses']['TI']
            payload['variables']['REFERENCE']=canonical_json({'baseline_TI_answer':response})
        if payload['image_path']:payload['image_path']=str(root/payload['image_path'])
        tokens=cfg['verification_max_tokens'] if stage=='verification' else cfg['analysis_max_tokens']
        p=provider_factory(item,tokens) if provider_factory else JudgeProvider.from_env(root=root,run_dir=item,
            max_requests=m['settings']['per_item_http_cap'],max_tokens=tokens,visual=True,model=m['settings']['model'])
        if isinstance(p,JudgeProvider):p.max_attempts=1
        identity=getattr(p,'identity',{'model':p.model});key='verification' if stage=='verification' else 'analysis'
        with file_lock(out/'.identity.lock'):
            f=out/'model_identities.json';identities=json.loads(f.read_text()) if f.exists() else {}
            if identities.get(key,identity)!=identity:raise ValueError('evaluation model changed')
            identities[key]=identity;write_json(f,identities)
        caller=ModelCaller(provider=p,prompts=prompts,run_dir=item,protocol_version=VERSION,max_attempts=1)
        call=call_model(caller,operation_id=stable_id('evaluate',out.name,source['attempt_id'],step),
            attempt_id=source['attempt_id'],role=step,task='assess_quality',**payload)
        result={'status':call['status'],'output':call.get('parsed_output'),'call_id':call['call_id']}
        row['outputs'][step]=result;row['steps'][step]='done';write_json(path,row);return result
    try:
        f=invoke('faithfulness','faithfulness');d=f.get('output')
        if f['status']=='ok':
            checks=d['question_checks']
            if set(checks)!=set(CHECKS):raise ValueError('invalid question check keys')
            row['question_answerability']=_three_and(*(_decision(checks[k]) for k in CHECKS))
            support=_decision(d['reference_supported'])
            response=source['responses']['TI'] or {}
            row['TI_answer_support']=support if response.get('status')=='answered' else None
        if row['question_answerability']==0:row['both_required']=0
        elif row['question_answerability']==1 and m['settings'].get('cascade'):
            seen={}       # cheap screen: can the retained modality alone give a unique full answer? a failed call never cuts
            for removed,name in (('I','text_only'),('T','image_only')):
                s=invoke('screen_'+name,'screen',removed);res=s.get('output')
                seen[name]=bool(isinstance(res,dict) and res.get('determinable') and res.get('answer')) if s['status']=='ok' else None
            row['screen']=seen
            if any(v is True for v in seen.values()):row['both_required']=0;row['screen_cut']=True
            else:
                for removed in ('I','T'):invoke('analysis_'+removed,'analysis',removed)    # witness proposals only; necessity-ensemble verifies them
        elif row['question_answerability']==1:
            for removed,key in (('I','image_necessary'),('T','text_necessary')):
                a=invoke('analysis_'+removed,'analysis',removed);proof=a.get('output')
                if a['status']!='ok' or not proof_shape(proof):continue
                v=invoke('verification_'+removed,'verification',removed,proof)
                gate=verification_gate(v.get('output'),strict=True) if v['status']=='ok' else 'unresolved'
                if gate=='accept':row[key]=0 if proof['kind']=='unique' else 1
                if row[key]==0:break
            row['both_required']=_three_and(row['question_answerability'],row['text_necessary'],row['image_necessary'])
    except (KeyError,TypeError,ValueError) as exc:row['errors'].append({'error':str(exc),'at':utc_now()})
    row['status']='complete';row['updated_at']=utc_now();write_json(path,row);return row


def report_evaluation(*,run_id,project_root='.',output_root='outputs'):
    root=Path(project_root).resolve();out=root/output_root/run_id
    with file_lock(out/'.report.lock'):
        m=json.loads((out/'manifest.json').read_text());sources=read_jsonl(out/'source_questions.jsonl')
        rows=[];calls=[];requests=[]
        for source in sources:
            folder=out/'items'/source['attempt_id'];p=folder/'record.json'
            rows.append(json.loads(p.read_text()) if p.exists() else _empty(source))
            calls+=read_jsonl(folder/'calls.jsonl');requests+=read_jsonl(folder/'requests.jsonl')
        completed=[r for r in rows if r['status']=='complete']
        def counts(key):return {name:sum(r[key]==v for r in completed) for name,v in [('pass',1),('fail',0),('unresolved',None)]}
        valid=[r for r in completed if r['question_answerability']==1]
        necessity={'both_required':sum(r['both_required']==1 for r in valid),
            'not_both_required':sum(r['both_required']==0 for r in valid),
            'unresolved':sum(r['both_required'] is None for r in valid)}
        usage=call_usage(calls)
        status={'n_planned':len(sources),'completed':len(completed),'pending':len(rows)-len(completed),
            'answerability_completed_only':counts('question_answerability'),
            'original_TI_answer_support_completed_only':counts('TI_answer_support'),
            'necessity_among_answerable_completed_only':necessity,'screen_cut_single_modality_sufficient':sum(bool(r.get('screen_cut')) for r in valid),'target_acceptance_completed_only':counts('both_required'),
            'new_http_requests':sum(r.get('event')=='started' for r in requests),'hard_http_cap':m['settings']['hard_http_cap'],
            'concurrency':m['settings']['concurrency'],'usage':usage,'source':'automatic_provisional'}
        write_jsonl(out/'records.jsonl',rows);write_jsonl(out/'calls.jsonl',calls);write_jsonl(out/'requests.jsonl',requests)
        write_json(out/'status.json',status);write_json(out/'usage.json',usage)
        (out/'probe_report.md').write_text('\n'.join(['# 完整批次后的baseline评估（自动暂定）','',
            f'固定N={len(rows)}，已评估{len(completed)}，其余尚未处理。源生成/回答运行：{m["source_run"]}；源问题、图片与答案不改。','',
            f'可回答性/忠实性：{status["answerability_completed_only"]}。','',
            f'在可回答题中，双模态必要性：{necessity}。','',
            f'原TI回答支持情况单列：{status["original_TI_answer_support_completed_only"]}；拒答/技术失败不冒充题目无效。','',
            f'目标验收：{status["target_acceptance_completed_only"]}。模型核验失败只留方向未决，不翻相反标签。','',
            '保留文本充分时即可确认图像不必要，并停止该题后续方向调用；另一方向不被强行填0。未完成项不算失败，始终保留全量固定分母。','',
            '按既有K与open/verify-v2证据规则执行；未运行四条件或BDR。所有结果暂定自动，模型角色同为gpt-6-luna时有相关误差，不是独立人类金标准。需要抽查典型缺陷、候选双必要题及条件边界后再形成研究结论。','']))
        return status


async def _run(root,out,m,prompts,sources,provider_factory):
    n=m['settings']['concurrency'];sem=asyncio.Semaphore(n)
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=n))
    async def worker(source):
        async with sem:return await asyncio.to_thread(_evaluate,root,out,m,prompts,source,provider_factory)
    tasks=[asyncio.create_task(worker(s)) for s in sources];failures=0
    try:
        for i,task in enumerate(asyncio.as_completed(tasks),1):
            row=await task;print(json.dumps({'attempt_id':row['attempt_id'],'status':row['status']}),flush=True)
            itemcalls=read_jsonl(out/'items'/row['attempt_id']/'calls.jsonl')
            failed=bool(itemcalls) and all(c['status']=='technical_failure' for c in itemcalls)
            failures=failures+1 if failed else 0
            if failures>=5:raise RuntimeError('five consecutive evaluation infrastructure failures; stop queued work')
            if i%16==0:await asyncio.to_thread(report_evaluation,run_id=out.name,project_root=root,output_root=str(out.parent))
    except BaseException:
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True);raise


def run_evaluation(*,run_id,project_root='.',output_root='outputs',provider_factory=None,limit=None):
    root=Path(project_root).resolve();_validate_run_id(run_id);out=root/output_root/run_id
    m=json.loads((out/'manifest.json').read_text())
    for p,h in m['source_sha256'].items():
        if sha256_file(root/p)!=h:raise ValueError('evaluation source changed')
    for n,h in m['code_sha256'].items():
        if sha256_file(Path(__file__).with_name(n))!=h:raise ValueError('evaluation code changed')
    if sha256_file(out/'source_questions.jsonl')!=m['source_questions_sha256']:raise ValueError('evaluation questions changed')
    prompts=PromptSet(out/'prompts',names=names_for(m['settings']))
    if prompts.snapshot_sha256()!=m['prompt_sha256']:raise ValueError('evaluation prompts changed')
    sources=read_jsonl(out/'source_questions.jsonl')
    if limit is not None:sources=sources[:limit]
    with file_lock(out/'.pipeline.lock'):asyncio.run(_run(root,out,m,prompts,sources,provider_factory))
    return report_evaluation(run_id=run_id,project_root=root,output_root=output_root)
