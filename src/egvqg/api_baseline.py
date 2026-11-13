"""Resumable API baseline: fixed test slots, isolated answering and provisional evidence."""
from __future__ import annotations

import concurrent.futures
import copy
import json
import random
from pathlib import Path

from .assessment_rules import load_assessment_config
from .baseline_analysis import analysis_input, proof_shape
from .evidence_verification import verification_gate
from .experiments import _validate_answer, answering_input, call_usage, label, new_record, normalize_question
from .faithfulness import CHECKS, _decision
from .metrics import _three_and, compute_record_metrics
from .pipeline import _validate_run_id, load_materials
from .prompts import PromptSet
from .providers import JudgeProvider, ModelCaller, call_model
from .records import (canonical_json, file_lock, read_jsonl, sha256_file, sha256_text,
                      stable_id, utc_now, write_json, write_jsonl)

VERSION='natural-api-baseline-v1'
NAMES=('plain-v1','answer-v1','solve-v1','score-v1','review-question-faithfulness-v1',
       'open-necessity-v2','verify-necessity-evidence-v2')
TOKENS={'generation':2048,'candidate_answer':4096,'faithfulness':4096,
        'answer':2048,'score':1536,'analysis':4096,'verification':3072}
EXTRA=('faithfulness','analysis_T','analysis_I','verification_T','verification_I')


def init_api_baseline(*,run_id,materials_path,project_root='.',output_root='outputs',model='gpt-6-luna',workers=2):
    root=Path(project_root).resolve();_validate_run_id(run_id);out=root/output_root/run_id
    if not 1<=workers<=4:raise ValueError('workers must be 1..4')
    materials=load_materials(root,root/materials_path)
    prompts=PromptSet(root/'prompts',names=NAMES)
    cfg=load_assessment_config(root,'configs/necessity_assessment_v2.json')
    random.Random(20261112).shuffle(materials)
    plan=[{'attempt_id':stable_id('attempt',run_id,m['material_id']),'run_id':run_id,
           'method':'plain','plan_index':i,'material_id':m['material_id'],'item_id':f'Q{i+1:04d}'} for i,m in enumerate(materials)]
    sources={str((root/materials_path).relative_to(root)):sha256_file(root/materials_path),
             'configs/necessity_assessment_v2.json':sha256_file(root/'configs/necessity_assessment_v2.json')}
    code={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    frozen={'model':model,'workers':workers,'plan':plan,'materials':materials,'assessment':cfg,
            'tokens':TOKENS,'prompts':prompts.snapshot(),'source_hashes':sources,'code_hashes':code}
    fingerprint=sha256_text(canonical_json(frozen));out.mkdir(parents=True,exist_ok=True)
    with file_lock(out/'.init.lock'):
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text())['fingerprint']!=fingerprint:raise ValueError('API baseline config changed; use new run id')
            return {'n_planned':len(plan),'resumed':True}
        write_jsonl(out/'materials.jsonl',materials);write_jsonl(out/'plan.jsonl',plan)
        (out/'prompts').mkdir(exist_ok=True);(out/'code_snapshots').mkdir(exist_ok=True)
        for name,txt in prompts.snapshot().items():(out/'prompts'/f'{name}.txt').write_text(txt)
        for name in code:(out/'code_snapshots'/name).write_bytes((Path(__file__).parent/name).read_bytes())
        write_json(out/'manifest.json',{'created_at':utc_now(),'protocol_version':VERSION,
            'fingerprint':fingerprint,'model':model,'workers':workers,'n_planned':len(plan),
            'hard_http_cap':15*len(plan),'per_item_http_cap':15,'technical_attempts':1,
            'config':cfg,'tokens':TOKENS,'prompt_sha256':prompts.snapshot_sha256(),
            'source_hashes':sources,'code_hashes':code,
            'frozen_files':{name:sha256_file(out/name) for name in ('plan.jsonl','materials.jsonl')},
            'budget_note':'15 per material maximum: 1 generation, 1 candidate reference, 1 faithfulness, 4 answers, 4 scores, 2 analyses, 2 proof verifications. Sum of per-item hard ledger caps is the run cap. No retries/resampling.',
            'roles':'Same configured API model in fresh stateless role calls; verifier agreement is correlated. All labels are provisional automatic, never human/adjudicated.',
            'sampling':'All supplied test hint+image materials; fixed shuffled processing order seed20261112. First3/30 are stage checks, remain in the full denominator.',
            'scope':'API model baseline; not Qwen, not proposed-method evidence. Only hint and original material figures enter generation; no source Q/options/answers/solutions.'})
    return report_api_baseline(run_id=run_id,project_root=root,output_root=output_root)


def _load(root,out):
    m=json.loads((out/'manifest.json').read_text())
    for name,h in m['code_hashes'].items():
        if sha256_file(Path(__file__).with_name(name))!=h:raise ValueError('API baseline code changed: '+name)
    for path,h in m['source_hashes'].items():
        if sha256_file(root/path)!=h:raise ValueError('API baseline source changed')
    for path,h in m['frozen_files'].items():
        if sha256_file(out/path)!=h:raise ValueError('API baseline plan/material changed')
    prompts=PromptSet(out/'prompts',names=NAMES)
    if prompts.snapshot_sha256()!=m['prompt_sha256']:raise ValueError('API baseline prompt changed')
    return m,prompts


def _empty(slot,material,manifest):
    row=new_record(slot,material,{'protocol_version':VERSION},manifest['prompt_sha256'])
    for k in ('independent_answer','reference_resolution','relatedness','necessity','shortcuts'):
        row['steps'][k]='not_applicable'
    row['steps'].update({k:'pending' for k in EXTRA})
    row['faithfulness']={'review':{'item_id':slot['item_id'],'source':'model_api','reviewer':manifest['model']},'question_result':None}
    return row


def _process(root,out,manifest,prompts,slot,material,provider_factory=None):
    item=out/'items'/slot['attempt_id'];item.mkdir(parents=True,exist_ok=True)
    path=item/'record.json'
    row=json.loads(path.read_text()) if path.exists() else _empty(slot,material,manifest)
    if row['status']=='automatic_complete':return row
    if sha256_file(root/material['image_path'])!=material['image_sha256']:raise ValueError('material image changed')
    cfg=manifest['config'];q=lambda:row['question']['normalized_text'];ref=row['reference']
    def invoke(step,template,variables,*,family=None,text=None,image_path=None,condition=None,output_format='json'):
        family=family or step
        tokens=manifest['tokens'][family]
        provider=(provider_factory(item,tokens) if provider_factory else JudgeProvider.from_env(
            root=root,run_dir=item,max_requests=manifest['per_item_http_cap'],max_tokens=tokens,visual=True,model=manifest['model']))
        if isinstance(provider,JudgeProvider):provider.max_attempts=1
        with file_lock(out/'.identity.lock'):
            identity_path=out/'model_identities.json'
            identities=json.loads(identity_path.read_text()) if identity_path.exists() else {}
            identity=getattr(provider,'identity',{'model':provider.model})
            if identities.get(family,identity)!=identity:raise ValueError('API baseline model identity changed')
            identities[family]=identity;write_json(identity_path,identities)
        caller=ModelCaller(provider=provider,prompts=prompts,run_dir=item,protocol_version=VERSION,max_attempts=1)
        call=call_model(caller,operation_id=stable_id('api-baseline',slot['attempt_id'],step),attempt_id=slot['attempt_id'],
            role=step,template=template,variables=variables,task='generate_question' if step=='generation' else 'assess_quality',
            text=text,image_path=image_path,question=q(),condition=condition,output_format=output_format)
        row['step_outputs'][step]={'call_id':call['call_id'],'status':call['status'],'output':call.get('parsed_output')}
        return call
    steps=('generation','candidate_answer','faithfulness',*[f'answer_{c}' for c in ('TI','T','I','0')],
           *[f'score_{c}' for c in ('TI','T','I','0')],*EXTRA[1:])
    for step in steps:
        if row['steps'][step]!='pending':continue
        if step!='generation' and not q():
            row['steps'][step]='not_applicable';continue
        try:
            image=str(root/material['image_path']);base={'T':material['text'],'Q':q()}
            if step=='generation':
                c=invoke(step,'plain-v1',{'T':material['text']},text=material['text'],image_path=image,output_format='text')
                answer=c.get('parsed_output')
                if c['status']!='ok' or not isinstance(answer,str) or not answer.strip():raise ValueError('no complete generated question')
                text=normalize_question(answer)
                row['generation'].update(status='ok',raw_output=c['raw_output'],call_ids=[c['call_id']])
                row['question'].update(normalized_text=text,sha256=sha256_text(text),question_id=stable_id('q',slot['attempt_id'],text))
                ref.update(question_id=row['question']['question_id'],question_sha256=row['question']['sha256'])
            elif step=='candidate_answer':
                c=invoke(step,'answer-v1',base,text=material['text'],image_path=image)
                if c['status']!='ok':raise ValueError('reference request failed')
                ref['candidate_answer']=_validate_answer(c['parsed_output']);ref['status']='candidate_unreviewed';ref['call_ids']=[c['call_id']]
            elif step=='faithfulness':
                c=invoke(step,'review-question-faithfulness-v1',{**base,'KNOWLEDGE':cfg['knowledge'],
                    'REFERENCE':canonical_json(ref['candidate_answer'])},text=material['text'],image_path=image)
                if c['status']!='ok':raise ValueError('faithfulness request failed')
                data=c['parsed_output'];checks=data['question_checks']
                if set(checks)!=set(CHECKS):raise ValueError('invalid faithfulness check keys')
                v=_three_and(*(_decision(checks[k]) for k in CHECKS));rr=_decision(data['reference_supported'])
                row['faithfulness'].update(question_result=v,review={**data,'item_id':slot['item_id'],
                    'source':'model_api','reviewer':manifest['model'],'call_id':c['call_id']})
                row['labels']['automatic']['validity']=label(v,data['reason'])
                if v==1 and rr==1 and ref['candidate_answer'] and ref['candidate_answer']['status']=='answerable':
                    a=ref['candidate_answer'];ref.update(status='verified_automatic',verified_answer=a['answer'],
                        acceptable_answers=a['acceptable_answers'],rubric=a['rubric'],text_evidence=a['text_evidence'],
                        image_evidence=a['image_evidence'],verification_source='same_api_model_explicit_faithfulness_review')
                else:ref['status']='unresolved'
                if v==0:
                    for k in ('text_necessary','image_necessary','integration'):
                        row['labels']['automatic'][k]={'value':None,'status':'not_applicable','reason':'Q fails answerability/faithfulness; no necessity conclusion.','source':'automatic'}
            elif step.startswith('answer_'):
                cond=step.removeprefix('answer_');payload=answering_input(row,cond,root)
                c=invoke(step,'solve-v1',payload['variables'],family='answer',text=payload['text'],image_path=payload['image_path'],condition=cond)
                response={'question_id':row['question']['question_id'],'question_sha256':row['question']['sha256'],
                    'condition':cond,'call_id':c['call_id'],'status':'technical_failure','answer':None,'abstain':None,
                    'correct':None,'score_status':'unresolved'}
                row['responses'][cond]=response
                if c['status']!='ok':raise ValueError('answer request failed')
                d=c['parsed_output']
                if type(d.get('abstain')) is not bool or not isinstance(d.get('reason_brief'),str):raise ValueError('answer contract failure')
                if not d['abstain'] and (not isinstance(d.get('answer'),str) or not d['answer'].strip()):raise ValueError('answer contract failure')
                response.update(status='refused' if d['abstain'] else 'answered',answer=d.get('answer'),
                                abstain=d['abstain'],reason_brief=d['reason_brief'])
            elif step.startswith('score_'):
                cond=step.removeprefix('score_');response=row['responses'][cond]
                if ref['status']!='verified_automatic' or not response or response['status']!='answered':
                    row['steps'][step]='not_applicable';continue
                c=invoke(step,'score-v1',{'Q':q(),'ANSWER':ref['verified_answer'],
                    'ACCEPTABLE_ANSWERS':canonical_json(ref['acceptable_answers']),'RUBRIC':canonical_json(ref['rubric']),
                    'RESPONSE':canonical_json({k:response.get(k) for k in ('answer','reason_brief')})},family='score')
                d=c.get('parsed_output')
                if c['status']!='ok' or not isinstance(d,dict) or d.get('correct') not in (0,1,None):raise ValueError('score request/contract failed')
                response.update(correct=d['correct'],score_status='scored' if d['correct'] is not None else 'unresolved',score_reason=d.get('reason'),
                    score_reference_sha256=sha256_text(canonical_json({k:ref[k] for k in ('verified_answer','acceptable_answers','rubric')})))
            else:
                stage,removed=step.rsplit('_',1)
                if row['labels']['automatic']['validity']['value']!=1:
                    row['steps'][step]='not_applicable';continue
                proof=None
                if stage=='verification':
                    a=row['step_outputs'].get('analysis_'+removed,{})
                    proof=a.get('output')
                    if a.get('status')!='ok' or not proof_shape(proof):row['steps'][step]='not_applicable';continue
                payload=analysis_input(row,stage,removed,cfg,proof)
                if payload['image_path']:payload['image_path']=str(root/payload['image_path'])
                c=invoke(step,payload.pop('template'),payload.pop('variables'),family=stage,
                         **{k:v for k,v in payload.items() if k!='question'})
                if stage=='verification':
                    gate=verification_gate(c.get('parsed_output'),strict=True) if c['status']=='ok' else 'unresolved'
                    row['step_outputs'][step]['gate']=gate
                    value=(0 if proof['kind']=='unique' else 1) if gate=='accept' else None
                    row['labels']['automatic']['text_necessary' if removed=='T' else 'image_necessary']=label(value,
                        'Provisional API proof + verifier; no independent semantic gold. '+gate)
            row['steps'][step]='done'
        except (ValueError,KeyError,TypeError) as exc:
            row['steps'][step]='failed';row['notes'].append({'step':step,'error':str(exc),'at':utc_now()})
            if step=='generation':row['generation'].update(status='generation_failure',failure_code=str(exc))
        row['updated_at']=utc_now();write_json(path,row)
    row['status']='automatic_complete';row['usage']=call_usage(read_jsonl(item/'calls.jsonl'));write_json(path,row)
    return row


def report_api_baseline(*,run_id,project_root='.',output_root='outputs'):
    root=Path(project_root).resolve();out=root/output_root/run_id;m=json.loads((out/'manifest.json').read_text())
    plan=read_jsonl(out/'plan.jsonl');materials={r['material_id']:r for r in read_jsonl(out/'materials.jsonl')}
    rows=[];calls=[];requests=[]
    for slot in plan:
        item=out/'items'/slot['attempt_id'];p=item/'record.json'
        rows.append(json.loads(p.read_text()) if p.exists() else _empty(slot,materials[slot['material_id']],m))
        calls+=read_jsonl(item/'calls.jsonl');requests+=read_jsonl(item/'requests.jsonl')
    write_jsonl(out/'records.jsonl',rows);write_jsonl(out/'calls.jsonl',calls);write_jsonl(out/'requests.jsonl',requests)
    usage=call_usage(calls);write_json(out/'usage.json',usage)
    metrics=compute_record_metrics(rows,source='automatic');write_json(out/'metrics.json',{'source':'automatic_provisional','n_planned':len(plan),'automatic':metrics})
    a={'pass':0,'fail':0,'unresolved':0};n={'both_required':0,'not_both_required':0,'unresolved':0,'not_applicable':0}
    for r in rows:
        labs=r['labels']['automatic'];v=labs['validity']['value'] if labs['validity']['status']=='decided' else None
        if not r['question']['normalized_text'] and r['generation']['status']=='generation_failure':v=0
        a['pass' if v==1 else 'fail' if v==0 else 'unresolved']+=1
        dt,di=[labs[k]['value'] if labs[k]['status']=='decided' else None for k in ('text_necessary','image_necessary')]
        ns='not_applicable' if v!=1 else 'not_both_required' if 0 in (dt,di) else 'both_required' if dt==di==1 else 'unresolved';n[ns]+=1
    summary={'n_planned':len(plan),'completed':sum(r['status']=='automatic_complete' for r in rows),
        'answerability':a,'modality_necessity':n,'new_http_requests_cumulative':sum(r.get('event')=='started' for r in requests),
        'hard_http_cap':m['hard_http_cap'],'usage':usage,'source':'automatic_provisional; no human/adjudicated labels'}
    write_json(out/'status.json',summary)
    report=['# 全test API baseline（运行中结果为暂定自动评估）','',
        f'固定N={len(plan)}，已完成{summary["completed"]}组。生成/答案/评估角色使用{m["model"]}的独立无历史请求，同基座自核验不作为独立金标准。','',
        '## 可回答性与忠实性','',f'暂定通过{a["pass"]}、不通过{a["fail"]}、尚未完成或待裁定{a["unresolved"]}。','',
        '## 双模态必要性','',f'在可回答题中：暂定两者都必需{n["both_required"]}、至少一种可省略{n["not_both_required"]}、必要性待裁定{n["unresolved"]}；其余{n["not_applicable"]}不进入必要性裁定。','',
        '模型暂定结果须通过固定抽查再用于研究结论；未完成项保留全1257分母，不能把当前读数称为全test完成结果。实际答题表现与BDR另见metrics.json。','',
        f'累计HTTP {summary["new_http_requests_cumulative"]}，上限{m["hard_http_cap"]}；tokens与可得费用见usage.json。所有原始输出包括失败保留在items/*/calls.jsonl，并汇总到calls.jsonl。','',
        '没有改变必要性定义、K或历史协议；这批为API模型普通提示baseline，不与Qwen旧结果合并。','']
    (out/'probe_report.md').write_text('\n'.join(report));return summary


def run_api_baseline(*,run_id,limit=None,project_root='.',output_root='outputs',provider_factory=None):
    root=Path(project_root).resolve();out=root/output_root/run_id;m,prompts=_load(root,out)
    plan=read_jsonl(out/'plan.jsonl');materials={r['material_id']:r for r in read_jsonl(out/'materials.jsonl')}
    if limit is not None:
        if not 1<=limit<=len(plan):raise ValueError('limit must belong to fixed plan')
        plan=plan[:limit]
    with file_lock(out/'.pipeline.lock'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=m['workers']) as pool:
            futures={pool.submit(_process,root,out,m,prompts,s,materials[s['material_id']],provider_factory):s for s in plan}
            failures=0
            try:
                for index,future in enumerate(concurrent.futures.as_completed(futures),1):
                    row=future.result();print(json.dumps({'item':futures[future]['item_id'],'status':row['status']},ensure_ascii=False),flush=True)
                    itemcalls=read_jsonl(out/'items'/row['attempt_id']/'calls.jsonl')
                    all_failed=bool(itemcalls) and all(c['status']=='technical_failure' for c in itemcalls)
                    failures=failures+1 if all_failed else 0
                    if failures>=3:raise RuntimeError('three consecutive material-level infrastructure failures; stopped pending work')
                    if index%3==0 and index<=30 or index%30==0:report_api_baseline(run_id=run_id,project_root=root,output_root=output_root)
            except BaseException:
                for f in futures:f.cancel()
                raise
    return report_api_baseline(run_id=run_id,project_root=root,output_root=output_root)
