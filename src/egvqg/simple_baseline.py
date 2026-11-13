"""Simple API baseline: bounded images, asynchronous generation and TI answering only."""
from __future__ import annotations

import asyncio
import copy
import io
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image,ImageOps

from .experiments import answering_input,call_usage,new_record,normalize_question
from .pipeline import _validate_run_id,load_materials
from .prompts import PromptSet
from .providers import JudgeProvider,ModelCaller,call_model
from .records import (canonical_json,file_lock,read_jsonl,sha256_file,sha256_text,
                      stable_id,utc_now,write_json,write_jsonl)

VERSION='simple-api-baseline-v1'
NAMES=('plain-v1','solve-v1')


def bounded_image(root,material,directory,policy):
    """Reference compliant originals; resize only when a declared bound is exceeded."""
    result=copy.deepcopy(material);source=root/material['image_path']
    with Image.open(source) as original:
        width,height=original.size
        if (max(width,height)<=policy['max_edge'] and width*height<=policy['max_pixels']
                and source.stat().st_size<=policy['max_bytes']):
            result['image_size_control']={'changed':False,'width':width,'height':height,'bytes':source.stat().st_size,**policy}
            return result
        image=ImageOps.exif_transpose(original).convert('RGB')
        scale=min(1,policy['max_edge']/max(image.size),math.sqrt(policy['max_pixels']/(image.width*image.height)))
        image.thumbnail((max(1,int(image.width*scale)),max(1,int(image.height*scale))),Image.Resampling.LANCZOS)
    while True:
        buf=io.BytesIO();image.save(buf,format='PNG',optimize=True);data=buf.getvalue();extension='png'
        if len(data)<=policy['max_bytes']:break
        for quality in (90,80,70,60):
            buf=io.BytesIO();image.save(buf,format='JPEG',quality=quality,optimize=True);data=buf.getvalue();extension='jpg'
            if len(data)<=policy['max_bytes']:break
        if len(data)<=policy['max_bytes']:break
        if min(image.size)<=16:raise ValueError('image byte budget too small')
        image=image.resize((max(1,int(image.width*.8)),max(1,int(image.height*.8))),Image.Resampling.LANCZOS)
    directory.mkdir(parents=True,exist_ok=True);target=directory/f'{material["material_id"]}.{extension}'
    if target.exists() and target.read_bytes()!=data:raise ValueError('bounded image changed')
    if not target.exists():target.write_bytes(data)
    result.update(source_image_path=material['image_path'],source_image_sha256=material['image_sha256'],
                  image_path=str(target.relative_to(root)),image_sha256=sha256_file(target))
    result['image_size_control']={'changed':True,'source_width':width,'source_height':height,
        'width':image.width,'height':image.height,'bytes':len(data),**policy}
    return result


def reuse_matches(prior,request,identity,root):
    """Check actual API-visible content and effective identity, excluding hidden run metadata."""
    a=prior.get('actual_input',{})
    images=[] if request['image_path'] is None else [{'sha256':sha256_file(root/request['image_path']),'detail':'high'}]
    old_images=[{k:i[k] for k in ('sha256','detail')} for i in a.get('images',[])]
    return (a.get('prompt')==request['prompt'] and a.get('question')==request['question']
        and a.get('text')==request['text'] and a.get('condition')==request['condition']
        and a.get('history')==[] and old_images==images and a.get('model_config')==identity
        and a.get('metadata',{}).get('output_format')==request['output_format'])


def init_simple_baseline(*,run_id,materials_path,project_root='.',output_root='outputs',
                         model='gpt-6-luna',concurrency=16,max_image_edge=1024,
                         max_image_pixels=1000000,max_image_bytes=524288,reuse_run=None):
    root=Path(project_root).resolve();_validate_run_id(run_id);out=root/output_root/run_id
    if not 1<=concurrency<=32 or not 128<=max_image_edge<=4096 or max_image_pixels<16384 or max_image_bytes<16384:
        raise ValueError('invalid concurrency/image limits')
    policy={'max_edge':max_image_edge,'max_pixels':max_image_pixels,'max_bytes':max_image_bytes}
    prompts=PromptSet(root/'prompts',names=NAMES);materials=load_materials(root,root/materials_path)
    random.Random(20261112).shuffle(materials)
    out.mkdir(parents=True,exist_ok=True)
    materials=[bounded_image(root,m,out/'images',policy) for m in materials]
    plan=[{'attempt_id':stable_id('attempt',run_id,m['material_id']),'run_id':run_id,'method':'plain',
           'material_id':m['material_id'],'plan_index':i,'item_id':f'Q{i+1:04d}'} for i,m in enumerate(materials)]
    reuse={};sources={str((root/materials_path).relative_to(root)):sha256_file(root/materials_path)}
    if reuse_run:
        _validate_run_id(reuse_run);prior=root/output_root/reuse_run
        for folder in (prior/'items').glob('*'):
            record=folder/'record.json'
            if not record.exists():continue
            row=json.loads(record.read_text());calls=[c for c in read_jsonl(folder/'calls.jsonl')
                if c['role'] in ('generation','answer_TI') and c['status']!='budget_paused']
            if calls:
                reuse[row['material_id']]={c['role']:{'source_run':reuse_run,'source_path':str((folder/'calls.jsonl').relative_to(root)),
                    'source_file_sha256':sha256_file(folder/'calls.jsonl'),'call':c} for c in calls}
                sources[str((folder/'calls.jsonl').relative_to(root))]=sha256_file(folder/'calls.jsonl')
    cfg={'model':model,'concurrency':concurrency,'max_tokens':2048,'image_policy':policy,
         'hard_http_cap':2*len(plan),'per_item_http_cap':2,'max_attempts':1,'reuse_run':reuse_run,
         'phase':'generation_and_TI_answering_only','evaluation':'after_generation_and_answering_complete'}
    code={name:sha256_file(Path(__file__).with_name(name)) for name in ('simple_baseline.py','providers.py','records.py','prompts.py','experiments.py')}
    fingerprint=sha256_text(canonical_json({'config':cfg,'plan':plan,'materials':materials,'reuse':reuse,
                         'prompts':prompts.snapshot(),'code':code,'sources':sources}))
    with file_lock(out/'.init.lock'):
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text())['fingerprint']!=fingerprint:raise ValueError('simple baseline config changed')
            return {'n_planned':len(plan),'resumed':True}
        write_jsonl(out/'materials.jsonl',materials);write_jsonl(out/'plan.jsonl',plan);write_json(out/'reuse_index.json',reuse)
        (out/'prompts').mkdir(exist_ok=True);(out/'code_snapshots').mkdir(exist_ok=True)
        for name,txt in prompts.snapshot().items():(out/'prompts'/f'{name}.txt').write_text(txt)
        for name in code:(out/'code_snapshots'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        write_json(out/'manifest.json',{'created_at':utc_now(),'protocol_version':VERSION,'fingerprint':fingerprint,
            'n_planned':len(plan),'config':cfg,'code_sha256':code,'source_sha256':sources,
            'prompt_sha256':prompts.snapshot_sha256(),
            'frozen_files':{n:sha256_file(out/n) for n in ('materials.jsonl','plan.jsonl','reuse_index.json')},
            'scope':'Plain natural generation and one independent TI answer per material. No reference construction, no four-condition run, no scoring/necessity calls during this phase. All failures remain in N.'})
    return report_simple_baseline(run_id=run_id,project_root=root,output_root=output_root)


def _empty(slot,material,manifest):
    row=new_record(slot,material,{'protocol_version':VERSION},manifest['prompt_sha256'])
    row['steps']={k:'pending' for k in ('generation','answer_TI')};row['evaluation_status']='not_started'
    row['reuse']={};return row


def _process_item(root,out,manifest,prompts,slot,material,reuse,provider_factory=None):
    item=out/'items'/slot['attempt_id'];item.mkdir(parents=True,exist_ok=True);path=item/'record.json'
    row=json.loads(path.read_text()) if path.exists() else _empty(slot,material,manifest)
    if row['status']=='generation_answer_complete':return row
    if sha256_file(root/material['image_path'])!=material['image_sha256']:raise ValueError('input image changed')
    control=material['image_size_control']
    if max(control['width'],control['height'])>control['max_edge'] or control['width']*control['height']>control['max_pixels'] or (root/material['image_path']).stat().st_size>control['max_bytes']:
        raise ValueError('image size policy violated')
    cfg=manifest['config'];q=lambda:row['question']['normalized_text']
    provider=(provider_factory(item,cfg['max_tokens']) if provider_factory else JudgeProvider.from_env(root=root,run_dir=item,
        max_requests=cfg['per_item_http_cap'],max_tokens=cfg['max_tokens'],visual=True,model=cfg['model']))
    if isinstance(provider,JudgeProvider):provider.max_attempts=1
    identity=getattr(provider,'identity',{'model':provider.model})
    with file_lock(out/'.identity.lock'):
        fp=out/'model_identity.json'
        if fp.exists() and json.loads(fp.read_text())!=identity:raise ValueError('model/effective decoding changed')
        if not fp.exists():write_json(fp,identity)
    caller=ModelCaller(provider=provider,prompts=prompts,run_dir=item,protocol_version=VERSION,max_attempts=1)
    for stage in ('generation','answer_TI'):
        if row['steps'][stage]!='pending':continue
        if stage=='answer_TI' and not q():row['steps'][stage]='not_applicable';continue
        if stage=='generation':
            template='plain-v1';variables={'T':material['text']};fmt='text';text=material['text'];image=material['image_path'];condition=None
        else:
            template='solve-v1';payload=answering_input(row,'TI',root);variables=payload['variables'];fmt='json'
            text=payload['text'];image=payload['image_path'];condition='TI'
        request={'prompt':prompts.render(template,**variables),'question':q(),'text':text,
                 'image_path':image,'condition':condition,'output_format':fmt}
        prior=reuse.get(stage)
        if prior and reuse_matches(prior['call'],request,identity,root):
            call=copy.deepcopy(prior['call']);row['reuse'][stage]={'source_run':prior['source_run'],
                'source_path':prior['source_path'],'source_file_sha256':prior['source_file_sha256'],'source_call_id':call['call_id']}
            if call['status']=='started':call.update(status='technical_failure',error='source_interrupted:billing_unknown')
            copies=read_jsonl(item/'reused_calls.jsonl')
            if not any(c['role']==stage for c in copies):write_jsonl(item/'reused_calls.jsonl',[*copies,prior['call']])
        else:
            call=call_model(caller,operation_id=stable_id('simple',slot['attempt_id'],stage),attempt_id=slot['attempt_id'],
                role=stage,template=template,variables=variables,task='generate_question' if stage=='generation' else 'assess_quality',
                text=text,image_path=str(root/image),question=q(),condition=condition,output_format=fmt)
        row['step_outputs'][stage]={'status':call['status'],'call_id':call['call_id'],'output':call.get('parsed_output'),
                                   'reused':stage in row['reuse']}
        try:
            if call['status']!='ok':raise ValueError(call.get('error') or call['status'])
            d=call.get('parsed_output')
            if stage=='generation':
                if not isinstance(d,str) or not d.strip():raise ValueError('empty generated question')
                question=normalize_question(d)
                row['generation'].update(status='ok',raw_output=call['raw_output'],call_ids=[call['call_id']])
                row['question'].update(question_id=stable_id('q',slot['attempt_id'],question),normalized_text=question,sha256=sha256_text(question))
            else:
                if not isinstance(d,dict) or type(d.get('abstain')) is not bool or not isinstance(d.get('reason_brief'),str):raise ValueError('answer contract failure')
                if not d['abstain'] and (not isinstance(d.get('answer'),str) or not d['answer'].strip()):raise ValueError('answer contract failure')
                row['responses']['TI']={'question_id':row['question']['question_id'],'question_sha256':row['question']['sha256'],
                    'condition':'TI','status':'refused' if d['abstain'] else 'answered','answer':d.get('answer'),
                    'abstain':d['abstain'],'reason_brief':d['reason_brief'],'call_id':call['call_id'],
                    'correct':None,'score_status':'not_evaluated'}
            row['steps'][stage]='done'
        except (ValueError,TypeError) as exc:
            row['steps'][stage]='failed';row['notes'].append({'step':stage,'error':str(exc),'at':utc_now()})
            if stage=='generation':row['generation'].update(status='generation_failure',failure_code=str(exc),raw_output=call.get('raw_output'))
            else:row['responses']['TI']={'question_id':row['question']['question_id'],'question_sha256':row['question']['sha256'],
                'condition':'TI','status':'technical_failure' if call['status']=='technical_failure' else 'contract_failure',
                'answer':None,'abstain':None,'reason_brief':None,'call_id':call['call_id'],'correct':None,'score_status':'not_evaluated'}
        row['updated_at']=utc_now();write_json(path,row)
    row['status']='generation_answer_complete';row['updated_at']=utc_now();row['usage']=call_usage(read_jsonl(item/'calls.jsonl'));write_json(path,row)
    return row


def report_simple_baseline(*,run_id,project_root='.',output_root='outputs'):
    root=Path(project_root).resolve();out=root/output_root/run_id
    with file_lock(out/'.report.lock'):
        m=json.loads((out/'manifest.json').read_text());plan=read_jsonl(out/'plan.jsonl')
        materials={r['material_id']:r for r in read_jsonl(out/'materials.jsonl')};rows=[];calls=[];requests=[];reused=[]
        for slot in plan:
            folder=out/'items'/slot['attempt_id'];p=folder/'record.json'
            rows.append(json.loads(p.read_text()) if p.exists() else _empty(slot,materials[slot['material_id']],m))
            calls+=read_jsonl(folder/'calls.jsonl');requests+=read_jsonl(folder/'requests.jsonl');reused+=read_jsonl(folder/'reused_calls.jsonl')
        completed=sum(r['status']=='generation_answer_complete' for r in rows)
        usage=call_usage(calls)
        status={'n_planned':len(plan),'completed':completed,'pending':len(plan)-completed,
            'questions_generated':sum(bool(r['question']['normalized_text']) for r in rows),
            'TI_answered':sum((r['responses']['TI'] or {}).get('status')=='answered' for r in rows),
            'TI_refused':sum((r['responses']['TI'] or {}).get('status')=='refused' for r in rows),
            'generation_failures':sum(r['generation']['status']=='generation_failure' for r in rows),
            'TI_failures':sum((r['responses']['TI'] or {}).get('status') in ('technical_failure','contract_failure') for r in rows),
            'concurrency':m['config']['concurrency'],'reused_prior_calls':len(reused),
            'new_http_requests':sum(r.get('event')=='started' for r in requests),'hard_http_cap':m['config']['hard_http_cap'],
            'evaluation_status':'ready_not_started' if completed==len(plan) else 'not_started',
            'phase':'generation_and_TI_answering_only','usage':usage}
        write_jsonl(out/'records.jsonl',rows);write_jsonl(out/'calls.jsonl',calls);write_jsonl(out/'requests.jsonl',requests)
        write_json(out/'status.json',status);write_json(out/'usage.json',usage)
        write_jsonl(out/'questions.jsonl',[{'attempt_id':r['attempt_id'],'material_id':r['material_id'],'question':r['question'],
                    'status':r['generation']['status']} for r in rows])
        write_jsonl(out/'answers.jsonl',[{'attempt_id':r['attempt_id'],'material_id':r['material_id'],
                    'question_sha256':r['question']['sha256'],'TI_response':r['responses']['TI']} for r in rows])
        (out/'probe_report.md').write_text('\n'.join(['# 简单API baseline：生成与完整图文回答','',
            f'固定N={len(plan)}，已完成{completed}，异步并发{m["config"]["concurrency"]}。模型{m["config"]["model"]}，普通plain-v1生成，独立TI回答。','',
            f'生成Q {status["questions_generated"]}，TI作答{status["TI_answered"]}，拒答{status["TI_refused"]}，生成失败{status["generation_failures"]}，TI技术/格式失败{status["TI_failures"]}。全部名额保留。','',
            f'本轮新HTTP {status["new_http_requests"]}，输入/模型完全匹配的旧调用复用{len(reused)}；旧调用原始用量不重复计入本轮新消费。图片上限见manifest，原图不改。','',
            '本阶段不构造参考、不做四条件、不评分、不判必要性。跑完生成与回答后再评估；当前不能用这些完成数宣称可回答性或双模态达标率。',
            '原始新输出含失败在calls.jsonl，复用原始输出在items/*/reused_calls.jsonl；questions.jsonl与answers.jsonl按attempt_id和question_sha256关联。','']))
        return status


async def _run_async(root,out,manifest,prompts,plan,materials,reuse,provider_factory):
    concurrency=manifest['config']['concurrency'];semaphore=asyncio.Semaphore(concurrency)
    loop=asyncio.get_running_loop();loop.set_default_executor(ThreadPoolExecutor(max_workers=concurrency))
    async def worker(slot):
        async with semaphore:
            return await asyncio.to_thread(_process_item,root,out,manifest,prompts,slot,
                materials[slot['material_id']],reuse.get(slot['material_id'],{}),provider_factory)
    tasks=[asyncio.create_task(worker(s)) for s in plan];technical_streak=0
    try:
        for i,future in enumerate(asyncio.as_completed(tasks),1):
            row=await future
            print(json.dumps({'item_index':row['plan_index'],'status':row['status'],'reused':list(row['reuse'])}),flush=True)
            current=read_jsonl(out/'items'/row['attempt_id']/'calls.jsonl')
            failed=bool(current) and all(c['status']=='technical_failure' for c in current)
            technical_streak=technical_streak+1 if failed else 0
            if technical_streak>=5:raise RuntimeError('five consecutive infrastructure-failed items; stop queued work')
            if i%16==0:await asyncio.to_thread(report_simple_baseline,run_id=out.name,project_root=root,output_root=str(out.parent))
    except BaseException:
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        raise


def run_simple_baseline(*,run_id,project_root='.',output_root='outputs',limit=None,provider_factory=None):
    root=Path(project_root).resolve();_validate_run_id(run_id);out=root/output_root/run_id
    m=json.loads((out/'manifest.json').read_text())
    for name,h in m['code_sha256'].items():
        if sha256_file(Path(__file__).with_name(name))!=h:raise ValueError('simple baseline code changed')
    for name,h in m['source_sha256'].items():
        if sha256_file(root/name)!=h:raise ValueError('simple baseline source changed')
    for name,h in m['frozen_files'].items():
        if sha256_file(out/name)!=h:raise ValueError('simple baseline plan changed')
    prompts=PromptSet(out/'prompts',names=NAMES)
    if prompts.snapshot_sha256()!=m['prompt_sha256']:raise ValueError('simple baseline prompt changed')
    plan=read_jsonl(out/'plan.jsonl')
    if limit is not None:
        if not 1<=limit<=len(plan):raise ValueError('limit outside fixed plan')
        plan=plan[:limit]
    materials={r['material_id']:r for r in read_jsonl(out/'materials.jsonl')};reuse=json.loads((out/'reuse_index.json').read_text())
    with file_lock(out/'.pipeline.lock'):
        asyncio.run(_run_async(root,out,m,prompts,plan,materials,reuse,provider_factory))
    return report_simple_baseline(run_id=run_id,project_root=root,output_root=output_root)
