"""Dataset-neutral Q generation/import, checklist review, bounded repair and yield metrics."""
from __future__ import annotations

import base64
import html
import json
import unicodedata
from collections import Counter
from pathlib import Path

from .generation import GenerationFailure, run_generation
from .pipeline import RunConfigMismatch, _validate_run_id, load_materials
from .prompts import PromptSet
from .providers import CallRecorder, CompletionRequest, JudgeProvider, ProviderError, RequestBudgetExceeded
from .quality import DIMENSIONS, _call, _check_assessment
from .records import JsonlTable, canonical_json, read_jsonl, sha256_text, stable_id, utc_now, write_json, write_jsonl

ROLES=('generator','checker','auditor','rewriter')
GATES=(*DIMENSIONS,'no_leakage')
REPAIR_ACTIONS={
    'scale_cancels':'Ask for a supported absolute modeled quantity whose value depends on the stated representation scale, not merely a ratio/comparison in which that scale cancels.',
    'unused_duration':'Choose a supported rate or other natural quantity that actually uses the passage duration; do not copy the duration into Q.',
    'quantity_scope':'Match the requested quantity and denominator to the actual source labels; for example solvent volume is not total solution volume. Do not invent missing measurements.',
    'ordinary_knowledge_shortcut':'Ordinary knowledge is a legitimate shortcut. Find a genuinely material-specific fact or return not_repairable; wording such as according to the passage is not a repair.',
    'text_repeated_in_question':'Remove copied decisive passage facts from Q, then ensure the requested answer genuinely requires reading them from T.',
    'image_fact_repeated':'Do not restate the decisive visible fact in T or Q; ask for a supported fact that must actually be read from I.',
    'independent_subquestions':'Replace concatenated independent questions with one solution that combines the two sources.',
    'unsupported_premise':'Use only supported premises and a clear referent. Do not invent facts or conditions to force a question.',
    'unclear_reference':'Make the requested answer scope and units explicit without disclosing the answer.'
}
REPAIR_DIAGNOSTICS=('Also include a repair_tags array in the same JSON object. Use only these exact codes when a concrete defect applies: '
                    + ', '.join(REPAIR_ACTIONS) + '. Use [] if none. These codes diagnose deficiencies, not desired answers. '
                    'In particular distinguish a scale that cancels in the requested ratio, an unused time interval, '
                    'and an unsupported quantity/denominator. Do not replace the other evidence fields.')


def rewrite_feedback(check,audit):
    codes=sorted({tag for row in (check,audit) for tag in row.get('repair_tags',[]) if tag in REPAIR_ACTIONS})
    return {'checks':{name:{k:row.get(k) for k in ('validity',*GATES)}
                      for name,row in (('checker',check),('auditor',audit))},
            'repair_tags':codes,'required_changes':[REPAIR_ACTIONS[c] for c in codes]}


def _check_repair_tags(data):
    tags=data.get('repair_tags',[])
    if not isinstance(tags,list) or any(not isinstance(t,str) or t not in REPAIR_ACTIONS for t in tags):
        raise ValueError('invalid controlled repair tags')


def normalized_question(q):
    # Case can encode scientific meaning (e.g. dominant I versus recessive i).
    return ' '.join(unicodedata.normalize('NFKC',q or '').split())


def gate_question(check, audit):
    """Accept requires two positive checks and a verified current-Q reference."""
    if check.get('error') or audit.get('error'):
        return 'unresolved', 'assessment_failure'
    cv,av=check.get('validity'),audit.get('validity')
    if {cv,av}=={'valid','invalid'}:
        return 'unresolved', 'validity_disagreement'
    if cv==av=='invalid':return 'rejected','invalid_question'
    if any(row.get(k) is False for row in (check,audit) for k in GATES):
        return 'rejected','checklist_defect'
    if check.get('repair_tags') or audit.get('repair_tags'):
        return 'unresolved','positive_checks_with_unresolved_defect'
    if cv==av=='valid' and all(row.get(k) is True for row in (check,audit) for k in GATES) and audit.get('reference_status')=='verified':
        return 'accepted','all_gates_passed'
    return 'unresolved','insufficient_or_conflicting_evidence'


def should_rewrite(evaluation):
    if evaluation['decision']=='rejected':return True
    check,audit=evaluation['check'],evaluation['audit']
    return (evaluation['decision']=='unresolved' and not check.get('error') and not audit.get('error')
            and bool(set(check.get('repair_tags',[])) & set(audit.get('repair_tags',[]))))


def refinement_metrics(plan, results):
    ids=[r['material_id'] for r in plan]
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('plan requires unique materials')
    by_id={}
    for r in results:
        mid=r['material_id']
        if mid not in ids or mid in by_id:raise ValueError('result missing from plan or duplicated')
        by_id[mid]=r
    rows=list(by_id.values());n=len(plan)
    attempted=[r for r in rows if r.get('rewrite') is not None]
    changed=[r for r in attempted if r['rewrite'].get('question') and
             normalized_question(r['initial'].get('question'))!=normalized_question(r['rewrite']['question'])]
    survived=[r for r in rows if r['final_status']=='accepted']
    initial_pass=[r for r in rows if r.get('initial_evaluation',{}).get('decision')=='accepted']
    repaired=[r for r in changed if r['final_status']=='accepted']
    unknown=sum(r['final_status']=='unresolved' for r in rows);pending=n-len(rows)
    def rate(a,b):return a/b if b else None
    return {'n_planned':n,'n_completed':len(rows),'n_pending':pending,
            'n_initial_questions':sum(bool(r['initial'].get('question')) for r in rows),
            'n_initial_survivors':len(initial_pass),'initial_survival_rate':len(initial_pass)/n,
            'n_final_survivors':len(survived),'final_survival_rate':len(survived)/n,
            'n_rewrite_attempted':len(attempted),'rewrite_trigger_rate':len(attempted)/n,
            'n_actually_changed':len(changed),'rewrite_rate':len(changed)/n,
            'change_rate_among_attempts':rate(len(changed),len(attempted)),
            'n_successfully_rewritten':len(repaired),'rewrite_success_rate':rate(len(repaired),len(attempted)),
            'n_recovered_rejected':sum(r['initial_evaluation']['decision']=='rejected' for r in repaired),
            'n_recovered_unresolved':sum(r['initial_evaluation']['decision']=='unresolved' for r in repaired),
            'rewritten_survivor_yield':len(repaired)/n,
            'net_survival_gain':(len(survived)-len(initial_pass))/n,
            'n_unresolved':unknown,'survival_bound_with_unresolved_and_pending':[len(survived)/n,(len(survived)+unknown+pending)/n],
            'final_states':dict(Counter(r['final_status'] for r in rows)),
            'rewrite_states':dict(Counter(r['rewrite']['status'] for r in attempted)),
            'human_confirmed_survival_rate':None,
            'interim_counts':'Numerators use completed material records; unfinished units are pending, not failures.',
            'scope':'Automatic pipeline gates only; not human quality accuracy. All planned materials remain in denominator.'}


def run_refinement_pipeline(*, materials_path, run_id, config_path='configs/refinement.json',
                            questions_path=None, project_root='.', output_root='outputs',
                            limit=30, max_requests=180, max_tokens=2048, dry_run=False, providers=None):
    _validate_run_id(run_id)
    if limit<1 or max_requests<1 or max_tokens<1:raise ValueError('limit and budgets must be positive')
    root=Path(project_root).resolve();out=root/output_root/run_id
    cfg=json.loads((root/config_path).read_text())
    allowed_settings={'protocol_version','generation_method','max_rewrites','roles','role_reuse_note'}
    if not isinstance(cfg,dict) or set(cfg)-allowed_settings:
        raise ValueError('unsupported refinement config fields; credentials belong in environment variables')
    if not isinstance(cfg.get('protocol_version'),str) or not cfg['protocol_version'].strip():
        raise ValueError('protocol_version must be a nonempty string')
    if cfg.get('generation_method') not in ('plain','strong') or cfg.get('max_rewrites')!=1:
        raise ValueError('portable pipeline supports plain/strong initialization and one bounded rewrite')
    if not isinstance(cfg.get('roles'),dict) or set(cfg['roles'])!=set(ROLES):raise ValueError('configure generator/checker/auditor/rewriter roles explicitly')
    for spec in cfg['roles'].values():
        if (not isinstance(spec,dict) or set(spec)-{'env_prefix','model'}
            or not isinstance(spec.get('env_prefix'),str) or not spec['env_prefix'].strip()
            or (spec.get('model') is not None and (not isinstance(spec['model'],str) or not spec['model'].strip()))):
            raise ValueError('each role accepts only a nonempty env_prefix and optional model')
    all_materials=load_materials(root,root/materials_path)
    materials=all_materials[:limit];all_ids={m['material_id'] for m in all_materials}
    supplied={}
    for r in read_jsonl(root/questions_path) if questions_path else []:
        mid=r.get('material_id');q=r.get('question')
        if not isinstance(mid,str) or mid in supplied or not isinstance(q,str) or not q.strip():
            raise ValueError('supplied questions require unique material_id and nonempty question')
        if mid not in all_ids:raise ValueError('supplied question material_id is not in the material file')
        source=r.get('source','supplied_question')
        if not isinstance(source,str) or not source.strip():raise ValueError('supplied question source must be a nonempty string')
        supplied[mid]={'question':q.strip(),'source':source}
    if questions_path and not (root/questions_path).is_file():raise FileNotFoundError('questions file missing')
    supplied={m['material_id']:supplied[m['material_id']] for m in materials if m['material_id'] in supplied}
    estimate={'n_planned':len(materials),'n_supplied_questions':len(supplied),
              'logical_calls_upper_without_retries':6*len(materials)-len(supplied),
              'http_budget':max_requests,'max_output_tokens_per_request':max_tokens,'estimated_cost':None}
    if dry_run:return estimate
    if providers is None:
        providers={role:JudgeProvider.from_env(root=root,run_dir=out,max_requests=max_requests,max_tokens=max_tokens,
                    visual=True,env_prefix=cfg['roles'][role]['env_prefix'],model=cfg['roles'][role].get('model')) for role in ROLES}
    if set(providers)!=set(ROLES):raise ValueError('all role providers are required')
    protocol=cfg['protocol_version']
    prompts=PromptSet(root/'prompts',names=(f'qg_{cfg["generation_method"]}','check_question','audit_question','refine_question'))
    plan=[{'material_id':m['material_id'],'source_group_id':m['source_group_id'],'split':m['split'],
           'initial_question_id':stable_id('q',run_id,m['material_id'],'initial'),
           'revised_question_id':stable_id('q',run_id,m['material_id'],'revised')} for m in materials]
    config={'settings':cfg,'materials':materials,'supplied_questions':supplied,'plan':plan,'prompts':prompts.snapshot(),
            'roles':{k:v.identity for k,v in providers.items()},
            'repair_diagnostics':REPAIR_DIAGNOSTICS,'repair_actions':REPAIR_ACTIONS,
            'feedback_policy':'controlled_defect_codes_and_static_actions_no_reference_answers',
            'decision_policy':'keep accepted; rewrite rejected or a shared controlled defect once; otherwise leave unresolved; no fallback relabeling'}
    fp=sha256_text(canonical_json(config));state_path=out/'run.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['config_sha256']!=fp:raise RunConfigMismatch('pipeline cohort/input/prompt/model changed; use a new run_id')
    else:
        if out.exists() and any(out.iterdir()):raise RunConfigMismatch('pipeline output directory is not empty')
        state={'run_id':run_id,'config_sha256':fp,'created_at':utc_now()}
    state.update(status='running',updated_at=utc_now(),http_budget=max_requests)
    state['execution']={role:{'timeout_seconds':getattr(p,'timeout',None),'max_http_attempts_per_request':getattr(p,'max_attempts',None)} for role,p in providers.items()}
    write_json(state_path,state);write_json(out/'config.json',config);write_json(out/'prompts.json',prompts.snapshot())
    write_jsonl(out/'materials.jsonl',materials);write_jsonl(out/'plan.jsonl',plan)
    tables={name:JsonlTable(out/(name+'.jsonl'),key) for name,key in (
        ('calls','call_id'),('initial_questions','material_id'),('checks','question_id'),('audits','question_id'),
        ('rewrites','material_id'),('results','material_id'))}
    recorder=CallRecorder(tables['calls'],run_id=run_id,protocol_version=protocol,default_model=providers['generator'].model)

    def assess(material,qid,question):
        mid=material['material_id'];image=str(root/material['image_path'])
        check=tables['checks'].get(qid)
        if check is None:
            prompt=prompts.render('check_question',text=material['text'],question=question)+'\n'+REPAIR_DIAGNOSTICS
            data,error=_call(providers['checker'],recorder,CompletionRequest(task='check_question',prompt=prompt,
                question=question,text=material['text'],image_path=image,metadata={'protocol_version':protocol}),
                role='checker',material_id=mid,question_id=qid)
            try:
                if error:raise ValueError(error)
                _check_assessment(data)
                _check_repair_tags(data)
                if 'no_leakage' not in data or (data['no_leakage'] is not None and type(data['no_leakage']) is not bool):raise ValueError('invalid leakage check')
            except ValueError as exc:data={'error':str(exc),'validity':'unresolved'}
            check={'question_id':qid,'material_id':mid,'question':question,**data}
            tables['checks'].add(check)
        audit=tables['audits'].get(qid)
        if audit is None:
            if check.get('error'):
                audit={'error':'checker_failed','validity':'unresolved'}
            else:
                prompt=prompts.render('audit_question',text=material['text'],question=question,
                    answer=check.get('answer') if isinstance(check.get('answer'),str) else '(no candidate answer)',
                    rubric=check.get('rubric') if isinstance(check.get('rubric'),str) else '(no rubric)')+'\n'+REPAIR_DIAGNOSTICS
                data,error=_call(providers['auditor'],recorder,CompletionRequest(task='audit_question',prompt=prompt,
                    question=question,text=material['text'],image_path=image,metadata={'protocol_version':protocol}),
                    role='auditor',material_id=mid,question_id=qid)
                try:
                    if error:raise ValueError(error)
                    _check_assessment(data)
                    _check_repair_tags(data)
                    if 'no_leakage' not in data or (data['no_leakage'] is not None and type(data['no_leakage']) is not bool):raise ValueError('invalid audit leakage check')
                    if data.get('reference_status') not in ('verified','rejected','unresolved') or not isinstance(data.get('reference_reason'),str) or not data['reference_reason'].strip():raise ValueError('invalid reference audit')
                    if data['reference_status']=='verified' and not (isinstance(check.get('answer'),str) and check['answer'].strip() and isinstance(check.get('rubric'),str) and check['rubric'].strip()):
                        raise ValueError('cannot verify missing reference answer/rubric')
                except ValueError as exc:data={'error':str(exc),'validity':'unresolved'}
                audit=data
            audit={'question_id':qid,'material_id':mid,'question':question,**audit}
            tables['audits'].add(audit)
        decision,reason=gate_question(check,audit)
        return {'question_id':qid,'decision':decision,'reason':reason,'check':check,'audit':audit}

    def publish(status):
        rows=tables['results'].values();metrics=refinement_metrics(plan,rows)
        write_json(out/'metrics.json',metrics)
        _export_questions(out,run_id,config,materials,rows)
        accounting=refinement_usage(out,tables['calls'].values())
        write_json(out/'usage.json',accounting)
        _report(out,root,materials,rows,metrics,accounting)
        state.update(status=status,updated_at=utc_now());write_json(state_path,state)
        return {'run_dir':str(out),'new_calls':recorder.added_count,'new_records':sum(t.added_count for t in tables.values()),
                'metrics':metrics,'usage':accounting}

    try:
        for material,slot in zip(materials,plan):
            mid=material['material_id']
            if tables['results'].get(mid) is not None:continue
            initial=tables['initial_questions'].get(mid)
            if initial is None:
                qid=slot['initial_question_id']
                if mid in supplied:
                    data={'status':'ok','question':supplied[mid]['question'],'source':supplied[mid]['source']}
                else:
                    try:
                        outcome=run_generation(providers['generator'],prompts,recorder,material,cfg['generation_method'],
                            material_image_path=root/material['image_path'],run_id=run_id,protocol_version=protocol,
                            question_id=qid,generator_model=providers['generator'].model)
                        data={'status':'ok','question':outcome.question,'source':'model_generated'}
                    except ProviderError as exc:data={'status':'technical_failure','question':None,'reason':str(exc),'source':'model_generated'}
                    except GenerationFailure as exc:data={'status':'generation_failure','question':None,'reason':exc.reason,'source':'model_generated'}
                initial={'material_id':mid,'question_id':qid,**data};tables['initial_questions'].add(initial)
            row={'material_id':mid,'initial':initial,'rewrite':None,'selected_question':None,'selected_question_id':None,
                 'human_review_status':'pending','selection':'none'}
            if not initial.get('question'):
                row['final_status']='no_question';tables['results'].add(row);continue
            before=assess(material,initial['question_id'],initial['question']);row['initial_evaluation']=before
            if before['decision']=='accepted':
                row.update(final_status='accepted',selected_question=initial['question'],selected_question_id=initial['question_id'],selection='kept_initial')
            elif not should_rewrite(before):
                row['final_status']='unresolved'
            else:
                rewrite=tables['rewrites'].get(mid)
                if rewrite is None:
                    # Only structural flags cross into the rewriter; reference answers and
                    # free-form evaluator reasoning cannot leak into generation.
                    feedback=rewrite_feedback(before['check'],before['audit'])
                    prompt=prompts.render('refine_question',text=material['text'],question=initial['question'],
                                         feedback=json.dumps(feedback,ensure_ascii=False))
                    data,error=_call(providers['rewriter'],recorder,CompletionRequest(task='refine_question',prompt=prompt,
                        question=initial['question'],text=material['text'],image_path=str(root/material['image_path']),
                        metadata={'protocol_version':protocol}),role='rewriter',material_id=mid,question_id=slot['revised_question_id'])
                    valid=not error and data.get('status') in ('revised','unchanged','not_repairable') and isinstance(data.get('reason'),str) and bool(data['reason'].strip()) and 'answer' not in data
                    if valid and data['status']=='not_repairable':valid=data.get('question') is None
                    elif valid:valid=isinstance(data.get('question'),str) and bool(data['question'].strip())
                    if valid and data['status']=='unchanged':valid=normalized_question(data['question'])==normalized_question(initial['question'])
                    if not valid:data={'status':'unresolved','question':None,'reason':error or 'invalid Q-only rewrite output'}
                    rewrite={'material_id':mid,'question_id':slot['revised_question_id'],'attempt_no':1,**data}
                    tables['rewrites'].add(rewrite)
                row['rewrite']=rewrite
                if not rewrite.get('question'):row['final_status']='no_proposal'
                else:
                    after=assess(material,rewrite['question_id'],rewrite['question']);row['revised_evaluation']=after
                    row['final_status']=after['decision']
                    if after['decision']=='accepted':row.update(selected_question=rewrite['question'],selected_question_id=rewrite['question_id'],selection='rewritten')
            tables['results'].add(row)
    except RequestBudgetExceeded:
        publish('budget_exhausted')
        raise
    return publish('complete')


def _export_questions(out,run_id,config,materials,rows):
    """Portable exports carry full run/material provenance, without reference answers."""
    by_mid={m['material_id']:m for m in materials};candidates=[];survivors=[]
    for row in rows:
        m=by_mid[row['material_id']]
        common={'run_id':run_id,'protocol_version':config['settings']['protocol_version'],
                'material_id':m['material_id'],'source_group_id':m['source_group_id'],'split':m['split']}
        initial={**common,'question_id':row['initial']['question_id'],'question':row['initial'].get('question'),
                 'phase':'initial','status':row['initial']['status'],'source':row['initial']['source'],
                 'assessment_status':row.get('initial_evaluation',{}).get('decision','not_evaluated')}
        candidates.append(initial)
        if row.get('rewrite') is not None:
            revision=row['rewrite']
            candidates.append({**common,'question_id':revision['question_id'],'question':revision.get('question'),
                               'phase':'revised','status':revision['status'],'source':'model_rewrite',
                               'assessment_status':row.get('revised_evaluation',{}).get('decision','not_evaluated')})
        if row['final_status']=='accepted':
            survivors.append({**common,'question_id':row['selected_question_id'],'question':row['selected_question'],
                              'selection':row['selection'],'status':'automatically_accepted','human_review_status':'pending'})
    write_jsonl(out/'candidates.jsonl',candidates)
    write_jsonl(out/'survivors.jsonl',survivors)


def refinement_usage(out, calls):
    ledger=read_jsonl(out/'requests.jsonl');started=sum(r['event']=='started' for r in ledger)
    usage=[r['usage'] for r in ledger if r['event']=='completed']
    tokens_known=len(usage)==started and all(u.get('total_tokens') is not None for u in usage)
    cost_known=len(usage)==started and all(u.get('cost') is not None for u in usage)
    accounting={'http_attempts':started,'completed_requests':len(usage),'requests_without_usage':started-len(usage),
        'known_total_tokens':sum(u.get('total_tokens') or 0 for u in usage),
        'total_tokens':sum(u['total_tokens'] for u in usage) if tokens_known else None,
        'reported_cost':sum(u['cost'] for u in usage) if cost_known else None,
        'cache_hits':sum(r.get('usage',{}).get('cache_hit',False) for r in calls),
        'calls_by_role':dict(Counter(r['role'] for r in calls))}
    return accounting


def _report(out,root,materials,rows,metrics,usage):
    n=metrics['n_planned'];a=metrics['n_rewrite_attempted']
    lines=['# 可迁移 checklist 改写 pipeline','',
           '同一材料、固定分母；通过原题保留，明确缺陷或两方共同定位的修复点最多改一次，其余未决留审。所有结果为自动判断，人工存活率未知。','',
           '| 指标 | 分子/分母 | 比例 |','|---|---:|---:|']
    for label,k,d in [('改前存活率','n_initial_survivors',n),('最终存活率','n_final_survivors',n),
                      ('改写触发率','n_rewrite_attempted',n),('实际改写率','n_actually_changed',n),
                      ('改写成功率（触发项）','n_successfully_rewritten',a)]:
        numerator=metrics[k];pct=f'{numerator/d:.1%}' if d else '未定义'
        lines.append(f'| {label} | {numerator}/{d} | {pct} |')
    lines += ['',f"待完成：{metrics['n_pending']}；评估未决：{metrics['n_unresolved']}；最终状态：{metrics['final_states']}。",
              '实际改写按Unicode/空白规范化后的Q文本比较，保留科学符号的大小写，不直接采用模型声称的revised标签；文字改变不等于质量改善。',
              f"请求{usage['http_attempts']}次，已知{usage['known_total_tokens']} tokens，费用缺失时记null。",'',
              '| 材料 | 原题关卡 | 改写状态 | 最终状态 |','|---|---|---|---|']
    for r in rows:lines.append(f"| {r['material_id']} | {r.get('initial_evaluation',{}).get('decision','no_question')} | {(r.get('rewrite') or {}).get('status','not_triggered')} | {r['final_status']} |")
    note_path=out/'qualitative_audit.json'
    note=json.loads(note_path.read_text()) if note_path.exists() else None
    if note:
        lines += ['', '## 非盲AI抽查提示（不改写自动计数）', '',
                  '自动通过仍可能误过常识捷径，不能据此直接外推人审合格数量。']
        for observation in note.get('observations',[]):
            lines.append(f"- {observation['material_id']}: {observation['reason']}")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    esc=lambda value:html.escape(str(value));by_mid={m['material_id']:m for m in materials}
    page=['<!doctype html><meta charset="utf-8"><title>Portable question refinement</title>',
          '<style>body{font:17px/1.6 sans-serif;max-width:1080px;margin:30px auto;padding:0 20px}article{border-top:2px solid #999;padding:20px 0}img{max-width:100%;max-height:340px}pre{white-space:pre-wrap}</style>',
          '<h1>生成/导入 → 检查 → 有界改写 → 复核</h1><p>自动存活不等于人工通过；全部原题、改题、未决与失败保留。</p>']
    if note:
        page += ['<details open><summary>非盲AI抽查提示：仍需独立核验</summary><pre>'+esc(json.dumps(note,ensure_ascii=False,indent=2))+'</pre></details>']
    for r in rows:
        m=by_mid[r['material_id']];data=(root/m['image_path']).read_bytes();mime='image/png' if data.startswith(b'\x89PNG') else 'image/jpeg'
        page += [f'<article><h2>{esc(r["material_id"])}</h2><p>{esc(m["text"])}</p>',
                 '<img alt="Original material" src="data:'+mime+';base64,'+base64.b64encode(data).decode()+'">',
                 '<h3>原题</h3><p>'+esc(r['initial'].get('question') or r['initial']['status'])+'</p>',
                 '<h3>改写候选</h3><p>'+esc((r.get('rewrite') or {}).get('question') or (r.get('rewrite') or {}).get('status','未触发'))+'</p>',
                 '<p>最终状态：'+esc(r['final_status'])+'</p><details><summary>检查、参考核验和来源</summary><pre>'+esc(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details></article>']
    (out/'review.html').write_text('\n'.join(page))


def summarize_refinement(*, run_id, project_root='.', output_root='outputs'):
    """Recompute reports from saved records; no credentials, network or providers."""
    _validate_run_id(run_id)
    root=Path(project_root).resolve();out=root/output_root/run_id
    config=json.loads((out/'config.json').read_text())
    if 'settings' not in config or 'decision_policy' not in config:
        raise ValueError('run is not a portable refinement pipeline')
    plan=read_jsonl(out/'plan.jsonl');rows=read_jsonl(out/'results.jsonl')
    materials=load_materials(root,out/'materials.jsonl')
    metrics=refinement_metrics(plan,rows)
    usage=refinement_usage(out,read_jsonl(out/'calls.jsonl'))
    write_json(out/'metrics.json',metrics);write_json(out/'usage.json',usage)
    write_json(out/'prompts.json',{**config['prompts'],'repair_diagnostics':config.get('repair_diagnostics','')})
    _export_questions(out,run_id,config,materials,rows)
    _report(out,root,materials,rows,metrics,usage)
    return {'run_dir':str(out),'metrics':metrics,'usage':usage,'new_calls':0,'new_records':0}
