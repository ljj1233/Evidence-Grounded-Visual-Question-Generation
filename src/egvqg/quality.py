"""Real-material diagnostic controls for question-quality measurement (not gold labels)."""
from __future__ import annotations

import base64
import html
import json
from collections import Counter
from pathlib import Path

from .evaluation import answer_question, bdr_state, score_response
from .metrics import compute_metrics, render_summary
from .pipeline import RunConfigMismatch, _validate_run_id, load_materials
from .prompts import PromptSet
from .providers import CallRecorder, CompletionRequest, JudgeProvider, OutputParseError, ProviderError, parse_json_object
from .records import CONDITIONS, JsonlTable, canonical_json, read_jsonl, sha256_text, stable_id, utc_now, write_json, write_jsonl

PROTOCOL = 'quality-diagnostic-v1'
LABELS = ('high', 'valid_non_target', 'invalid', 'unresolved')
DIMENSIONS = ('text_needed', 'image_needed', 'integrated', 'educational')
PROBE = ('Describe only what is actually visible in the attached image, including labels, numbers and their positions. '
         'Do not infer a question or use other context. Return JSON: {"visible_text":["labels"],"description":"brief description"}.')


def quality_label(assessment: dict, reference_status: str) -> str:
    if assessment.get('validity') == 'invalid':
        return 'invalid'
    if assessment.get('validity') != 'valid' or reference_status != 'verified':
        return 'unresolved'
    if any(assessment.get(k) is False for k in DIMENSIONS):
        return 'valid_non_target'
    if all(assessment.get(k) is True for k in DIMENSIONS):
        return 'high'
    return 'unresolved'


def _check_assessment(data: dict) -> dict:
    if data.get('validity') not in ('valid', 'invalid', 'unresolved'):
        raise ValueError('invalid validity field')
    for key in DIMENSIONS:
        if data.get(key) is not None and type(data.get(key)) is not bool:
            raise ValueError('quality dimensions must be booleans or null')
        if key not in data:
            raise ValueError('missing quality dimension')
    for key in ('reason', 'evidence_text', 'evidence_image'):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError('missing quality evidence/reason')
    if not isinstance(data.get('flags'), list) or not all(isinstance(f, str) for f in data['flags']):
        raise ValueError('invalid quality flags')
    return data


def _call(provider, recorder, request, *, role, material_id, question_id=None):
    try:
        completion = recorder.complete(provider, request, role=role, material_id=material_id,
                                       question_id=question_id, model=provider.model)
        if completion.finish_reason != 'stop':
            raise OutputParseError('incomplete completion')
        return parse_json_object(completion.text), None
    except (ProviderError, OutputParseError) as exc:
        return {}, str(exc)


def diagnose_quality(*, materials_path: str, cases_path: str, run_id: str,
                     project_root: str | Path = '.', output_root: str | Path = 'outputs',
                     limit: int = 3, max_requests: int = 120, max_tokens: int = 2048,
                     dry_run: bool = False, provider=None) -> dict:
    _validate_run_id(run_id)
    if limit < 1:
        raise ValueError('material limit must be positive')
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    materials = load_materials(root, root / materials_path)[:limit]
    by_material = {m['material_id']: m for m in materials}
    cases = [c for c in read_jsonl(root / cases_path) if c['material_id'] in by_material]
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('controls must be nonempty with unique case IDs')
    for case in cases:
        if case.get('expected_label') not in LABELS or not isinstance(case.get('question'), str) or not case['question'].strip():
            raise ValueError('invalid control question/expected label')
        if case.get('expected_label_source') != 'AI_authored_before_model_evaluation' or case.get('human_review_status') != 'pending':
            raise ValueError('diagnostic expects explicitly AI-authored, non-gold controls')
        if case.get('reference_answer') is not None and (not isinstance(case['reference_answer'], str) or not case.get('rubric')):
            raise ValueError('reference requires an answer and rubric')
        for probe in case.get('answer_probes', []):
            if not isinstance(probe.get('answer'), str) or type(probe.get('expected_correct')) is not bool:
                raise ValueError('invalid answer probe')
    provider = provider or JudgeProvider.from_env(root=root, run_dir=out, max_requests=max_requests,
                                                  max_tokens=max_tokens, visual=True)
    prompts = PromptSet(root / 'prompts', names=('assess_quality', 'verify_reference', 'answer_question', 'score_response'))
    references = sum(c.get('reference_answer') is not None for c in cases)
    n_probes = sum(len(c.get('answer_probes', [])) for c in cases)
    upper = len(materials) + len(cases) * 5 + references * 5 + n_probes
    estimate = {'n_materials': len(materials), 'n_questions': len(cases),
                'request_upper_without_retries_or_cache': upper, 'hard_http_budget': max_requests,
                'max_output_tokens_per_request': max_tokens, 'estimated_cost': None,
                'label_source': 'AI controls, not human gold', 'concurrency': 1}
    if dry_run:
        return estimate
    config = {'protocol_version': PROTOCOL, 'materials': materials, 'cases': cases,
              'provider': provider.identity, 'prompts': prompts.snapshot(), 'probe_prompt': PROBE,
              'limit': limit, 'roles': {r: provider.model for r in ('quality_assessor', 'verifier', 'answerer', 'scorer')},
              'role_reuse': 'Same endpoint model; separate stateless calls. No independent human truth.',
              'label_source': 'AI authored before evaluation; not human gold'}
    fingerprint = sha256_text(canonical_json(config))
    state_path = out / 'run.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['config_sha256'] != fingerprint:
            raise RunConfigMismatch('diagnostic materials/cases/prompts/model changed; use a new run_id')
    else:
        if out.exists() and any(out.iterdir()):
            raise RunConfigMismatch('diagnostic output directory is not empty')
        state = {'run_id': run_id, 'config_sha256': fingerprint, 'created_at': utc_now()}
    state.update(status='running', updated_at=utc_now(), max_requests=max_requests)
    write_json(state_path, state)
    write_json(out / 'config.json', config)
    write_json(out / 'prompts.json', {**prompts.snapshot(), 'probe_image': PROBE})
    write_jsonl(out / 'materials.jsonl', materials)
    write_jsonl(out / 'controls.jsonl', cases)
    keys = {'calls': 'call_id', 'probes': 'material_id', 'assessments': 'question_id',
            'questions': 'question_id', 'validity': 'question_id', 'answers': 'answer_id',
            'responses': 'response_id', 'scores': 'score_id', 'answer_probes': 'response_id'}
    tables = {name: JsonlTable(out / f'{name}.jsonl', key) for name, key in keys.items()}
    recorder = CallRecorder(tables['calls'], run_id=run_id, protocol_version=PROTOCOL, default_model=provider.model)
    for material in materials:
        mid = material['material_id']
        if tables['probes'].get(mid) is None:
            data, error = _call(provider, recorder, CompletionRequest(task='probe_image', prompt=PROBE,
                                image_path=str(root/material['image_path']), metadata={'protocol_version': PROTOCOL}),
                                role='vision_probe', material_id=mid)
            tables['probes'].add({'material_id': mid, 'output': data, 'error': error})
    plan = []
    for case in cases:
        mid = case['material_id']; material = by_material[mid]
        qid = stable_id('q', run_id, case['case_id'])
        common = dict(run_id=run_id, protocol_version=PROTOCOL, question_id=qid,
                      material_id=mid, method='diagnostic_control')
        plan.append(dict(question_id=qid, material_id=mid, source_group_id=material['source_group_id'],
                         split=material['split'], method='diagnostic_control', case_id=case['case_id']))
        if tables['questions'].get(qid) is None:
            tables['questions'].add({**common, 'question': case['question'], 'status': 'ok',
                                     'source': 'AI_authored_diagnostic_control', 'case_id': case['case_id']})
        image = root / material['image_path']
        assessment = tables['assessments'].get(qid)
        if assessment is None:
            prompt = prompts.render('assess_quality', text=material['text'], question=case['question'])
            data, error = _call(provider, recorder, CompletionRequest(task='assess_quality', prompt=prompt,
                question=case['question'], text=material['text'], image_path=str(image), metadata={'protocol_version': PROTOCOL}),
                role='quality_assessor', material_id=mid, question_id=qid)
            try:
                if error:
                    raise ValueError(error)
                _check_assessment(data)
            except ValueError as exc:
                data = {'validity': 'unresolved', **{k: None for k in DIMENSIONS}, 'reason': str(exc),
                        'evidence_text': '', 'evidence_image': '', 'flags': ['unresolved_assessment']}
            assessment = {**common, **data, 'label_source': 'automatic_model_assessment'}
            tables['assessments'].add(assessment)
        if tables['validity'].get(qid) is None:
            tables['validity'].add({**common, 'status': assessment['validity'], 'source': 'automatic_model',
                                    'reasons': [assessment['reason']], 'validator_model': provider.model})
        aid = stable_id('answer', run_id, qid)
        answer = tables['answers'].get(aid)
        if answer is None:
            verification = {'status': 'unresolved', 'reason': 'no_predefined_reference'}
            if case.get('reference_answer') is not None:
                prompt = prompts.render('verify_reference', text=material['text'], question=case['question'],
                                         answer=case['reference_answer'], rubric=case['rubric'])
                verification, error = _call(provider, recorder, CompletionRequest(task='verify_reference',
                    prompt=prompt, question=case['question'], text=material['text'], image_path=str(image),
                    metadata={'protocol_version': PROTOCOL}), role='verifier', material_id=mid, question_id=qid)
                if (error or verification.get('status') not in ('verified', 'rejected', 'unresolved')
                    or not all(isinstance(verification.get(k), str) and verification[k].strip()
                               for k in ('reason', 'evidence_text', 'evidence_image'))):
                    verification = {'status': 'unresolved', 'reason': error or 'invalid_reference_check'}
            verified = verification['status'] == 'verified'
            answer = {**common, 'answer_id': aid, 'status': 'verified' if verified else 'unresolved',
                      'candidate_answer': case.get('reference_answer'),
                      'answer': case.get('reference_answer') if verified else None,
                      'rubric': case.get('rubric') if verified else None, 'acceptable_answers': [],
                      'verification': verification, 'source': 'AI_proposal_with_model_check_not_human_gold'}
            tables['answers'].add(answer)
        for condition in CONDITIONS:
            rid = stable_id('response', run_id, qid, condition)
            response = tables['responses'].get(rid)
            if response is None:
                response = answer_question(provider, prompts, recorder, material=material, method='diagnostic_control',
                    question=case['question'], condition=condition, material_image_path=image, run_id=run_id,
                    protocol_version=PROTOCOL, question_id=qid, answerer_model=provider.model).to_dict()
                tables['responses'].add(response)
            sid = stable_id('score', run_id, rid)
            if tables['scores'].get(sid) is None:
                tables['scores'].add(score_response(response=response, answer=answer, run_id=run_id,
                    protocol_version=PROTOCOL, scorer=provider.model, question=case['question'],
                    provider=provider, prompts=prompts, recorder=recorder).to_dict())
        for index, probe in enumerate(case.get('answer_probes', [])):
            rid = stable_id('answer_probe', run_id, qid, index)
            if tables['answer_probes'].get(rid) is None:
                response = {**common, 'response_id': rid, 'status': 'answered', 'answer': probe['answer'], 'condition': 'TI'}
                score = score_response(response=response, answer=answer, run_id=run_id, protocol_version=PROTOCOL,
                    scorer=provider.model, question=case['question'], provider=provider, prompts=prompts, recorder=recorder)
                tables['answer_probes'].add({'response_id': rid, 'question_id': qid, 'response': probe['answer'],
                    'expected_correct': probe['expected_correct'], 'correct': score.correct, 'rationale': score.rationale,
                    'label_source': 'AI_expected_not_human_gold'})
    write_jsonl(out / 'plan.jsonl', plan)
    scienceqa = all(m.get('provenance', '').startswith('ScienceQA@') for m in materials)
    metrics = compute_metrics(plan, tables['questions'].values(), tables['validity'].values(), tables['answers'].values(),
        tables['responses'].values(), tables['scores'].values(), run_id=run_id, protocol_version=PROTOCOL,
        evidence_class=('real_scienceqa_AI_authored_diagnostic_controls_not_human_gold' if scienceqa
                        else 'synthetic_or_other_diagnostic_controls_not_human_gold'))
    rows = []
    answers_by_q = {a['question_id']: a for a in tables['answers'].values()}
    scores_by_q = {}
    for score in tables['scores'].values():
        scores_by_q.setdefault(score['question_id'], {})[score['condition']] = score['correct']
    for case, slot in zip(cases, plan):
        qid = slot['question_id']; a = tables['assessments'].get(qid)
        label = quality_label(a, answers_by_q[qid]['status'])
        bdr, bdr_reason = bdr_state(tables['questions'].get(qid), tables['validity'].get(qid), scores_by_q.get(qid, {}))
        rows.append({**case, 'question_id': qid, 'predicted_label': label, 'assessment': a,
                     'reference': answers_by_q[qid], 'bdr': bdr, 'bdr_reason': bdr_reason,
                     'condition_scores': scores_by_q.get(qid, {})})
    confusion = {label: {other: 0 for other in LABELS} for label in LABELS}
    for row in rows:
        confusion[row['expected_label']][row['predicted_label']] += 1
    qmetrics = {'n_planned': len(rows), 'label_source': 'AI expectations; agreement is NOT accuracy',
                'n_agree': sum(r['expected_label'] == r['predicted_label'] for r in rows),
                'n_unresolved': sum(r['predicted_label'] == 'unresolved' for r in rows),
                'class_counts': dict(Counter(r['predicted_label'] for r in rows)), 'confusion': confusion,
                'false_high_against_AI_expectations': sum(r['expected_label'] != 'high' and r['predicted_label'] == 'high' for r in rows),
                'answer_probe_n': len(tables['answer_probes']),
                'answer_probe_agree': sum(r['expected_correct'] == r['correct'] for r in tables['answer_probes'].values())}
    qmetrics.update(agreement_of_planned=qmetrics['n_agree']/len(rows),
                    validity_only_accepts=sum(r['assessment']['validity'] == 'valid' for r in rows),
                    both_needed_without_integration_accepts=sum(r['assessment']['validity'] == 'valid'
                        and r['assessment']['text_needed'] is True and r['assessment']['image_needed'] is True for r in rows),
                    condition_response_states=dict(Counter(r['status'] for r in tables['responses'].values())))
    metrics['quality_diagnostic'] = qmetrics
    metrics['warning'] = ('Real ScienceQA materials' if scienceqa else 'Synthetic/other diagnostic materials') + ' + AI-authored controls + automatic judgements; no human gold or method comparison.'
    write_json(out / 'metrics.json', metrics)
    write_jsonl(out / 'quality_results.jsonl', rows)
    ledger = read_jsonl(out / 'requests.jsonl')
    started = sum(r['event'] == 'started' for r in ledger)
    usage = [r['usage'] for r in ledger if r['event'] == 'completed']
    tokens_known = len(usage) == started and all(u.get('total_tokens') is not None for u in usage)
    cost_known = len(usage) == started and all(u.get('cost') is not None for u in usage)
    accounting = {'http_attempts': started, 'completed_requests': len(usage),
                  'requests_without_usage': started - len(usage),
                  'known_total_tokens': sum(u.get('total_tokens') or 0 for u in usage),
                  'total_tokens': sum(u['total_tokens'] for u in usage) if tokens_known else None,
                  'reported_cost': sum(u['cost'] for u in usage) if cost_known else None,
                  'elapsed_request_seconds': round(sum(u.get('elapsed_seconds', 0) for u in usage), 3),
                  'cache_hits': sum(r.get('usage', {}).get('cache_hit', False) for r in tables['calls'].values())}
    write_json(out / 'usage.json', accounting)
    _report(out, root, materials, rows, tables, metrics, accounting)
    state.update(status='complete', updated_at=utc_now())
    write_json(state_path, state)
    return {**estimate, 'run_dir': str(out), 'new_calls': recorder.added_count,
            'new_records': sum(t.added_count for t in tables.values()), 'quality': qmetrics, 'usage': accounting}


def _report(out, root, materials, rows, tables, metrics, accounting):
    base_summary = render_summary(metrics).replace('预先安排的生成尝试数 N', '预先固定的诊断问题数 N')
    base_summary = base_summary.replace('## 分方法', '## 诊断问题组（不是方法比较）').replace(
        '本汇总只验证记录、隔离、分母与恢复逻辑，不能作为方法有效性或真实模型表现的证据。',
        '本汇总展示小规模真实材料诊断；AI预期标签和同模型评估不能建立人工准确率或方法有效性。')
    lines = [base_summary,
             '\n## Question-quality diagnostic (AI expectations, not human accuracy)\n',
             '| Case | Expected | Automatic | T necessary | I necessary | Integrated | BDR proxy |', '|---|---|---|---|---|---|---|']
    for r in rows:
        a = r['assessment']
        lines.append(f"| {r['case_id']} | {r['expected_label']} | {r['predicted_label']} | {a['text_needed']} | {a['image_needed']} | {a['integrated']} | {r['bdr']} ({r['bdr_reason']}) |")
    qm = metrics['quality_diagnostic']
    lines += ['', '## 判据对照', '',
              f"- 仅按有效性筛选会接受 {qm['validity_only_accepts']}/{qm['n_planned']} 题。",
              f"- 有效且图文均必要、但不检查整合性，会接受 {qm['both_needed_without_integration_accepts']}/{qm['n_planned']} 题。",
              f"- 完整规则自动标高档 {qm['class_counts'].get('high', 0)} 题；与AI预期类别一致 {qm['n_agree']}/{qm['n_planned']}，这不是人工准确率。",
              f"- 回答边界对照与AI预期一致 {qm['answer_probe_agree']}/{qm['answer_probe_n']}；不混入四条件答题或BDR分母。",
              f"- 四条件回答状态：{qm['condition_response_states']}。材料不足拒答与技术失败分开保留；都不冒充答错。",
              f"- 请求 {accounting['http_attempts']} 次，完成 {accounting['completed_requests']} 次；已知用量 {accounting['known_total_tokens']} tokens，未返回用量的请求 {accounting['requests_without_usage']} 次；总量/费用未知不补零。"]
    lines += ['', 'Expected labels and proposed answers were AI-authored before endpoint calls. All evaluation roles reuse the same model.',
              'Refusal stays unresolved under the current protocol; BDR is not a direct quality score.',
              'See review.html for actual images, passages, evidence, reference checks and four-condition responses.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    esc = lambda value: html.escape(str(value))
    parts = ['<!doctype html><meta charset="utf-8"><title>Real-material question-quality diagnostic</title>',
             '<style>body{font:17px/1.6 sans-serif;max-width:1040px;margin:32px auto;padding:0 20px}img{max-width:100%;max-height:360px}article{border-top:2px solid #aaa;margin-top:30px;padding-top:16px}pre{white-space:pre-wrap}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>',
             '<h1>真实材料：问题质量诊断</h1><p>原始 ScienceQA 图文 + AI 起草对照题。全部标签待人工确认；不是正式实验或评分准确率。</p>',
             '<p>每题先独立查看图文和问题，再展开自动评估与预期标签。没有预填人工审查结论。</p>']
    for m in materials:
        image = root/m['image_path']
        parts += [f'<h2>{esc(m["material_id"])}</h2>', f'<p>{esc(m["text"])}</p>',
                  '<img alt="Original ScienceQA image" src="data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()+'">']
        for r in rows:
            if r['material_id'] != m['material_id']: continue
            parts += [f'<article><h3>{esc(r["case_id"])} · {esc(r["question"])}</h3>',
                      '<details><summary>自动判定与证据</summary><pre>'+esc(json.dumps(r['assessment'],ensure_ascii=False,indent=2))+'</pre>',
                      '<pre>'+esc(json.dumps(r['reference'],ensure_ascii=False,indent=2))+'</pre>',
                      '<p>Classification: '+esc(r['predicted_label'])+'; BDR proxy: '+esc(r['bdr'])+'</p><table><tr><th>条件</th><th>实际回答</th><th>评分</th></tr>']
            for response in tables['responses'].values():
                if response['question_id'] == r['question_id']:
                    parts.append('<tr><td>'+esc(response['condition'])+'</td><td>'+esc(response.get('answer') or response['status'])+'</td><td>'+esc(r['condition_scores'].get(response['condition']))+'</td></tr>')
            parts += ['</table></details><details><summary>AI预期标签（非人工金标准）</summary>',
                      '<p>'+esc(r['expected_label'])+': '+esc(r['expected_reason'])+'</p></details></article>']
    (out/'review.html').write_text('\n'.join(parts))


def refine_questions(*, materials_path: str, cases_path: str, run_id: str,
                     project_root: str | Path = '.', output_root: str | Path = 'outputs',
                     limit: int = 3, max_requests: int = 30, max_tokens: int = 2048,
                     dry_run: bool = False, provider=None) -> dict:
    """One checklist-guided Q-only rewrite, with blind before/after checks.

    No-proposal attempts stay in the fixed denominator. This is adaptive
    training-material diagnosis, not a held-out method comparison.
    """
    protocol='checklist-repair-v2'
    _validate_run_id(run_id)
    if limit<1:raise ValueError('limit must be positive')
    root=Path(project_root).resolve();out=root/output_root/run_id
    all_materials=load_materials(root,root/materials_path)
    cases=read_jsonl(root/cases_path)[:limit]
    if not cases or len({c['case_id'] for c in cases})!=len(cases):raise ValueError('unique nonempty cases required')
    mids={c['material_id'] for c in cases}
    materials=[m for m in all_materials if m['material_id'] in mids];by_mid={m['material_id']:m for m in materials}
    if set(by_mid)!=mids:raise ValueError('case material missing')
    for c in cases:
        if not all(isinstance(c.get(k),str) and c[k].strip() for k in ('question','feedback')):raise ValueError('question and feedback required')
        if 'proposed_revision' in c and (not isinstance(c['proposed_revision'],dict) or not c['proposed_revision'].get('source')):
            raise ValueError('an externally proposed revision requires explicit provenance')
    provider=provider or JudgeProvider.from_env(root=root,run_dir=out,max_requests=max_requests,max_tokens=max_tokens,visual=True)
    prompts=PromptSet(root/'prompts',names=('refine_question','check_question','verify_reference'))
    estimate={'n_cases':len(cases),'logical_call_upper_without_retries':sum(
                5 if 'proposed_revision' not in c else (4 if c['proposed_revision'].get('question') else 2) for c in cases),
              'max_http_attempts':max_requests,'max_tokens':max_tokens,'estimated_cost':None}
    if dry_run:return estimate
    config={'protocol_version':protocol,'cases':cases,'materials':materials,'prompts':prompts.snapshot(),
            'provider':provider.identity,'rewrite_attempts_per_case':1,
            'feedback_policy':'review_plus_blind_precheck_excluding_candidate_answer_and_rubric',
            'roles':{r:provider.model for r in ('rewriter','checklist_assessor','reference_verifier')},
            'scope':'Adaptive training-material pilot; same model reused; no human gold or held-out claims.'}
    if any('proposed_revision' in c for c in cases):
        config['revision_sources']={c['case_id']:c.get('proposed_revision',{}).get('source',provider.model) for c in cases}
    fingerprint=sha256_text(canonical_json(config));state_path=out/'run.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['config_sha256']!=fingerprint:raise RunConfigMismatch('repair inputs/prompts/model changed; use new run_id')
    else:
        if out.exists() and any(out.iterdir()):raise RunConfigMismatch('repair directory not empty')
        state={'run_id':run_id,'config_sha256':fingerprint,'created_at':utc_now()}
    state.update(status='running',updated_at=utc_now());write_json(state_path,state)
    write_json(out/'config.json',config);write_jsonl(out/'materials.jsonl',materials);write_jsonl(out/'cases.jsonl',cases)
    tables={name:JsonlTable(out/(name+'.jsonl'),key) for name,key in (
        ('calls','call_id'),('repairs','case_id'),('checks','question_id'),('references','question_id'),('results','case_id'))}
    recorder=CallRecorder(tables['calls'],run_id=run_id,protocol_version=protocol,default_model=provider.model)

    def evaluate(case,question,phase):
        mid=case['material_id'];material=by_mid[mid];qid=stable_id('q',run_id,case['case_id'],phase)
        check=tables['checks'].get(qid)
        if check is None:
            prompt=prompts.render('check_question',text=material['text'],question=question)
            data,error=_call(provider,recorder,CompletionRequest(task='check_question',prompt=prompt,question=question,
                text=material['text'],image_path=str(root/material['image_path']),metadata={'protocol_version':protocol}),
                role='checklist_assessor',material_id=mid,question_id=qid)
            try:
                if error:raise ValueError(error)
                _check_assessment(data)
                if 'no_leakage' not in data or (data['no_leakage'] is not None and type(data['no_leakage']) is not bool):raise ValueError('invalid leakage check')
                if not all(isinstance(data.get(k),str) and data[k].strip() for k in ('shortcut_check','text_change_effect','image_change_effect')):raise ValueError('missing shortcut/effect check')
            except ValueError as exc:
                data={'validity':'unresolved',**{k:None for k in DIMENSIONS},'no_leakage':None,'reason':str(exc),
                      'evidence_text':'','evidence_image':'','flags':['unresolved_check'],'answer':None,'rubric':None}
            check={'question_id':qid,'case_id':case['case_id'],'phase':phase,'question':question,**data}
            tables['checks'].add(check)
        reference=tables['references'].get(qid)
        if reference is None:
            verified={'status':'unresolved','reason':'candidate_answer_or_rubric_missing'}
            if isinstance(check.get('answer'),str) and check['answer'].strip() and isinstance(check.get('rubric'),str) and check['rubric'].strip():
                prompt=prompts.render('verify_reference',text=material['text'],question=question,answer=check['answer'],rubric=check['rubric'])
                verified,error=_call(provider,recorder,CompletionRequest(task='verify_reference',prompt=prompt,question=question,
                    text=material['text'],image_path=str(root/material['image_path']),metadata={'protocol_version':protocol}),
                    role='reference_verifier',material_id=mid,question_id=qid)
                if error or verified.get('status') not in ('verified','rejected','unresolved') or not all(
                    isinstance(verified.get(k),str) and verified[k].strip() for k in ('reason','evidence_text','evidence_image')):
                    verified={'status':'unresolved','reason':error or 'invalid reference verification'}
            reference={'question_id':qid,**verified,'human_review_status':'pending'}
            tables['references'].add(reference)
        label=quality_label(check,reference['status'])
        if check['validity']=='invalid' and reference['status']=='verified':
            label='unresolved'  # Conflicting automatic judges cannot establish invalidity.
        if label=='high' and check['no_leakage'] is not True:
            label='valid_non_target' if check['no_leakage'] is False else 'unresolved'
        return {'question_id':qid,'question':question,'label':label,'checks':check,'reference_check':reference}

    for case in cases:
        if tables['results'].get(case['case_id']) is not None:continue
        before=evaluate(case,case['question'],'before')
        repair=tables['repairs'].get(case['case_id']);material=by_mid[case['material_id']]
        if repair is None:
            feedback={'review':case['feedback'],'precheck':{k:before['checks'].get(k) for k in (
                'validity',*DIMENSIONS,'no_leakage','evidence_text','evidence_image',
                'shortcut_check','text_change_effect','image_change_effect','reason')}}
            if 'proposed_revision' in case:
                data,error=dict(case['proposed_revision']),None
            else:
                prompt=prompts.render('refine_question',text=material['text'],question=case['question'],feedback=json.dumps(feedback,ensure_ascii=False))
                data,error=_call(provider,recorder,CompletionRequest(task='refine_question',prompt=prompt,question=case['question'],
                    text=material['text'],image_path=str(root/material['image_path']),metadata={'protocol_version':protocol}),
                    role='rewriter',material_id=case['material_id'])
            valid=(not error and data.get('status') in ('revised','unchanged','not_repairable')
                   and isinstance(data.get('reason'),str) and bool(data['reason'].strip()) and 'answer' not in data)
            if valid and data['status']=='not_repairable':valid=data.get('question') is None
            elif valid:valid=isinstance(data.get('question'),str) and bool(data['question'].strip())
            if valid and data['status']=='unchanged':valid=data['question']==case['question']
            if not valid:data={'status':'unresolved','question':None,'reason':error or 'invalid Q-only revision output'}
            repair={'case_id':case['case_id'],**data,'attempt_no':1,
                    'source':case.get('proposed_revision',{}).get('source',provider.model)};tables['repairs'].add(repair)
        after=evaluate(case,repair['question'],'after') if repair.get('question') else {'question':None,'label':'no_proposal','reason':repair['reason']}
        tables['results'].add({'case_id':case['case_id'],'material_id':case['material_id'],
                              'before':before,'repair':repair,'after':after,'human_review_status':'pending'})
    rows=tables['results'].values()
    stats={'n_planned_repairs':len(cases),'before_counts':dict(Counter(r['before']['label'] for r in rows)),
           'after_counts':dict(Counter(r['after']['label'] for r in rows)),
           'n_improved_to_automatic_high':sum(r['before']['label']!='high' and r['after']['label']=='high' for r in rows),
           'n_no_proposal':sum(r['after']['label']=='no_proposal' for r in rows),
           'scope':config['scope'],'human_confirmed':0}
    ledger=read_jsonl(out/'requests.jsonl');started=sum(r['event']=='started' for r in ledger)
    usage=[r['usage'] for r in ledger if r['event']=='completed']
    accounting={'http_attempts':started,'completed_requests':len(usage),'known_total_tokens':sum(u.get('total_tokens') or 0 for u in usage),
                'reported_cost':sum(u['cost'] for u in usage) if len(usage)==started and all(u.get('cost') is not None for u in usage) else None}
    write_json(out/'metrics.json',stats);write_json(out/'usage.json',accounting)
    lines=['# Checklist Q改写小样本','',config['scope'],'',f"预先安排{len(cases)}次改写，每题最多一次；没有返回Q的尝试保留。",'',
           '| 材料 | 改前自动类别 | 改后自动类别 | 改写状态 |','|---|---|---|---|']
    for r in rows:lines.append(f"| {r['material_id']} | {r['before']['label']} | {r['after']['label']} | {r['repair']['status']} |")
    for r in rows:
        lines += ['', '## '+r['material_id'], '', '**原题：** '+r['before']['question'], '',
                  '**改后：** '+(r['after']['question'] or '本轮未得到合规改写。'), '', '**改写理由：** '+r['repair']['reason'],
                  '', '**改前检查：** '+r['before']['checks']['reason'], '',
                  '**改后检查：** '+(r['after'].get('checks',{}).get('reason') or r['after'].get('reason',''))]
    lines += ['', '以上是自适应训练材料上的同模型反馈试验，不是独立校准或正式方法比较。未改原图文；原题、失败及全部调用保留。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    esc=lambda value:html.escape(str(value))
    page=['<!doctype html><meta charset="utf-8"><title>Checklist question revision</title>',
          '<style>body{font:17px/1.6 sans-serif;max-width:1050px;margin:30px auto;padding:0 16px}img{max-width:100%;max-height:350px}article{border-top:2px solid #aaa;padding:20px 0}pre{white-space:pre-wrap}</style>',
          '<h1>Checklist：原题与改写题</h1><p>真实图文未修改。编辑来源逐项记录；模型检查并非人工金标准，失败与未修复项全部保留。</p>']
    for r in rows:
        material=by_mid[r['material_id']];data=(root/material['image_path']).read_bytes()
        mime='image/png' if data.startswith(b'\x89PNG') else 'image/jpeg'
        page += [f'<article><h2>{esc(r["material_id"])}</h2><p>{esc(material["text"])}</p>',
                 '<img alt="Original image" src="data:'+mime+';base64,'+base64.b64encode(data).decode()+'">',
                 '<h3>原题</h3><p>'+esc(r['before']['question'])+'</p>',
                 '<h3>改写/编辑后</h3><p>'+esc(r['after']['question'] or '本轮未接受合规改写')+'</p>',
                 '<p>来源：'+esc(r['repair'].get('source'))+'；'+esc(r['repair']['reason'])+'</p>',
                 '<details><summary>检查与参考答案核验记录（AI判断）</summary><pre>'+esc(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details></article>']
    (out/'review.html').write_text('\n'.join(page))
    state.update(status='complete',updated_at=utc_now());write_json(state_path,state)
    return {'run_dir':str(out),'new_calls':recorder.added_count,'new_records':sum(t.added_count for t in tables.values()),'metrics':stats,'usage':accounting}
