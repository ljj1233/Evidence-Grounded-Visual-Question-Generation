"""Offline, evidence-bound question faithfulness review; source runs stay read-only."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .metrics import _three_and, compute_record_metrics
from .pipeline import _validate_run_id
from .records import (canonical_json, file_lock, read_jsonl, sha256_file, sha256_text,
                      stable_id, utc_now, write_json, write_jsonl)

VERSION = 'question-faithfulness-review-v1'
CHECKS = ('presupposition_supported', 'answer_supported', 'scope_clear')
CODES = {'unsupported_presupposition', 'missing_decisive_premise',
         'ambiguous_request_scope', 'unsupported_reference'}
REPORT_LAYOUT = 'separated-answerability-necessity-v1'


def separated_assessment(rows):
    """Presentation of existing V and D labels; never infer new necessity labels."""
    answerability = {'pass': 0, 'fail': 0, 'unresolved': 0}
    necessity = {'both_required': 0, 'not_both_required': 0,
                 'unresolved': 0, 'not_applicable': 0}
    target = {'pass': 0, 'fail': 0, 'unresolved': 0}
    items = []
    for row in rows:
        labels = row['labels']['adjudicated']
        value = lambda k: labels[k]['value'] if labels[k]['status'] == 'decided' else None
        v, dt, di = (value(k) for k in ('validity', 'text_necessary', 'image_necessary'))
        a = 'pass' if v == 1 else 'fail' if v == 0 else 'unresolved'
        n = ('not_applicable' if v != 1 else 'not_both_required' if 0 in (dt, di)
             else 'both_required' if dt == di == 1 else 'unresolved')
        t = ('fail' if v == 0 or n == 'not_both_required' else
             'pass' if v == 1 and n == 'both_required' else 'unresolved')
        answerability[a] += 1; necessity[n] += 1; target[t] += 1
        items.append({'item_id': row['faithfulness']['review']['item_id'],
                      'answerability': a, 'text_necessary': dt, 'image_necessary': di,
                      'modality_necessity': n, 'target_acceptance': t})
    return {'answerability': answerability, 'modality_necessity': necessity,
            'target_acceptance': target, 'items': items}


def separated_report(rows, assessment, source_run, analysis_note=None):
    """Two distinct review tables plus the requested combined acceptance count."""
    a, n, t = (assessment[k] for k in ('answerability', 'modality_necessity', 'target_acceptance'))
    escape = lambda x: str(x).replace('|', '\\|').replace('\n', '<br>')
    out = ['# Baseline：可回答性与双模态必要性分开报告', '',
        f'固定全部生成N={len(rows)}。目标是“有材料依据、能够回答，且文本和图像缺一不可”的题数。', '',
        f'**目标验收：确认达标{t["pass"]}题；确认不达标{t["fail"]}题；必要性或可回答性待裁定{t["unresolved"]}题。** 待裁定项不计入已确认达标分子，也不冒充已证明不达标。', '',
        '## 一、可回答性与忠实性', '',
        '只问：完整图文与固定知识K能否支持原问题的完整答案？不要求两种模态都必需。这里的“能回答”是证据支持，不是某次解题器碰巧答对。', '',
        '| 判定 | 题数 | 含义 |', '|---|---:|---|',
        f'| 通过 | {a["pass"]} | 有材料支持的答案 |',
        f'| 不通过 | {a["fail"]} | 已证明存在错误预设、缺少决定条件或题意范围缺陷 |',
        f'| 可回答性待裁定 | {a["unresolved"]} | 评审尚不能确定是否可回答 |', '',
        '## 二、双模态必要性', '',
        f'在上述{a["pass"]}道可回答题中，另问：移除任一种模态后，是否都无法确定完整答案？这里的“必要性待裁定”不表示问题不能回答。', '',
        '| 判定 | 题数 | 含义 |', '|---|---:|---|',
        f'| 确认图文都必需 | {n["both_required"]} | 文本必要与图像必要均已确认 |',
        f'| 确认不需要两者同时具备 | {n["not_both_required"]} | 至少一种模态可省略 |',
        f'| 必要性待裁定 | {n["unresolved"]} | 可回答，但至少一个方向的必要性证据尚未通过 |',
        f'| 不进入必要性裁定 | {n["not_applicable"]} | 可回答性未通过或未确认；不赋相反必要性标签 |', '',
        '## 三、逐题可回答性与参考答案', '',
        '| 题号 | 原问题 | 已核验参考答案 | 可回答性 | 材料依据或缺陷 |', '|---|---|---|---|---|']
    for row, item in zip(rows, assessment['items']):
        status = {'pass': '通过', 'fail': '不通过', 'unresolved': '可回答性待裁定'}[item['answerability']]
        ref = row['reference']['verified_answer'] if row['reference']['status'] == 'verified' else '无已核验参考；原因见右栏'
        out.append('| '+' | '.join(map(escape, [item['item_id'], row['question']['normalized_text'],
            ref, status, row.get('api_validity_review', {}).get('reason', row['faithfulness']['review']['reason'])]))+' |')
    out += ['', '## 四、逐题双模态必要性与目标验收', '',
            '“文本必要”指不能省去文本；“图像必要”指不能省去图像。两项分别记录，失败证明不自动翻成相反标签。', '',
            '| 题号 | 文本必要 | 图像必要 | 双模态必要性 | 目标验收 |', '|---|---|---|---|---|']
    for item in assessment['items']:
        show = lambda v: '是' if v == 1 else '否' if v == 0 else '该方向待裁定'
        ns = {'both_required': '两者都必需', 'not_both_required': '至少一种可省略',
              'unresolved': '必要性待裁定', 'not_applicable': '不适用（可回答性未通过/未确认）'}[item['modality_necessity']]
        ds = [show(item[k]) if item['modality_necessity'] != 'not_applicable' else '不适用'
              for k in ('text_necessary', 'image_necessary')]
        ts = {'pass': '达标', 'fail': '不达标', 'unresolved': '待裁定'}[item['target_acceptance']]
        out.append('| '+' | '.join([item['item_id'], *ds, ns, ts])+' |')
    out += ['', '## 五、可回答但必要性待裁定的具体原因', '',
            '| 题号 | 已核验答案 | 未通过的必要性证据 |', '|---|---|---|']
    for row, item in zip(rows, assessment['items']):
        if item['modality_necessity'] != 'unresolved':
            continue
        reasons = [f'{name}：{row["labels"]["adjudicated"][key]["reason"]}'
                   for name, key in [('文本方向', 'text_necessary'), ('图像方向', 'image_necessary')]
                   if item[key] is None]
        out.append('| '+' | '.join(map(escape, [item['item_id'], row['reference']['verified_answer'], '\n'.join(reasons)]))+' |')
    out += ['', '## 来源与边界', '',
        analysis_note or f'逐题复用{source_run}的原Q/T/I、作答、参考及已记录评审；仅重排展示与逐题标签计数，没有新模型调用或新增裁决。证据及具名复核见reviews.jsonl和records.jsonl，冻结知识K及来源哈希见manifest.json。', '',
        'F落实既有V判据，不新增主指标；必要性定义、固定分母与BDR公式均未改变。实际解题器作答及BDR另见metrics.json，作答/评分未决与这里的必要性待裁定分别统计。', '',
        '此次为已见旧结果的Codex具名AI复评，不是独立人类金标准。0确认达标不等于真实达标数必为0；没有证据把必要性待裁定题当成不能回答。结果只描述这批baseline，不能外推到全test、其他模型或所提生成方法效果。本次未启动扩量或新的生成/答题。', '']
    return '\n'.join(out)


def review_binding(record):
    """Bind review to Q, exact T/I and the current reference; no source labels."""
    m = record['material']
    return {'question_sha256': record['question']['sha256'],
            'material_sha256': sha256_text(canonical_json({
                'text': m['text'], 'image_sha256': m['image_sha256']})),
            'reference_sha256': sha256_text(canonical_json({k: record['reference'].get(k)
                for k in ('candidate_answer', 'verified_answer', 'acceptable_answers', 'rubric')}))}


def _decision(item):
    if not isinstance(item, dict) or 'value' not in item or item.get('value') not in (0, 1, None) or (
            item.get('value') is not None and type(item['value']) is not int):
        raise ValueError('review decision must be 0/1/null')
    if not isinstance(item.get('reason'), str) or not item['reason'].strip():
        raise ValueError('review decision needs a reason')
    return item['value']


def question_gate(review):
    if review.get('assessment_status') != 'reviewed':
        return None
    checks = review.get('question_checks')
    if not isinstance(checks, dict) or set(checks) != set(CHECKS):
        raise ValueError('all three question faithfulness checks required')
    return _three_and(*(_decision(checks[k]) for k in CHECKS))


def apply_review(record, review):
    """An invalid reference alone never invalidates a supported question."""
    result = copy.deepcopy(record)
    gate = question_gate(review)
    old = result['labels']['adjudicated']['validity']
    effective = _three_and(old['value'], gate)
    result['faithfulness'] = {'version': VERSION, 'review': review, 'question_result': gate,
                              'source_validity': copy.deepcopy(old), 'effective_validity': effective}
    if _decision(review['reference_supported']) == 0:
        result['faithfulness']['source_reference'] = copy.deepcopy(record['reference'])
        result['reference']['status'] = 'unresolved'
        result['reference']['faithfulness_reason'] = review['reference_supported']['reason']
    if effective != old['value']:
        result['labels']['adjudicated']['validity'] = {
            'value': effective, 'status': 'decided' if effective is not None else 'undecided',
            'reason': '原有效性与具名忠实性复核合取；' + review['reason'],
            'source': review['source'], 'reviewer': review['reviewer'],
            'review_id': stable_id('faithfulness', result['attempt_id'], canonical_json(review))}
    if effective == 0:
        for key in ('text_necessary', 'image_necessary', 'integration'):
            result['labels']['adjudicated'][key] = {
                'value': None, 'status': 'not_applicable', 'source': review['source'],
                'reviewer': review['reviewer'],
                'reason': '问题不支持完整作答/无效；必要性不适用，不由无效Q推相反D。'}
    return result


def derive_faithfulness(*, source_run, run_id, reviews_path, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve(); _validate_run_id(source_run); _validate_run_id(run_id)
    source = root/output_root/source_run; out = root/output_root/run_id
    if source.resolve() == out.resolve():
        raise ValueError('faithfulness review must use a separate run')
    rows = read_jsonl(source/'records.jsonl'); reviews = read_jsonl(root/reviews_path)
    by_id = {r['attempt_id']: r for r in reviews}
    if not rows or len(reviews) != len(rows) or len(by_id) != len(rows) or set(by_id) != {r['attempt_id'] for r in rows}:
        raise ValueError('faithfulness review must retain every unique planned slot')
    knowledge = json.loads((root/'configs/necessity_assessment_v2.json').read_text())['knowledge']
    for record in rows:
        if (record['question']['sha256'] != sha256_text(record['question']['normalized_text']) or
                record['material']['text_sha256'] != sha256_text(record['material']['text']) or
                record['material']['image_sha256'] != sha256_file(root/record['material']['image_path'])):
            raise ValueError('source question/image changed')
        r = by_id[record['attempt_id']]
        if any(r.get(k) != v for k, v in review_binding(record).items()):
            raise ValueError('stale question/material/reference review binding')
        if r.get('knowledge_sha256') != sha256_text(knowledge):
            raise ValueError('review knowledge binding changed')
        if r.get('source') != 'user_delegated_ai' or r.get('user_authorized') is not True or not r.get('authorization') or not r.get('reviewer'):
            raise ValueError('explicit delegated review provenance required')
        if r.get('assessment_status') not in ('reviewed', 'technical_failure', 'evaluation_uncertain'):
            raise ValueError('invalid review status')
        if not r.get('item_id'):
            raise ValueError('review item id required')
        gate = question_gate(r)
        _decision(r['reference_supported'])
        if not isinstance(r.get('reason'), str) or not r['reason'].strip():
            raise ValueError('review needs an explanation')
        evidence = r.get('evidence')
        if not isinstance(evidence, list) or any(not isinstance(e, dict) or
                e.get('source') not in ('Q', 'T', 'I', 'K') or not e.get('quote_or_location') or not e.get('explanation') for e in evidence):
            raise ValueError('invalid faithfulness evidence')
        if gate is not None and not evidence:
            raise ValueError('a completed verdict requires evidence')
        if not isinstance(r.get('failure_codes'), list) or not set(r['failure_codes']) <= CODES:
            raise ValueError('invalid failure category')
        if r['assessment_status'] != 'reviewed' and r['failure_codes']:
            raise ValueError('review failure is not a proven generation defect')
    protected = {str(f.relative_to(root)): sha256_file(f) for f in source.rglob('*')
                 if f.is_file() and not f.name.endswith('.lock')}
    template = root/'prompts/review-question-faithfulness-v1.txt'
    fingerprint = sha256_text(canonical_json({'source': protected, 'reviews': reviews,
        'version': VERSION, 'knowledge': knowledge, 'template_sha256': sha256_file(template)}))
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out/'.init.lock'):
        if (out/'manifest.json').exists():
            manifest = json.loads((out/'manifest.json').read_text())
            if manifest['fingerprint'] != fingerprint:
                raise ValueError('derived assessment changed; use a new run id')
            for file, digest in manifest['frozen_files'].items():
                if sha256_file(out/file) != digest:
                    raise ValueError('derived assessment artifact changed')
            return {'n_planned': len(rows), 'new_model_calls': 0, 'resumed': True}
        derived = []
        for row in rows:
            r = apply_review(row, by_id[row['attempt_id']])
            r.update(run_id=run_id, source_run_id=source_run, source_attempt_id=row['attempt_id'])
            derived.append(r)
        write_jsonl(out/'records.jsonl', derived); write_jsonl(out/'reviews.jsonl', reviews)
        write_json(out/'metrics.json', {'protocol_version': rows[0]['protocol_version'],
                   'assessment_variant': VERSION, 'n_planned': len(rows),
                   'adjudicated': compute_record_metrics(derived, source='adjudicated')})
        (out/'review-question-faithfulness-v1.txt').write_bytes(template.read_bytes())
        counts = {}
        for label, value in [('pass', 1), ('fail', 0), ('unresolved', None)]:
            counts[label] = sum(r['faithfulness']['question_result'] == value for r in derived)
        categories = {k: 0 for k in ('non_joint', 'joint', 'invalid', 'unresolved')}
        validity_unresolved = 0
        for row in derived:
            labels = row['labels']['adjudicated']
            v = labels['validity']['value']
            dt, di = (labels[k]['value'] if labels[k]['status'] == 'decided' else None
                      for k in ('text_necessary', 'image_necessary'))
            category = ('invalid' if v == 0 else 'unresolved' if v is None else
                        'non_joint' if 0 in (dt, di) else 'joint' if dt == di == 1 else 'unresolved')
            categories[category] += 1
            validity_unresolved += v is None
        summary = {'n_planned': len(rows), 'question_faithfulness': counts,
                   'category_counts': categories, 'validity_unresolved': validity_unresolved,
                   'valid_necessity_unresolved': categories['unresolved']-validity_unresolved,
                   'source_run_id': source_run, 'assessment_variant': VERSION,
                   'new_model_calls': 0,
                   'note': 'Faithfulness explains existing validity; no new headline metric, necessity definition or BDR formula. Topic relatedness remains separate. Candidate-reference failure alone never invalidates Q.'}
        assessment = separated_assessment(derived)
        summary.update(report_layout=REPORT_LAYOUT, **{k: assessment[k] for k in
                       ('answerability', 'modality_necessity', 'target_acceptance')})
        write_jsonl(out/'assessment_table.jsonl', assessment['items'])
        write_json(out/'results.json', summary)
        write_json(out/'usage.json', {'new_experimental_http_requests': 0, 'new_experimental_model_tokens': 0,
                   'reviewer': 'Codex user_delegated_ai; conversation review cost not metered here',
                   'source_usage_path': str((source/'usage.json').relative_to(root)),
                   'note': 'Existing model outputs reused. No GPU or new experimental model calls.'})
        (out/'probe_report.md').write_text(separated_report(derived, assessment, source_run))
        frozen = {f.name: sha256_file(f) for f in out.iterdir() if f.is_file() and not f.name.endswith('.lock')}
        write_json(out/'manifest.json', {'created_at': utc_now(), 'version': VERSION, 'source_run_id': source_run,
                   'n_planned': len(rows), 'fingerprint': fingerprint, 'source_files_sha256': protected,
                   'reviews_source_path': str(reviews_path),
                   'reviews_source_sha256': sha256_file(root/reviews_path),
                   'knowledge_sha256': sha256_text(knowledge), 'frozen_files': frozen,
                   'report_layout': REPORT_LAYOUT,
                   'protocol_policy': 'Existing necessity definitions, frozen protocols, source outputs and metrics unchanged. Offline derived validity review only.'})
    for file, digest in protected.items():
        if sha256_file(root/file) != digest:
            raise ValueError('source run changed during derived review')
    return {**summary, 'resumed': False}


def report_targeted_analysis(*,run_id,project_root='.',output_root='outputs'):
    """Merge newly verified, explicitly reviewed proofs; no failed proof flips D."""
    from .baseline_analysis import proof_shape
    from .evidence_verification import verification_gate
    from .experiments import label,call_usage
    root=Path(project_root).resolve();_validate_run_id(run_id);out=root/output_root/run_id
    manifest=json.loads((out/'manifest.json').read_text())
    for name,h in manifest['frozen_files'].items():
        if sha256_file(out/name)!=h:raise ValueError('frozen targeted input changed')
    parent=read_jsonl(out/'parent_records.jsonl');tasks=read_jsonl(out/'records.jsonl')
    reviews=read_jsonl(out/'evidence_reviews.jsonl')
    analyses={(r['item_id'],r['removed']):r for r in tasks if r['stage']=='analysis'}
    verifiers={(r['item_id'],r['removed']):r for r in tasks if r['stage']=='verification'}
    indexed={(r['item_id'],r['removed']):r for r in reviews}
    if set(indexed)!=set(analyses) or len(indexed)!=len(reviews):raise ValueError('complete unique targeted proof reviews required')
    for ident,r in indexed.items():
        if r.get('analysis_sha256')!=sha256_text(canonical_json(analyses[ident])) or r.get('verification_sha256')!=sha256_text(canonical_json(verifiers[ident])):
            raise ValueError('targeted proof review binding changed')
        if r.get('source')!='user_delegated_ai' or not r.get('reviewer') or not r.get('reason') or r.get('result') not in ('supported','uncertain','not_supported'):
            raise ValueError('explicit targeted review provenance required')
    derived=copy.deepcopy(parent);conflicts=[]
    for row in derived:
        item=row['faithfulness']['review']['item_id'];labs=row['labels']['adjudicated'];row['prior_adjudicated']=copy.deepcopy(labs)
        row['run_id']=run_id
        if labs['validity']['value']!=1:continue
        for removed,key in (('T','text_necessary'),('I','image_necessary')):
            ident=(item,removed)
            if ident not in analyses:continue
            a=analyses[ident];v=verifiers[ident];r=indexed[ident];proof=a.get('output')
            accept=(a['status']=='ok' and proof_shape(proof) and v['status']=='ok'
                    and verification_gate(v['output'],strict=True)=='accept' and r['result']=='supported')
            if not accept:continue
            new=0 if proof['kind']=='unique' else 1
            old=labs[key]['value'] if labs[key]['status']=='decided' else None
            if old is not None and old!=new:
                conflicts.append({'item_id':item,'removed':removed,'old':old,'new':new})
                labs[key]=label(None,'Conflicting accepted proofs; requires semantic adjudication.',source='user_delegated_ai')
            else:
                labs[key]=label(new,r['reason'],source='user_delegated_ai');labs[key]['reviewer']=r['reviewer']
        if 0 in (labs['text_necessary']['value'],labs['image_necessary']['value']):
            labs['integration']=label(0,'At least one modality has a sufficient solution.',source='user_delegated_ai')
    disagreements=[];faithfulness_audit=[]
    for task in (r for r in tasks if r['stage']=='faithfulness'):
        row=next(r for r in parent if r['faithfulness']['review']['item_id']==task['item_id'])
        old=row['labels']['adjudicated']['validity']['value'];new=None
        try:new=_three_and(*(_decision(task['output']['question_checks'][k]) for k in CHECKS))
        except (KeyError,TypeError,ValueError):pass
        audit={'item_id':task['item_id'],'prior_validity':old,'api_validity':new,'api_reason':(task.get('output') or {}).get('reason')}
        faithfulness_audit.append(audit)
        if new is not None and old!=new:disagreements.append(audit)
    validity_reviews=read_jsonl(out/'validity_reviews.jsonl')
    seen=set()
    for review in validity_reviews:
        ident=review['item_id']
        task=next((t for t in tasks if t['stage']=='faithfulness' and t['item_id']==ident),None)
        if ident in seen or not task or review.get('api_record_sha256')!=sha256_text(canonical_json(task)):
            raise ValueError('validity API review binding changed')
        seen.add(ident)
        row=next(r for r in derived if r['faithfulness']['review']['item_id']==ident)
        if (review.get('question_sha256')!=row['question']['sha256'] or
                review.get('knowledge_sha256')!=sha256_text(manifest['config']['knowledge']) or
                review.get('source')!='user_delegated_ai' or not review.get('reviewer')):
            raise ValueError('validity API review provenance changed')
        new=_decision(review)
        row['api_validity_review']=review
        row['labels']['adjudicated']['validity']=label(new,review['reason'],source='user_delegated_ai')
        row['labels']['adjudicated']['validity']['reviewer']=review['reviewer']
        if new is None:
            for k in ('text_necessary','image_necessary','integration'):
                row['labels']['adjudicated'][k]=label(None,'Validity assumption boundary pending; no necessity decision.',source='user_delegated_ai')
    assessment=separated_assessment(derived)
    write_jsonl(out/'assessed_questions.jsonl',derived);write_jsonl(out/'assessment_table.jsonl',assessment['items'])
    write_jsonl(out/'validity_api_audit.jsonl',faithfulness_audit)
    write_json(out/'assessed_metrics.json',{'source':'adjudicated; explicit V reviews and new reviewed D merged',
                'n_planned':len(derived),'adjudicated':compute_record_metrics(derived,source='adjudicated')})
    usage=call_usage(read_jsonl(out/'calls.jsonl'))
    result={'n_planned':len(derived),'answerability':assessment['answerability'],
            'modality_necessity':assessment['modality_necessity'],'target_acceptance':assessment['target_acceptance'],
            'validity_api_disagreements':disagreements,'conflicting_proofs':conflicts,'usage':usage}
    write_json(out/'assessed_results.json',result)
    note='原Q、实际T/I、参考和作答未改变；本轮新增有上限的API复评及具名证明复核。原始API输出在calls.jsonl，12方向复核在evidence_reviews.jsonl，30题派生联表在assessed_questions.jsonl。有效性API复评只作为分歧审计；V的变更必须有绑定原Q/K/API输出的validity_reviews.jsonl具名复核，不以同模型投票自动覆盖。'
    report=separated_report(derived,assessment,manifest['source_run'],analysis_note=note)
    report+='\n## 本轮API分析及判定边界\n\n'
    report+=f'计划{manifest["n_planned_calls"]}个调用名额；实际HTTP {usage["remote_requests"]}，tokens {usage["total_tokens"]}，费用未知。无完整证明的核验名额明确跳过，失败原样保留。\n\n'
    report+=f'对原{len(faithfulness_audit)}道可答性不通过题复评，API与原整体判定有{len(disagreements)}题分歧。下表保留原V作对照；上面的当前V采用具名边界复核，不把模型分歧直接改成通过。\n\n'
    report+='| 题号 | 原V | API可答性判断 | API依据 |\n|---|---:|---:|---|\n'
    for r in faithfulness_audit:report+='| '+' | '.join(str(r[k]).replace('|','\\|') for k in ('item_id','prior_validity','api_validity','api_reason'))+' |\n'
    report+='\n气体题的分歧在于热平衡/速率分布是否可作为教材默认条件：API按该默认条件作答，原复核要求材料给出。冻结K不变，本轮不私自选择更有利解释。平均速率本身不确定二阶矩，这是需要明确报告的模型假设边界。\n'
    report+='\n## 每份新证明的核验与复核\n\n| 题号 | 移除模态 | API证明形式 | API核验 | 具名复核 | 依据 |\n|---|---|---|---|---|---|\n'
    for ident,r in indexed.items():
        a=analyses[ident];v=verifiers[ident]
        values=[r['item_id'],r['removed'],(a.get('output') or {}).get('kind'),
                v.get('gate',v['status']),r['result'],r['reason']]
        report+='| '+' | '.join(str(x).replace('|','\\|') for x in values)+' |\n'
    (out/'probe_report.md').write_text(report)
    return result
