"""Fixed random material screening and explicitly conditional yield estimates."""
from __future__ import annotations

import base64
import html
import json
import math
from collections import Counter
from pathlib import Path
from statistics import NormalDist

from .dataset import REVISION, import_scienceqa
from .pipeline import RunConfigMismatch, _validate_run_id, load_materials
from .prompts import PromptSet
from .providers import CallRecorder, CompletionRequest, JudgeProvider
from .quality import _call
from .records import JsonlTable, canonical_json, read_jsonl, sha256_text, utc_now, write_json, write_jsonl

PROTOCOL = 'material-yield-screen-v1'
CHECKS = ('readable', 'question_valid', 'text_needed', 'image_needed', 'integrated', 'answer_supported', 'educational')
CODES = ('none', 'text_redundant', 'image_redundant', 'no_joint_goal', 'missing_information', 'unreadable', 'uncertain')
INSTRUCTIONS = {
    'screen_material': 'Find one best natural integrated witness if possible. If none is justified, state the concrete limitation. Do not force a positive example.',
    'audit_material': 'Audit the proposed witness adversarially, without knowing the first decision. Look for a single-source shortcut, hidden assumptions, wrong answers and concatenated questions. If the proposal is missing or flawed, independently consider a natural alternative. Only approve with a complete supported witness; otherwise give the material limitation or uncertainty.',
}


def wilson_interval(successes: int, n: int) -> list[float]:
    if n < 1 or not 0 <= successes <= n:
        raise ValueError('invalid Wilson counts')
    z = NormalDist().inv_cdf(.975)
    p = successes/n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [max(0., center-radius), min(1., center+radius)]


def combine_screen(first: dict, audit: dict) -> str:
    if first.get('status') == audit.get('status') == 'suitable':
        return 'pass'
    if first.get('status') == audit.get('status') == 'unsuitable':
        return 'fail'
    return 'unresolved'


def validate_screen(data: dict) -> dict:
    if data.get('status') not in ('suitable', 'unsuitable', 'unresolved'):
        raise ValueError('invalid screen status')
    if not isinstance(data.get('reason'), str) or not data['reason'].strip() or data.get('exclusion_code') not in CODES:
        raise ValueError('missing screen reason/code')
    checks = data.get('checks')
    if not isinstance(checks, dict) or any(k not in checks or (checks[k] is not None and type(checks[k]) is not bool) for k in CHECKS):
        raise ValueError('checks must all be booleans/null')
    witness = data.get('witness')
    if witness is not None and (not isinstance(witness, dict) or not all(
        isinstance(witness.get(k), str) and witness[k].strip()
        for k in ('question', 'answer', 'rubric', 'text_evidence', 'image_evidence'))):
        raise ValueError('incomplete witness')
    if data['status'] == 'suitable' and (witness is None or not all(checks[k] is True for k in CHECKS)):
        raise ValueError('suitable requires a complete witness and all checks true')
    if data['status'] == 'unsuitable' and not any(checks[k] is False for k in CHECKS):
        raise ValueError('unsuitable requires a concrete failed check')
    return data


def estimate_yield(results: list[dict], population: int, split_populations: dict[str, int]) -> dict:
    counts = Counter(r['decision'] for r in results)
    if set(counts) - {'pass', 'fail', 'unresolved'} or not results or len(results) > population:
        raise ValueError('invalid screening population/decisions')
    n = len(results); accepted = counts['pass']; unknown = counts['unresolved']
    lower, upper = accepted/n, (accepted+unknown)/n
    strict_interval = wilson_interval(accepted, n)
    possible_interval = wilson_interval(accepted+unknown, n)
    if n == population:  # A census has no sampling error; unresolved labels still have bounds.
        strict_interval, possible_interval = [lower,lower], [upper,upper]
    envelope = [strict_interval[0], possible_interval[1]]
    def project(size):
        return {'population_rows': size, 'point_strict_rows': round(size*lower),
                'point_if_all_unresolved_pass_rows': round(size*upper),
                'wilson95_strict_rows': [math.floor(size*strict_interval[0]), math.ceil(size*strict_interval[1])],
                'sampling_and_unresolved_envelope_rows': [math.floor(size*envelope[0]), math.ceil(size*envelope[1])]}
    return {'n_sampled': n, 'n_pass': accepted, 'n_fail': counts['fail'], 'n_unresolved': unknown,
            'strict_automatic_acceptance_rate': lower, 'unresolved_acceptance_bounds': [lower,upper],
            'wilson95_strict_acceptance': strict_interval,
            'wilson95_if_all_unresolved_pass': possible_interval,
            'sampling_and_unresolved_envelope': envelope,
            'sample_fraction': n/population,
            'training_projection': project(population),
            'other_split_scenarios_same_retention_assumption': {s:project(size) for s,size in split_populations.items()},
            'whole_dataset_scenario_same_retention_assumption': project(sum(split_populations.values())),
            'scope': 'Fixed automatic two-pass screen of candidate ROWS, before deduplication/human review; not intrinsic capacity or method question yield.',
            'interval_scope': 'Approximate 95% Wilson sampling intervals for non-census samples; no finite-population correction. Census intervals collapse. No allowance for model misclassification. Other splits require an untested equal-retention assumption.',
            'interval_source': 'https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm'}


def estimate_reviewed_yield(results: list[dict], reviews: list[dict], population: int,
                           split_populations: dict[str, int]) -> dict:
    ids = {r['material_id'] for r in results}
    if len(reviews) != len(ids) or {r.get('material_id') for r in reviews} != ids:
        raise ValueError('reviews must cover every sampled material exactly once')
    for review in reviews:
        if (review.get('review_source') != 'codex_visual_review_not_human_gold'
            or review.get('human_review_status') != 'pending' or not review.get('reason')):
            raise ValueError('reviews require explicit AI provenance, pending human status and reasons')
        if review.get('decision') == 'pass' and (not isinstance(review.get('witness'), dict)
                or not all(isinstance(review['witness'].get(k), str) and review['witness'][k].strip()
                           for k in ('question', 'answer'))):
            raise ValueError('a review pass requires a concrete witness question and answer')
    stats = estimate_yield(reviews, population, split_populations)
    original = {r['material_id']:r['decision'] for r in results}
    stats.update(review_source='Codex visual review, non-blind, not human gold',
                 review_sha256=sha256_text(canonical_json(reviews)),
                 scope='Provisional AI visual re-review of all sampled rows under the existing criteria, before deduplication/human review.',
                 n_changed_from_endpoint=sum(r['decision'] != original[r['material_id']] for r in reviews),
                 original_endpoint_pass=sum(r['decision']=='pass' for r in results))
    return stats


def screen_sample(*, run_id: str, project_root: str | Path = '.', output_root: str | Path = 'outputs',
                  limit: int = 30, seed: int = 20261111, max_requests: int = 120,
                  max_tokens: int = 2048, dry_run: bool = False, provider=None,
                  reviews_path: str | None = None) -> dict:
    _validate_run_id(run_id)
    root = Path(project_root).resolve(); out = root/output_root/run_id
    # Importer's seeded shuffle is simple random sampling without replacement.
    imported = import_scienceqa(root, split='train', limit=limit, seed=seed, offline=True)
    material_path = Path(imported['materials'])
    materials = load_materials(root, material_path)
    manifest = json.loads(material_path.with_name('manifest.json').read_text())
    if len(materials) != limit:
        raise ValueError('requested sample exceeds available candidate pool')
    raw = json.loads((root/'data/raw/scienceqa'/REVISION/'problems.json').read_text())
    pools = {split:[pid for pid,p in raw.items() if p['split']==split and p.get('image')
                   and isinstance(p.get('hint'),str) and p['hint'].strip()] for split in ('train','val','test')}
    selected_ids = [m['source_id'] for m in materials]
    sampling = {'design': 'simple_random_without_replacement', 'seed': seed, 'n': len(materials),
                'source_split':'train', 'population':len(pools['train']),
                'frame_ids_sha256':sha256_text(canonical_json(sorted(pools['train']))),
                'selected_ids':selected_ids, 'import_manifest':manifest,
                'sample_subjects':dict(Counter(raw[i]['subject'] for i in selected_ids)),
                'population_subjects':dict(Counter(raw[i]['subject'] for i in pools['train'])),
                'sample_unique_images':len({m['image_sha256'] for m in materials}),
                'sample_unique_text_image_pairs':len({(m['text'],m['image_sha256']) for m in materials}),
                'rule':'No replacement based on model results. All sampled rows stay in denominator.'}
    provider = provider or JudgeProvider.from_env(root=root,run_dir=out,max_requests=max_requests,max_tokens=max_tokens,visual=True)
    prompts = PromptSet(root/'prompts',names=('screen_material',))
    config = {'protocol_version':PROTOCOL,'sampling':sampling,'materials':materials,
              'prompts':prompts.snapshot(),'instructions':INSTRUCTIONS,'provider':provider.identity,
              'role_reuse':'Same configured model, separate proposal and skeptical audit; not independent human labels.'}
    estimate = {'n_sampled':len(materials),'population':len(pools['train']),
                'logical_calls_without_retries':len(materials)*2,'max_http_attempts':max_requests,
                'max_output_tokens_per_request':max_tokens,'estimated_cost':None,'sampling':sampling['design']}
    if dry_run: return estimate
    fingerprint = sha256_text(canonical_json(config));state_path=out/'run.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['config_sha256']!=fingerprint:
            raise RunConfigMismatch('sampling/material/prompt/model changed; use a new run_id')
    else:
        if out.exists() and any(out.iterdir()):raise RunConfigMismatch('output directory not empty')
        state={'run_id':run_id,'config_sha256':fingerprint,'created_at':utc_now()}
    state.update(status='running',updated_at=utc_now(),max_requests=max_requests)
    write_json(state_path,state);write_json(out/'config.json',config);write_json(out/'sampling.json',sampling)
    write_jsonl(out/'materials.jsonl',materials)
    tables={name:JsonlTable(out/(name+'.jsonl'),key) for name,key in (
        ('calls','call_id'),('screens','material_id'),('audits','material_id'),('results','material_id'))}
    recorder=CallRecorder(tables['calls'],run_id=run_id,protocol_version=PROTOCOL,default_model=provider.model)
    for material in materials:
        mid=material['material_id']
        if tables['results'].get(mid) is not None:continue
        stages=[]
        for task,table_name in (('screen_material','screens'),('audit_material','audits')):
            row=tables[table_name].get(mid)
            if row is None:
                witness=stages[0].get('witness') if stages else None
                # Only the proposed question/answer/rubric is shared. No initial decision or rationale.
                candidate={k:witness[k] for k in ('question','answer','rubric')} if witness else None
                prompt=prompts.render('screen_material',text=material['text'],instruction=INSTRUCTIONS[task],
                                      candidate=json.dumps(candidate,ensure_ascii=False))
                data,error=_call(provider,recorder,CompletionRequest(task=task,prompt=prompt,
                    text=material['text'],image_path=str(root/material['image_path']),metadata={'protocol_version':PROTOCOL}),
                    role=task,material_id=mid)
                try:
                    if error:raise ValueError(error)
                    validate_screen(data)
                except ValueError as exc:
                    data={'status':'unresolved','reason':str(exc),'exclusion_code':'uncertain',
                          'checks':{k:None for k in CHECKS},'witness':None}
                row={'material_id':mid,**data,'source':'automatic_model','human_review_status':'pending'}
                tables[table_name].add(row)
            stages.append(row)
        tables['results'].add({'material_id':mid,'source_id':material['source_id'],
            'decision':combine_screen(*stages),'screen_status':stages[0]['status'],'audit_status':stages[1]['status'],
            'screen_reason':stages[0]['reason'],'audit_reason':stages[1]['reason'],
            'exclusion_code':stages[1]['exclusion_code'],'human_review_status':'pending'})
    results=tables['results'].values()
    stats=estimate_yield(results,len(pools['train']),{k:len(v) for k,v in pools.items()})
    stats['exclusion_codes']=dict(Counter(r['exclusion_code'] for r in results if r['decision']=='fail'))
    stats['sample_subjects']=sampling['sample_subjects'];stats['sample_unique_images']=sampling['sample_unique_images']
    ledger=read_jsonl(out/'requests.jsonl');started=sum(r['event']=='started' for r in ledger)
    usage=[r['usage'] for r in ledger if r['event']=='completed']
    accounting={'http_attempts':started,'completed_requests':len(usage),'requests_without_usage':started-len(usage),
        'known_total_tokens':sum(u.get('total_tokens') or 0 for u in usage),
        'total_tokens':sum(u['total_tokens'] for u in usage) if len(usage)==started and all(u.get('total_tokens') is not None for u in usage) else None,
        'reported_cost':sum(u['cost'] for u in usage) if len(usage)==started and all(u.get('cost') is not None for u in usage) else None,
        'cache_hits':sum(r.get('usage',{}).get('cache_hit',False) for r in tables['calls'].values())}
    write_json(out/'estimate.json',stats);write_json(out/'usage.json',accounting)
    reviews, reviewed = None, None
    if reviews_path is not None:
        reviews=read_jsonl(root/reviews_path)
        material_hashes={m['material_id']:m['image_sha256'] for m in materials}
        if any(r.get('image_sha256')!=material_hashes.get(r.get('material_id')) for r in reviews):
            raise ValueError('review image hash does not match sampled material')
        reviewed=estimate_reviewed_yield(results,reviews,len(pools['train']),{k:len(v) for k,v in pools.items()})
        review_output=out/'reviewed_estimate.json'
        if review_output.exists() and json.loads(review_output.read_text())['review_sha256']!=reviewed['review_sha256']:
            raise RunConfigMismatch('review annotations changed; preserve this review and use a new run_id')
        write_json(review_output,reviewed)
    _report(out,root,materials,tables,stats,sampling,accounting,reviews,reviewed)
    state.update(status='complete',updated_at=utc_now());write_json(state_path,state)
    return {'run_dir':str(out),'new_calls':recorder.added_count,'new_records':sum(t.added_count for t in tables.values()),
            'estimate':stats,'reviewed_estimate':reviewed,'usage':accounting}


def _report(out,root,materials,tables,stats,sampling,accounting,reviews=None,reviewed=None):
    n=stats['n_sampled'];lo,hi=stats['unresolved_acceptance_bounds'];ci=stats['wilson95_strict_acceptance']
    train=stats['training_projection']
    lines=[f'# {n}组随机材料的自动筛选产出估计','',
           f"简单随机、不放回；seed={sampling['seed']}；训练候选{sampling['population']}组，实际抽取{n}组。",
           '原hint与图像不变；一次见证搜索+一次质疑审核；同一模型，未作人工核验、未完成去重。','',
           f"- 自动通过：{stats['n_pass']}/{n}（{lo:.1%}）；自动未通过：{stats['n_fail']}；待审：{stats['n_unresolved']}。",
           f"- 将待审分别视为不通过/通过的保留率边界：{lo:.1%}–{hi:.1%}，不是置信区间。",
           f"- 严格自动通过率的近似95% Wilson抽样区间：{ci[0]:.1%}–{ci[1]:.1%}。区间不含模型误判，也不是人工合格率区间。",
           f"- 训练候选估计：{train['point_strict_rows']}组；若待审全通过则{train['point_if_all_unresolved_pass_rows']}组。严格抽样区间折算{train['wilson95_strict_rows']}组；待审与抽样误差合并范围{train['sampling_and_unresolved_envelope_rows']}组。",'',
           '| 分区 | 候选数 | 按严格通过率折算 | 若待审全通过 |','|---|---:|---:|---:|']
    for name,p in stats['other_split_scenarios_same_retention_assumption'].items():
        lines.append(f"| {name} | {p['population_rows']} | {p['point_strict_rows']} | {p['point_if_all_unresolved_pass_rows']} |")
    total=stats['whole_dataset_scenario_same_retention_assumption']
    lines += [f"| 合计 | {total['population_rows']} | {total['point_strict_rows']} | {total['point_if_all_unresolved_pass_rows']} |",'',
              '**val/test/合计仅是假设各分区保留率相同的情景外推，尚未抽样验证。上表都是去重前材料行数。**',
              f"样本学科组成：{sampling['sample_subjects']}；图像哈希去重后{sampling['sample_unique_images']}张，文本+图像精确组合{sampling['sample_unique_text_image_pairs']}组。",'',
              '未通过只表示固定搜索/审核预算下没有获得双重认可的见证，不证明材料永远不能出好题；模型的共同偏差不会被抽样区间修复。','',
              f"调用审计：{accounting['http_attempts']}次尝试，{accounting['completed_requests']}次完成；已知{accounting['known_total_tokens']} tokens，费用未返回则为null。",'',
              '区间方法：[NIST Wilson interval](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm)。抽样比例很小，使用二项近似且未作有限总体修正。','',
              '| 材料 | 初筛 | 复核 | 最终自动状态 | 排除类别 |','|---|---|---|---|---|']
    for r in tables['results'].values():lines.append(f"| {r['material_id']} | {r['screen_status']} | {r['audit_status']} | {r['decision']} | {r['exclusion_code']} |")
    if reviewed is not None:
        rp=reviewed['training_projection'];ci=reviewed['wilson95_strict_acceptance']
        lead=[f'# {n}组随机材料：AI逐图复核后的暂定估计','',
              f"模型两轮初筛通过{stats['n_pass']}组；逐图AI复核后，通过{reviewed['n_pass']}组、未通过{reviewed['n_fail']}组、待审{reviewed['n_unresolved']}组。原模型判断和原估计保留在estimate.json。",
              '**本次复核由Codex完成，非盲评、非人工金标准；未重新抽样、未改变原图文或筛选定义。不能将其称为人工确认有效率。**','',
              f"保守通过率{reviewed['strict_automatic_acceptance_rate']:.1%}；若待审全部通过则{reviewed['unresolved_acceptance_bounds'][1]:.1%}。严格通过率近似95%抽样区间{ci[0]:.1%}–{ci[1]:.1%}，不包括AI判断错误。",
              f"训练候选的保守点估计为{rp['point_strict_rows']}组；严格抽样区间折算{rp['wilson95_strict_rows']}组。若待审全通过点估计为{rp['point_if_all_unresolved_pass_rows']}组；合并待审与抽样不确定性的范围为{rp['sampling_and_unresolved_envelope_rows']}组。",'',
              '| 范围 | 候选数 | 保守点估计 | 待审全通过情景 |','|---|---:|---:|---:|']
        for name,r in reviewed['other_split_scenarios_same_retention_assumption'].items():
            lead.append(f"| {name} | {r['population_rows']} | {r['point_strict_rows']} | {r['point_if_all_unresolved_pass_rows']} |")
        r=reviewed['whole_dataset_scenario_same_retention_assumption']
        lead += [f"| 合计 | {r['population_rows']} | {r['point_strict_rows']} | {r['point_if_all_unresolved_pass_rows']} |",'',
                 'val/test/合计是假设与训练池保留率相同的情景，不是这些分区的实测估计；所有数量均为去重前。',
                 '复核保留的见证包括符号映射+绝对粒子数、性状映射+杂交格、文字中的冷却时长+图中温差。磁铁比较、残缺动物文本与喷发类型判断等争议保留待审。',
                 '模型的长颈鹿体温题把普通生物知识当作必要文字，说明两轮同模型同意不能替代证据核验。电池材料可用，但模型原题把5分钟写入题干，复核改为不泄露时长的平均降温速率问题。', '',
                 '## 原始模型两轮筛查（保留作对照，不直接当有效样本数）','']
        lines=lead+lines[2:]
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    esc=lambda x:html.escape(str(x))
    parts=['<!doctype html><meta charset="utf-8"><title>Random material screen</title>',
           '<style>body{font:17px/1.6 sans-serif;max-width:1050px;margin:30px auto;padding:0 16px}img{max-width:100%;max-height:380px}article{border-top:2px solid #aaa;padding:20px 0}pre{white-space:pre-wrap}</style>',
           '<h1>随机材料筛选：逐组审阅</h1><p>原图文未改；同模型两轮自动判断，全部待人工核验。没有删除失败或待审项。</p>']
    for m in materials:
        image=root/m['image_path'];data=image.read_bytes();mime='image/png' if data.startswith(b'\x89PNG') else 'image/jpeg'
        parts += [f'<article><h2>{esc(m["material_id"])}</h2><p>{esc(m["text"])}</p>',
                  '<img alt="Original image" src="data:'+mime+';base64,'+base64.b64encode(data).decode()+'">',
                  '<details><summary>查看自动判断、见证问题与依据</summary>']
        for name in ('screens','audits','results'):
            parts.append('<h3>'+esc(name)+'</h3><pre>'+esc(json.dumps(tables[name].get(m['material_id']),ensure_ascii=False,indent=2))+'</pre>')
        if reviews is not None:
            review=next(r for r in reviews if r['material_id']==m['material_id'])
            parts.append('<h3>AI逐图复核（非人工金标准）</h3><pre>'+esc(json.dumps(review,ensure_ascii=False,indent=2))+'</pre>')
        parts.append('</details></article>')
    (out/'review.html').write_text('\n'.join(parts))


ADMISSION_CHECKS = ('hint_present_original','readable','source_traceable','text_substantive',
                    'image_unique','joint_goal','no_extra_facts','no_answer_leakage',
                    'witness_isolated','development_isolated','source_isolated','pre_generation')


def prepare_admission(*, run_id, seed=20261112, limit=60, project_root='.', output_root='outputs'):
    """Random unseen train materials, no generation or model output is consulted."""
    import hashlib
    import random
    import zipfile
    from .records import sha256_file
    root = Path(project_root).resolve(); _validate_run_id(run_id); out = root / output_root / run_id
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved.get('seed') != seed or saved.get('initial_n_sampled',saved.get('n_sampled')) != limit:
            raise ValueError('admission sample changed; use a new run_id')
        if saved['candidate_sha256'] != sha256_file(out / 'candidates.jsonl'):
            raise ValueError('admission candidates changed')
        return {'run_dir': str(out), 'n_sampled': saved['n_sampled'], 'resumed': True, 'status': saved['status']}
    raw = root / 'data/raw/scienceqa' / REVISION
    problems = json.loads((raw / 'problems.json').read_text())
    splits = json.loads((raw / 'pid_splits.json').read_text())
    seen_ids, seen_hashes, exclusion_files = set(), set(), []
    # Read provenance only, never questions, labels or model results. All previously
    # imported material is conservatively treated as development exposure.
    paths = sorted(set((root / 'data/processed/scienceqa').glob('**/materials.jsonl')) |
                   set((root / 'outputs').glob('*/materials.jsonl')))
    for path in paths:
        rows = read_jsonl(path)
        for row in rows:
            if row.get('source_id'): seen_ids.add(str(row['source_id']))
            if row.get('material_id', '').startswith('scienceqa_'):
                seen_ids.add(row['material_id'].removeprefix('scienceqa_'))
            if row.get('image_sha256'): seen_hashes.add(row['image_sha256'])
        exclusion_files.append({'path': str(path.relative_to(root)), 'sha256': sha256_file(path)})
    eligible, exclusions = [], []
    with zipfile.ZipFile(raw / 'train.zip') as images:
        for pid in map(str, splits['train']):
            p = problems[pid]
            if not isinstance(p.get('hint'), str) or not p['hint'].strip() or not p.get('image'):
                continue
            if pid in seen_ids:
                exclusions.append({'source_id': pid, 'reason': 'previous_development_source'}); continue
            filename = p['image']
            if Path(filename).name != filename:
                raise ValueError('unsafe raw image filename')
            digest = hashlib.sha256(images.read(f'train/{pid}/{filename}')).hexdigest()
            if digest in seen_hashes:
                exclusions.append({'source_id': pid, 'reason': 'shared_image_with_development'}); continue
            eligible.append(pid)
    if limit < 1 or len(eligible) < limit:
        raise ValueError('not enough unseen candidates')
    ordered = list(eligible); random.Random(seed).shuffle(ordered); selected = ordered[:limit]
    imported = import_scienceqa(root, split='train', limit=limit, source_ids=selected, offline=True)
    indexed = {m['source_id']: m for m in load_materials(root, Path(imported['materials']))}
    materials, selections, review = [], [], []
    hashes = set()
    for index, pid in enumerate(selected):
        m = indexed[pid]; m['text'] = problems[pid]['hint']
        m.update(experiment_partition='candidate', review_status='pending', processing_version='admission-v1')
        duplicate = m['image_sha256'] in hashes; hashes.add(m['image_sha256'])
        checks = {k: {'value': None, 'reason': None} for k in ADMISSION_CHECKS}
        if duplicate:
            checks['source_isolated'] = {'value': 0, 'reason': 'within_sample_shared_image'}
        materials.append(m)
        selections.append({'material_id': m['material_id'], 'source_id': pid, 'sample_index': index,
            'image_sha256': m['image_sha256'], 'decision': 'pending', 'source': None,
            'checks': checks, 'reason': None, 'reviewer': None})
        review.append(copy_json(selections[-1]))
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'candidates.jsonl', materials); write_jsonl(out / 'selection.jsonl', selections)
    write_jsonl(out / 'admission_template.jsonl', review)
    write_json(out / 'manifest.json', {'protocol_version': 'three-phase-prompt-v1', 'seed': seed,
        'n_sampled': limit, 'dataset_revision': REVISION, 'status': 'pending_human_admission',
        'candidate_sha256': sha256_file(out / 'candidates.jsonl'), 'sampled_ids': selected,
        'sampling_frame_sha256': sha256_text(canonical_json(eligible)), 'frame_size': len(eligible),
        'excluded_development_sources': exclusions, 'exclusion_provenance': exclusion_files,
        'methods': ['plain', 'strong', 'anchored', 'no_image_anchor'], 'target_n_materials': 30})
    fragments = ['<!doctype html><meta charset="utf-8"><title>Material admission</title>',
                 '<p>仅审核原始材料；无生成问题、原题答案或自动通过标签。请回填admission_template.jsonl。</p>']
    for index, m in enumerate(materials):
        fragments.append(f"<article><h2>{index+1}. {html.escape(m['material_id'])}</h2><pre>"+html.escape(m['text'])+
            '</pre><img style="max-width:700px" src="'+html.escape((root/m['image_path']).as_uri(), quote=True)+'"></article>')
    (out / 'review.html').write_text('\n'.join(fragments))
    return {'run_dir': str(out), 'n_sampled': limit, 'status': 'pending_human_admission',
            'frame_size': len(eligible), 'model_calls': 0}


def copy_json(value):
    return json.loads(json.dumps(value))


def freeze_admission(*, run_id, reviews_path, project_root='.', output_root='outputs'):
    from .records import sha256_file
    from .experiments import init_experiment
    root = Path(project_root).resolve(); out = root / output_root / run_id
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['candidate_sha256'] != sha256_file(out / 'candidates.jsonl'):
        raise ValueError('sampled candidates changed')
    reviews = read_jsonl(root / reviews_path); candidates = read_jsonl(out / 'candidates.jsonl')
    raw_problems = root/'data/raw/scienceqa'/REVISION/'problems.json'
    metadata = json.loads(raw_problems.read_text()) if raw_problems.exists() else {}
    if len(reviews) != len(candidates) or len({r['material_id'] for r in reviews}) != len(reviews):
        raise ValueError('review every sampled candidate, not only passed materials')
    index = {r['material_id']: r for r in reviews}; selected, selections, hashes = [], [], set()
    for order, m in enumerate(candidates):
        r = index.get(m['material_id'])
        from .records import review_source
        if not r or not r.get('reviewer') or r.get('image_sha256') != m['image_sha256']:
            raise ValueError('admission reviewer/image binding is required')
        review_source(r)
        checks = r['checks']
        for key in ADMISSION_CHECKS:
            item = checks[key]
            if item.get('value') not in (None, 0, 1) or not isinstance(item.get('reason'), str) or not item['reason'].strip():
                raise ValueError('every admission check needs a specific reason')
        values = [checks[k]['value'] for k in ADMISSION_CHECKS]
        decision = 'fail' if 0 in values else ('pass' if all(v == 1 for v in values) else 'undecided')
        if m['image_sha256'] in hashes:
            decision = 'fail'; checks['source_isolated'] = {'value': 0, 'reason': 'within_sample_shared_image'}
        hashes.add(m['image_sha256'])
        selections.append({**r, 'sample_index': order, 'decision': decision})
        if decision == 'pass' and len(selected) < 30:
            p=metadata.get(m.get('source_id'),{})
            selected.append({**m, 'experiment_partition': 'pilot', 'review_status': 'review_passed',
                             'review_source':r['source'], 'dataset_split':m['split'],
                             'sample_index':order,
                             'subject':p.get('subject'), 'grade':p.get('grade'),
                             'text_sha256':sha256_text(m['text'])})
    # Save the complete review even if 60 candidates yield fewer than 30 passes.
    digest = sha256_file(root / reviews_path)
    if manifest.get('status') in ('frozen_human_reviewed','frozen_delegated_reviewed') and manifest.get('review_sha256') != digest:
        raise ValueError('admission review changed; preserve frozen run and use a new run_id')
    from .records import atomic_upsert
    audit = out/'reviews.jsonl'
    if not any(r['review_id'] == digest for r in read_jsonl(audit)):
        atomic_upsert(audit, 'review_id', {'review_id': digest, 'sources': sorted({r['source'] for r in reviews}),
            'reviews': reviews, 'created_at': utc_now()})
    write_jsonl(out / 'selection.jsonl', selections)
    write_jsonl(out / 'materials.jsonl', selected)
    manifest.update(review_sha256=digest, n_passed=sum(r['decision']=='pass' for r in selections),
                    n_selected=len(selected), materials_sha256=sha256_file(out/'materials.jsonl'),
                    status=('frozen_delegated_reviewed' if any(r['source']=='user_delegated_ai' for r in reviews)
                            else 'frozen_human_reviewed') if len(selected)==30 else 'insufficient_eligible_materials')
    write_json(out / 'manifest.json', manifest)
    if len(selected) < 30:
        return {'status': manifest['status'], 'n_selected': len(selected), 'n_planned': 0}
    alias=out/'materials_final_30.jsonl'
    if not alias.exists(): alias.symlink_to('materials.jsonl')
    if sha256_file(alias)!=manifest['materials_sha256']:
        raise ValueError('final material alias changed')
    manifest['final_materials_path']=str(alias)
    manifest.setdefault('frozen_at',utc_now())
    write_json(out/'manifest.json',manifest)
    result = init_experiment(materials_path=str(alias), run_id=run_id+'-plan',
        methods=tuple(manifest['methods']), partition='pilot', stage=2,
        project_root=root, output_root=output_root)
    return {'status': manifest['status'], 'n_selected': 30, 'n_planned': 120, 'plan_run': result['run_dir']}


def render_material_review_sheets(materials, *, root: Path, output_dir: Path, start_index=0):
    """Source-only sheets for visual review; never used as experiment inputs."""
    import textwrap
    from PIL import Image, ImageDraw, ImageFont
    output_dir.mkdir(parents=True,exist_ok=True)
    font=ImageFont.load_default(size=17); title=ImageFont.load_default(size=21)
    pages=[]
    for offset in range(0,len(materials),6):
        group=materials[offset:offset+6]; panels=[]
        for j,m in enumerate(group):
            lines=[]
            for paragraph in m['text'].splitlines():
                lines.extend(textwrap.wrap(paragraph,width=69) or [''])
            im=Image.open(root/m['image_path']).convert('RGB'); im.thumbnail((610,330))
            height=50+im.height+15+len(lines)*21+15
            panel=Image.new('RGB',(650,height),'white'); draw=ImageDraw.Draw(panel)
            draw.text((10,8),f"{start_index+offset+j+1}. {m['material_id']}",font=title,fill='black')
            panel.paste(im,((650-im.width)//2,45))
            draw.multiline_text((12,im.height+60),'\n'.join(lines),font=font,fill='black',spacing=3)
            panels.append(panel)
        heights=[max(p.height for p in panels[i:i+2]) for i in range(0,len(panels),2)]
        sheet=Image.new('RGB',(1304,sum(heights)+4*(len(heights)-1)),(180,180,180)); y=0
        for i,h in zip(range(0,len(panels),2),heights):
            for col,panel in enumerate(panels[i:i+2]):sheet.paste(panel,(col*654,y))
            y+=h+4
        path=output_dir/f"sources-{start_index+offset+1:04d}-{start_index+offset+len(group):04d}.png"
        sheet.save(path)
        pages.append({'path':str(path),'material_ids':[m['material_id'] for m in group]})
    return pages


def extend_admission(*, run_id, add=60, project_root='.', output_root='outputs'):
    """Append the next fixed-order random candidates before final material freeze."""
    import random
    from .records import sha256_file
    root=Path(project_root).resolve(); out=root/output_root/run_id
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['status'].startswith('frozen_') or add<1:
        raise ValueError('cannot extend frozen admission or request nonpositive extension')
    old=read_jsonl(out/'candidates.jsonl')
    if sha256_file(out/'candidates.jsonl')!=manifest['candidate_sha256']:
        raise ValueError('candidate file changed')
    raw=root/'data/raw/scienceqa'/REVISION
    problems=json.loads((raw/'problems.json').read_text()); splits=json.loads((raw/'pid_splits.json').read_text())
    excluded={r['source_id'] for r in manifest['excluded_development_sources']}
    frame=[str(pid) for pid in splits['train'] if str(pid) not in excluded and problems[str(pid)].get('image')
           and isinstance(problems[str(pid)].get('hint'),str) and problems[str(pid)]['hint'].strip()]
    if sha256_text(canonical_json(frame))!=manifest['sampling_frame_sha256']:
        raise ValueError('cannot reconstruct original sampling frame')
    random.Random(manifest['seed']).shuffle(frame)
    if frame[:len(old)]!=manifest['sampled_ids']:
        raise ValueError('random sequence no longer matches frozen prefix')
    ids=frame[len(old):len(old)+add]
    if not ids: raise ValueError('candidate frame exhausted')
    imported=import_scienceqa(root,split='train',limit=len(ids),source_ids=ids,offline=True)
    by_id={m['source_id']:m for m in load_materials(root,Path(imported['materials']))}
    new=[]
    for pid in ids:
        m=by_id[pid];m.update(text=problems[pid]['hint'],experiment_partition='candidate',
                            review_status='pending',processing_version='admission-v1')
        new.append(m)
    manifest.setdefault('initial_n_sampled',len(old))
    manifest.setdefault('sampling_batches',[{'start':0,'end':len(old),'candidate_sha256':manifest['candidate_sha256']}])
    write_jsonl(out/'candidates.jsonl',old+new)
    manifest['sampling_batches'].append({'start':len(old),'end':len(old)+len(new),'source_ids':ids})
    manifest.update(n_sampled=len(old)+len(new),sampled_ids=manifest['sampled_ids']+ids,
        candidate_sha256=sha256_file(out/'candidates.jsonl'),status='pending_additional_admission')
    write_json(out/'manifest.json',manifest)
    pages=render_material_review_sheets(new,root=root,output_dir=out/'review_sheets',start_index=len(old))
    index_path=out/'review_sheets/index.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else []
    write_json(index_path,index+pages)
    fragments=['<!doctype html><meta charset="utf-8"><title>Source-only admission review</title>']
    for i,m in enumerate(old+new,1):
        fragments.append(f"<article><h2>{i}. {html.escape(m['material_id'])}</h2><pre>"+html.escape(m['text'])+
            '</pre><img style="max-width:700px" src="'+html.escape((root/m['image_path']).as_uri(),quote=True)+'"></article>')
    (out/'review.html').write_text('\n'.join(fragments))
    return {'n_sampled':manifest['n_sampled'],'new_candidates':len(new),'review_sheets':pages}
