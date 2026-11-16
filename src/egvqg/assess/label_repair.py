"""Stage 0 / P1: re-label a stratified candidate sample with two independent evidence-based judges.

Each candidate is asked twice per judge: can the *text alone* determine the answer, and can the *image alone*?
The payload is built with `baseline_analysis.analysis_input(..., stage='screen')`, so the T-only / I-only
isolation of the existing evaluator is reused. Verdicts: dual_necessary / text_suff / image_suff / both_suff /
undetermined. Disagreement between the two groups goes to a third judge; a failed or unparsable call stays
undetermined (never flipped to a label). Sampling is stratified by old redundant label x skill, capped per parent.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..generate import candidate_table as CT
from .assessment_rules import load_assessment_config
from .baseline_analysis import analysis_input
from ..generate.candidates import load_materials_any
from ..core.llm import CachedCaller, json_object
from ..core.prompts import PromptSet
from ..core.records import write_json, write_jsonl
from ..core.stats import wilson

VERDICTS = ('dual_necessary', 'text_suff', 'image_suff', 'both_suff', 'undetermined')
PROMPT = 'ensemble-sufficiency-v1'


def stratified_sample(rows, *, rate=1.0, per_parent=3, seed=20261113):
    """Sample candidates per (old redundant label, skill) stratum, then cap each parent material."""
    strata = defaultdict(list)
    for r in rows:
        if r.get('valid') == 0:
            continue
        strata[(r['label_old'].get('redundant'), r['skill'])].append(r)
    rng = random.Random(seed)
    picked = []
    for key in sorted(strata, key=lambda k: (str(k[0]), k[1])):
        items = sorted(strata[key], key=lambda r: r['cand_id'])
        rng.shuffle(items)
        n = max(1, round(len(items) * rate))
        picked += items[:n]
    by_parent = defaultdict(list)
    for r in picked:
        by_parent[r['parent_id']].append(r)
    out = []
    for parent in sorted(by_parent):
        items = sorted(by_parent[parent], key=lambda r: (r['label_old'].get('redundant') is None,
                                                         r['label_old'].get('redundant'), r['cand_id']))
        out += items[:per_parent]
    return out


def select_parents(rows, n_parents):
    """Round-robin across skills (one parent per skill per rank, sorted, deterministic) without skipping parents.
    Only rows whose old evaluation is not invalid (valid == 0) can contribute a parent."""
    by_skill = defaultdict(set)
    for r in rows:
        if r.get('valid') != 0:
            by_skill[r['skill']].add(r['parent_id'])
    ordered = {skill: sorted(parents) for skill, parents in sorted(by_skill.items())}
    limit = int(n_parents)
    picked = []
    for rank in range(max((len(v) for v in ordered.values()), default=0)):
        for parents in ordered.values():
            if rank < len(parents) and len(picked) < limit:
                picked.append(parents[rank])
        if len(picked) >= limit:
            break
    return set(picked)


def _judge_side(caller, model, prompts, q, material, cfg, side):
    removed = 'I' if side == 'text' else 'T'
    payload = analysis_input({'question': {'normalized_text': q}, 'material': material}, 'screen', removed, cfg)
    prompt = prompts.render(payload['template'], **payload['variables'])
    image = str(caller.root / payload['image_path']) if payload.get('image_path') else None
    try:
        reply = caller.call(model=model, prompt=prompt, image_path=image, task='assess_quality')
    except Exception as exc:                     # transport/parse failure: keep undetermined, never flip the label
        return {'status': 'technical_failure', 'error': f'{type(exc).__name__}: {str(exc)[:160]}',
                'determinable': None, 'answer': None, 'reason': None}
    parsed = json_object((reply or {}).get('text', ''))
    value = parsed.get('determinable') if isinstance(parsed, dict) else None
    if isinstance(value, str) and value.strip().lower() in ('true', 'false'):
        value = value.strip().lower() == 'true'
    ok = isinstance(parsed, dict) and isinstance(value, bool)
    return {'status': 'ok' if ok else 'unparsable', 'determinable': value if ok else None,
            'answer': parsed.get('answer') if ok else None, 'reason': parsed.get('reason') if ok else None}


def _verdict(sides):
    t, i = sides['text']['determinable'], sides['image']['determinable']
    if t is None or i is None:
        return 'undetermined'
    if t and i:
        return 'both_suff'
    if t:
        return 'text_suff'
    if i:
        return 'image_suff'
    return 'dual_necessary'


DIRECTION = {'text_suff': 'text', 'image_suff': 'image', 'both_suff': 'both'}


def label_candidate(caller, models, arbiter, prompts, row, material, cfg):
    groups = {}
    for n, model in enumerate(models):
        sides = {side: _judge_side(caller, model, prompts, row['question'], material, cfg, side) for side in ('text', 'image')}
        for side, s in sides.items():
            if s['status'] != 'ok':
                print(f'[label-repair] {row["cand_id"]} {model} {side}: {s["status"]} {s.get("error", "")}', flush=True)
        groups[f'group{n + 1}'] = {'model': model, 'sides': sides, 'verdict': _verdict(sides)}
    verdicts = [g['verdict'] for g in groups.values()]
    chosen = verdicts[0]
    arbitration = None
    if len(set(verdicts)) > 1:
        if arbiter and arbiter not in models:
            sides = {side: _judge_side(caller, arbiter, prompts, row['question'], material, cfg, side) for side in ('text', 'image')}
            arbitration = {'model': arbiter, 'sides': sides, 'verdict': _verdict(sides)}
            chosen = arbitration['verdict']
        else:
            chosen = 'undetermined' if 'undetermined' in verdicts else verdicts[0]
    return {'cand_id': row['cand_id'], 'verdict': chosen, 'direction': DIRECTION.get(chosen),
            'evidence': {'groups': groups, 'arbitration': arbitration}}


def consensus_verdict(label):
    """A decided label requires the two independent judges to return the same verdict. Disagreements stay undetermined;
    the arbiter verdict is still stored as provisional evidence but is not allowed to create a decided label on its own."""
    verdicts = [g['verdict'] for g in label.get('evidence', {}).get('groups', {}).values()]
    if len(verdicts) >= 2 and len(set(verdicts)) == 1 and verdicts[0] != 'undetermined':
        return verdicts[0]
    return 'undetermined'


def _report(rows, labelled):
    new = {r['cand_id']: r for r in labelled}
    pairs = [(r, new[r['cand_id']]) for r in rows if r['cand_id'] in new]
    old_nonred = [(r, n) for r, n in pairs if r['label_old'].get('redundant') == 0]
    new_dual = [(r, n) for r, n in pairs if n['verdict'] == 'dual_necessary']
    old_dual = [(r, n) for r, n in old_nonred if n['verdict'] == 'dual_necessary']
    both_dir = [(r, n) for r, n in pairs if r['label_old'].get('direction') in ('image', 'text')
                and n['direction'] in ('image', 'text')]
    agree = [(r, n) for r, n in both_dir if r['label_old']['direction'] == n['direction']]
    precision = wilson(len(old_dual), len(old_nonred))
    recall = wilson(len(old_dual), len(new_dual))
    direction = wilson(len(agree), len(both_dir))
    decided = [(r, n) for r, n in pairs
               if len(n.get('evidence', {}).get('groups', {})) >= 2
               and all(g['verdict'] != 'undetermined' for g in n['evidence']['groups'].values())]
    judge_agree = [(r, n) for r, n in decided
                   if len({g['verdict'] for g in n['evidence']['groups'].values()}) == 1]
    inter_judge = wilson(len(judge_agree), len(decided))
    consensus = {n['cand_id']: consensus_verdict(n) for _, n in pairs}
    cons_pairs = [(r, n) for r, n in pairs if consensus[n['cand_id']] != 'undetermined']
    cons_nonred = [(r, n) for r, n in cons_pairs if r['label_old'].get('redundant') == 0]
    cons_dual = [(r, n) for r, n in cons_nonred if consensus[n['cand_id']] == 'dual_necessary']
    cons_new_dual = [(r, n) for r, n in cons_pairs if consensus[n['cand_id']] == 'dual_necessary']
    cons_precision = wilson(len(cons_dual), len(cons_nonred))
    cons_recall = wilson(len(cons_dual), len(cons_new_dual))
    return {'sampled': len(pairs), 'verdicts': dict(Counter(n['verdict'] for _, n in pairs)),
            'consensus_verdicts': dict(Counter(consensus[n['cand_id']] for _, n in pairs)),
            'inter_judge_verdict_agreement': inter_judge,
            'old_nonredundant_precision': precision, 'old_nonredundant_recall_estimate': recall,
            'consensus_old_nonredundant_precision': cons_precision, 'consensus_old_nonredundant_recall_estimate': cons_recall,
            'direction_agreement': direction,
            'criterion': {'nonredundant_precision_lower_ge_0.80': bool(precision['lo'] is not None and precision['lo'] >= 0.80),
                          'nonredundant_recall_lower_ge_0.80': bool(recall['lo'] is not None and recall['lo'] >= 0.80),
                          'inter_judge_agreement_lower_ge_0.80': bool(inter_judge['lo'] is not None and inter_judge['lo'] >= 0.80),
                          'direction_agreement_lower_ge_0.80': bool(direction['lo'] is not None and direction['lo'] >= 0.80),
                          'reference': 'inter-judge consensus replaces the historical cascade label as reference; '
                                       'the historical-label comparison is reported and not used as a gate',
                          'note': 'printed only; no gate and no label is changed automatically'}}


def run_label_repair(*, capture_run=None, label_file=None, candidates=None, project_root='.', output_root='outputs', run_id=None,
                     reference_run='qwen3vl8b-baseline-test1257-simple-v1',
                     materials_file=None, materials=None, judges=('gpt-6-luna', 'gpt-5.6-terra'), arbiter='DeepSeek-V4.1-Flash',
                     sample_rate=1.0, per_parent=3, n_parents=None, dry_run=False, max_calls=0, seed=20261113, concurrency=1):
    root = Path(project_root).resolve()
    if candidates:
        rows = CT.load_table(candidates)
    else:
        if not (capture_run and label_file):
            raise ValueError('either --candidates or both --capture-run and --label-file are required')
        rows = CT.build_table(capture_run=capture_run, label_file=label_file, project_root=project_root, output_root=output_root)
    if materials:
        ids = {line.split('#')[0].strip() for line in Path(materials).read_text().splitlines() if line.split('#')[0].strip()}
        rows = [r for r in rows if r['parent_id'] in ids or r['material_id'] in ids]
    if n_parents:
        picked = select_parents(rows, n_parents)
        rows = [r for r in rows if r['parent_id'] in picked]
    sample = stratified_sample(rows, rate=sample_rate, per_parent=per_parent, seed=seed)
    cfg = load_assessment_config(root, 'configs/necessity_assessment_v2.json')
    prompts = PromptSet(root / 'prompts', names=(PROMPT,))
    caller = CachedCaller(project_root=project_root, run_id=run_id, output_root=output_root, dry_run=dry_run, max_calls=max_calls)
    n_judges = len([m for m in judges if m])
    arb_rate = 0.10
    if dry_run:
        try:
            mat = load_materials_any(root, output_root, reference_run, materials_file)
        except (FileNotFoundError, KeyError):
            mat = {}
        example = sample[0] if sample else None
        payload = (analysis_input({'question': {'normalized_text': example['question']},
                                   'material': mat[example['parent_id']]}, 'screen', 'I', cfg) if example and example['parent_id'] in mat else None)
        if payload:
            prompt = prompts.render(payload['template'], **payload['variables'])
            caller.examples.append({'model': judges[0], 'image': False, 'prompt': prompt[:600]})
        parents = len({r['parent_id'] for r in sample}) or 1
        per_parent_calls = 2 * n_judges * (1 + arb_rate)
        for n_parents in (60, 320):
            est = round(n_parents * per_parent * per_parent_calls)
            print(f'[dry-run] label-repair: {n_parents} parents x <= {per_parent} candidates x 2 sides x {n_judges} judges '
                  f'(+{int(arb_rate * 100)}% arbitration) ~= {est} calls')
        print(f'[dry-run] would sample {len(sample)} candidates from {parents} parents; sampled rows per stratum are deterministic (seed={seed})')
        print('[dry-run] route budget note: roughly 6,700 calls remain; the 320-parent estimate must be approved before launch')
        for ex in caller.examples[:2]:
            print(f'[dry-run] example prompt ({ex["model"]}, image={ex["image"]}):\n{ex["prompt"][:400]}')
        return {'dry_run': True, 'sampled_candidates': len(sample), 'parents': parents, **caller.summary()}

    if not sample:
        raise ValueError('nothing to label after filtering/sampling')
    if not 1 <= int(concurrency) <= 32:
        raise ValueError('invalid label-repair concurrency')
    mat = load_materials_any(root, output_root, reference_run, materials_file)

    def label_one(row):
        material = mat.get(row['parent_id'])
        if material is None:
            return {'cand_id': row['cand_id'], 'verdict': 'undetermined', 'direction': None,
                    'evidence': {'error': 'material not found'}}
        return label_candidate(caller, [m for m in judges if m], arbiter, prompts, row, material, cfg)

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            labelled = list(pool.map(label_one, sample))
    else:
        labelled = []
        for n, row in enumerate(sample, 1):
            labelled.append(label_one(row))
            if n % 10 == 0:
                print(f'[label-repair] {n}/{len(sample)} candidates; calls={caller.actual} cache_hits={caller.cache_hits}', flush=True)
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    print(f'[label-repair] writing {len(labelled)} labels to {out}', flush=True)
    write_jsonl(out / 'labels_new.jsonl', labelled)
    consensus = {n['cand_id']: consensus_verdict(n) for n in labelled}
    write_jsonl(out / 'labels_consensus.jsonl', [
        {'cand_id': n['cand_id'], 'verdict': consensus[n['cand_id']], 'direction': DIRECTION.get(consensus[n['cand_id']]),
         'agreement': consensus[n['cand_id']] != 'undetermined' or n['verdict'] != 'undetermined',
         'provisional_verdict': n['verdict']} for n in labelled])
    if candidates:
        merged = CT.load_table(candidates)
        by_id = {r['cand_id']: r for r in labelled}
        for r in merged:
            if r['cand_id'] in by_id:
                n = by_id[r['cand_id']]
                r['label_new'] = {'verdict': n['verdict'], 'direction': n['direction'], 'evidence': n['evidence']}
                r['label_consensus'] = {'verdict': consensus[n['cand_id']], 'direction': DIRECTION.get(consensus[n['cand_id']])}
    else:
        merged = CT.build_table(capture_run=capture_run, label_file=label_file, project_root=project_root,
                                output_root=output_root, label_new=str(out / 'labels_new.jsonl'))
        for r in merged:
            if r['cand_id'] in consensus:
                r['label_consensus'] = {'verdict': consensus[r['cand_id']], 'direction': DIRECTION.get(consensus[r['cand_id']])}
    write_jsonl(out / 'candidates.jsonl', merged)
    report = {**_report(rows, labelled), 'run_id': run_id, 'capture_run': capture_run, 'sample_rate': sample_rate,
              'per_parent': per_parent, 'n_parents': n_parents, 'judges': list(judges), 'arbiter': arbiter, **caller.summary()}
    write_json(out / 'label_repair_report.json', report)
    print(f'[label-repair] verdicts={report["verdicts"]} precision={report["old_nonredundant_precision"]}')
    print(f'[label-repair] criterion={report["criterion"]}')
    return report
