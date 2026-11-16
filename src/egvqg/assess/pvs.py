"""Portfolio Verify-and-Select (clean version): local prefilter, SAQG v2, isolated selection/evaluation.

Selection judge: gpt-6.1-sol (serial branches, early stop).
Evaluation jury: gpt-6-luna + DeepSeek-V4.1-Flash (independent; DeepSeek breaks ties).
Local generator/answerer/extractor: Qwen3-VL-8B.
The two judge pools never share labels: selection writes its own file; jury only reads delivered questions.
"""
from __future__ import annotations

import json
from pathlib import Path

from .baseline_analysis import analysis_input
from .assessment_rules import load_assessment_config
from ..core.llm import json_object
from ..core.prompts import PromptSet
from ..core.records import append_jsonl, read_jsonl, utc_now, write_json
from ..core.providers import CompletionRequest, JudgeProvider

SELECTION_MODEL = 'gpt-6.1-sol'
JURY = ('gpt-6-luna', 'DeepSeek-V4.1-Flash')
SEL_MAX_TOKENS = 4096


def _cand_id(candidate):
    """Candidate files in this project use either `cand_id` or `candidate_id`."""
    return candidate.get('cand_id') or candidate.get('candidate_id')


def _as_bool(parsed, key):
    value = parsed.get(key) if isinstance(parsed, dict) else None
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ('true', 'false'):
        return value.strip().lower() == 'true'
    return None


def _provider(model, root, run_dir, *, max_requests=20000, visual=True, max_tokens=SEL_MAX_TOKENS):
    p = JudgeProvider.from_env(root=root, run_dir=run_dir, max_requests=max_requests, max_tokens=max_tokens,
                               visual=visual, model=model)
    p.max_attempts = 1
    return p


def _ask(provider, prompt, image_path=None):
    return provider.complete(CompletionRequest(task='assess_quality', prompt=prompt, image_path=image_path))


def _screen(provider, prompts, cfg, question, material, removed):
    row = {'question': {'normalized_text': question}, 'material': material}
    payload = analysis_input(row, 'screen', removed, cfg)
    prompt = prompts.render(payload['template'], **payload['variables'])
    image = payload.get('image_path')
    return _as_bool(json_object(_ask(provider, prompt, image).text), 'determinable')


def _stitch(provider, prompts, question, material, root):
    prompt = prompts.render('pvs-stitch-v1', Q=question, T=material['text'])
    image = str(root / material['image_path']) if material.get('image_path') else None
    return _as_bool(json_object(_ask(provider, prompt, image).text), 'stitching')


def prefilter(*, run_id, materials_file, model_path, project_root='.', output_root='outputs', device='cuda:0', limit=None):
    from ..core.local_vision import QwenVLBackend
    root = Path(project_root).resolve(); out = root / output_root / run_id; out.mkdir(parents=True, exist_ok=True)
    materials = read_jsonl(root / materials_file)
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=256)
    prompts = PromptSet(root / 'prompts', names=('pvs-prefilter-v1',))
    backend._load()
    for n, material in enumerate(materials, 1):
        if limit is not None and n > limit: break
        request = CompletionRequest(task='assess_quality', prompt=prompts.render('pvs-prefilter-v1', T=material['text']),
                                    image_path=str(root / material['image_path']))
        text = backend.generate([request])[0].text
        parsed = json_object(text) or {}
        potential = parsed.get('potential') if parsed.get('potential') in ('likely-positive', 'likely-negative') else 'likely-positive'
        append_jsonl(out / 'prefilter.jsonl', {'material_id': material['material_id'], 'potential': potential,
                                               'parsed': bool(parsed), 'reason': parsed.get('reason'),
                                               'raw': text[:400], 'created_at': utc_now()})
    return {'run_id': run_id, 'materials': len(materials)}


def select(*, run_id, candidates_file, materials_file, project_root='.', output_root='outputs', workers=1):
    root = Path(project_root).resolve(); out = root / output_root / run_id; out.mkdir(parents=True, exist_ok=True)
    materials = {m['material_id']: m for m in read_jsonl(root / materials_file)}
    prompts = PromptSet(root / 'prompts', names=('ensemble-sufficiency-v1', 'pvs-stitch-v1'))
    cfg = load_assessment_config(root, 'configs/necessity_assessment_v2.json')
    import itertools, threading
    from concurrent.futures import ThreadPoolExecutor
    done = {r['cand_id'] for r in read_jsonl(out / 'selection.jsonl')} if (out / 'selection.jsonl').exists() else set()
    todo = [c for c in read_jsonl(root / candidates_file) if _cand_id(c) not in done]
    counts = {'text_suff': 0, 'image_suff': 0, 'stitching': 0, 'dual': 0, 'undetermined': 0}
    lock = threading.Lock(); counter = itertools.count(); local = threading.local()
    def provider():
        if not hasattr(local, 'p'):
            with lock: i = next(counter)
            local.p = _provider(SELECTION_MODEL, root, out / f'selection_ledger_{i}')
        return local.p
    def one(c):
        q = c['question']; m = materials[c['material_id']]
        row = {'cand_id': _cand_id(c), 'material_id': c['material_id'], 'question': q,
               'sol_branch1_label': None, 'sol_branch2_label': None, 'sol_stitching_label': None,
               'sol_final_label': 'undetermined', 'created_at': utc_now()}
        try:
            text_suff = _screen(provider(), prompts, cfg, q, m, 'I'); row['sol_branch1_label'] = text_suff
            if text_suff is True: row['sol_final_label'] = 'rejected_text_suff'; return row
            if text_suff is not False: return row
            image_suff = _screen(provider(), prompts, cfg, q, m, 'T'); row['sol_branch2_label'] = image_suff
            if image_suff is True: row['sol_final_label'] = 'rejected_image_suff'; return row
            if image_suff is not False: return row
            stitch = _stitch(provider(), prompts, q, m, root); row['sol_stitching_label'] = stitch
            if stitch is True: row['sol_final_label'] = 'rejected_stitching'; return row
            if stitch is not False: return row
            row['sol_final_label'] = 'dual_candidate'; return row
        except Exception as exc:
            row['sol_error'] = f'{type(exc).__name__}: {str(exc)[:160]}'; return row
    if workers <= 1:
        results = [one(c) for c in todo]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(one, todo))
    for row in results:
        label = row['sol_final_label']
        counts[{'rejected_text_suff': 'text_suff', 'rejected_image_suff': 'image_suff',
                'rejected_stitching': 'stitching', 'dual_candidate': 'dual'}.get(label, 'undetermined')] += 1
        append_jsonl(out / 'selection.jsonl', row)
    write_json(out / 'selection_summary.json', {'run_id': run_id, 'counts': counts, 'workers': workers})
    return counts


def deliver(*, run_id, selection_file, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve(); out = root / output_root / run_id
    rows = read_jsonl(out / selection_file)
    by = {}
    for r in rows:
        if r.get('sol_final_label') == 'dual_candidate' and r['material_id'] not in by:
            by[r['material_id']] = r
    with (out / 'delivered.jsonl').open('w') as f:
        for r in by.values(): f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
    write_json(out / 'delivery_summary.json', {'delivered': len(by), 'candidates': len(rows)})
    return {'delivered': len(by)}


def jury(*, run_id, delivered_file, materials_file, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve(); out = root / output_root / run_id
    materials = {m['material_id']: m for m in read_jsonl(root / materials_file)}
    prompts = PromptSet(root / 'prompts', names=('ensemble-sufficiency-v1', 'pvs-stitch-v1'))
    cfg = load_assessment_config(root, 'configs/necessity_assessment_v2.json')
    providers = {m: _provider(m, root, out / f'jury_ledger_{i}', max_tokens=SEL_MAX_TOKENS)
                 for i, m in enumerate(JURY)}
    done = {r['cand_id'] for r in read_jsonl(out / 'jury.jsonl')} if (out / 'jury.jsonl').exists() else set()
    for d in read_jsonl(out / delivered_file):
        if d['cand_id'] in done:
            continue
        q = d['question']; m = materials[d['material_id']]
        verdicts = {}
        for name in JURY:
            p = providers[name]
            try:
                text_suff = _screen(p, prompts, cfg, q, m, 'I')
                image_suff = _screen(p, prompts, cfg, q, m, 'T') if text_suff is False else None
                stitch = _stitch(p, prompts, q, m, root) if image_suff is False else None
            except Exception as exc:
                verdicts[name] = {'verdict': 'undetermined', 'error': f'{type(exc).__name__}: {str(exc)[:160]}'}
                continue
            if text_suff is True or image_suff is True or stitch is True:
                verdicts[name] = {'verdict': 'not_dual', 'text_suff': text_suff, 'image_suff': image_suff, 'stitching': stitch}
            elif text_suff is False and image_suff is False and stitch is False:
                verdicts[name] = {'verdict': 'dual', 'text_suff': False, 'image_suff': False, 'stitching': False}
            else:
                verdicts[name] = {'verdict': 'undetermined', 'text_suff': text_suff, 'image_suff': image_suff, 'stitching': stitch}
        a, b = verdicts[JURY[0]]['verdict'], verdicts[JURY[1]]['verdict']
        if a == b:
            final = a
        elif 'undetermined' in (a, b):
            final = 'undetermined'
        else:
            final = b       # DeepSeek arbitration, as specified
        append_jsonl(out / 'jury.jsonl', {'cand_id': d['cand_id'], 'material_id': d['material_id'],
                                          'question': q, 'jury': verdicts, 'jury_consensus_label': final,
                                          'created_at': utc_now()})
    return {'delivered': len(read_jsonl(out / delivered_file))}
