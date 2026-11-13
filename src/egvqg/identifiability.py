"""Finite-world diagnostic of answer identifiability, not an open-world oracle."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from .experiments import call_usage
from .pipeline import _validate_run_id
from .prompts import PromptSet
from .providers import JudgeProvider, ModelCaller, call_model
from .records import (atomic_upsert, canonical_json, file_lock, read_jsonl,
                      sha256_file, sha256_text, stable_id, utc_now, write_json, write_jsonl)

TEMPLATE = 'identify-answers-v1'
PROTOCOL = 'finite-answer-identifiability-dev-v1'


def answer_set(values, n_components):
    """Numeric semantic equivalence; no free-text or unit guessing."""
    if not isinstance(values, list) or not values:
        raise ValueError('a nonempty list of answer vectors is required')
    result = set()
    for vector in values:
        if not isinstance(vector, list) or len(vector) != n_components:
            raise ValueError('answer vector has wrong component count')
        if any(isinstance(v, bool) or not isinstance(v, (str, int, float)) for v in vector):
            raise ValueError('numeric answer components required')
        result.add(tuple(str(Fraction(str(v))) for v in vector))
    return sorted(result)


def finite_oracle(case):
    """Exhaust a declared, closed world list. Components are predeclared semantic asks."""
    worlds = case['worlds']; actual = worlds[case['actual_index']]
    n = len(case['components'])
    if not worlds or not n:
        raise ValueError('nonempty worlds and answer components required')
    sets, selected = {}, {}
    for condition in ('TI', 'T', 'I', '0'):
        selected[condition] = [j for j, w in enumerate(worlds) if all(
            w[m] == actual[m] for m in ('T', 'I') if m in condition)]
        sets[condition] = answer_set([worlds[j]['answer'] for j in selected[condition]], n)
    if len(sets['TI']) != 1:
        return {'validity': None, 'answer_sets': sets, 'selected_worlds': selected,
                'text_necessary': None, 'image_necessary': None, 'integration': None}
    component_dependencies = []
    for k in range(n):
        component_dependencies.append({
            'T': int(len({v[k] for v in sets['I']}) > 1),
            'I': int(len({v[k] for v in sets['T']}) > 1)})
    dt = int(len(sets['I']) > 1); di = int(len(sets['T']) > 1)
    return {'validity': 1, 'answer_sets': sets, 'selected_worlds': selected,
            'text_necessary': dt, 'image_necessary': di,
            'components': component_dependencies,
            'integration': int(any(d['T'] and d['I'] for d in component_dependencies)) if dt and di else None}


def init_identifiability(*, run_id, cases_path, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve(); _validate_run_id(run_id)
    out = root/output_root/run_id; cases = read_jsonl(root/cases_path)
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('unique nonempty planned cases required')
    prompts = PromptSet(root/'prompts', names=(TEMPLATE,))
    sources = {str(root/cases_path): sha256_file(root/cases_path),
               str(Path(__file__).resolve()): sha256_file(Path(__file__))}
    for c in cases:
        image = root/c['image_path']
        if sha256_file(image) != c['image_sha256']:
            raise ValueError('image hash mismatch')
        sources[str(image)] = c['image_sha256']
        finite_oracle(c)
    fingerprint = sha256_text(canonical_json({'cases': cases, 'prompts': prompts.snapshot(), 'sources': sources}))
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out/'.init.lock'):
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text())['fingerprint'] != fingerprint:
                raise ValueError('frozen finite probe changed; use new run_id')
            return {'n_planned': len(cases), 'resumed': True}
        plan = [{'case_id': c['case_id'], 'attempt_id': stable_id('attempt', run_id, c['case_id'])} for c in cases]
        write_jsonl(out/'cases.jsonl', cases); write_jsonl(out/'plan.jsonl', plan)
        write_jsonl(out/'records.jsonl', [{**p, 'analysis': {'T': None, 'I': None}} for p in plan])
        write_jsonl(out/'oracle.jsonl', [{'case_id': c['case_id'], **finite_oracle(c)} for c in cases])
        (out/'prompts').mkdir(exist_ok=True)
        (out/'prompts'/f'{TEMPLATE}.txt').write_text(prompts.snapshot()[TEMPLATE])
        write_json(out/'manifest.json', {'fingerprint': fingerprint, 'protocol_version': PROTOCOL,
            'n_planned': len(cases), 'created_at': utc_now(), 'sources': sources,
            'frozen_files': {f: sha256_file(out/f) for f in ('cases.jsonl', 'plan.jsonl', 'oracle.jsonl')},
            'prompt_sha256': prompts.snapshot_sha256(), 'model_identity': None,
            'max_http_requests': 2*len(cases)+4, 'max_output_tokens': 1200,
            'review_source': 'user_delegated_ai', 'partition': 'dev',
            'scope': 'closed finite worlds; no open-world reliability claim'})
    return {'n_planned': len(cases), 'resumed': False}


def _load(out):
    manifest = json.loads((out/'manifest.json').read_text())
    for f, digest in manifest['frozen_files'].items():
        if sha256_file(out/f) != digest: raise ValueError('frozen file changed: '+f)
    for p, digest in manifest['sources'].items():
        if sha256_file(Path(p)) != digest: raise ValueError('source changed: '+p)
    prompts = PromptSet(out/'prompts', names=(TEMPLATE,))
    if prompts.snapshot_sha256() != manifest['prompt_sha256']: raise ValueError('prompt changed')
    rows = read_jsonl(out/'records.jsonl'); plan = read_jsonl(out/'plan.jsonl')
    if (len(rows) != len(plan) or len({r['attempt_id'] for r in rows}) != len(plan)
            or {(r['attempt_id'], r['case_id']) for r in rows} != {(p['attempt_id'], p['case_id']) for p in plan}):
        raise ValueError('fixed planned slots changed')
    return manifest, prompts, rows, {c['case_id']: c for c in read_jsonl(out/'cases.jsonl')}


def run_identifiability(*, run_id, max_requests=26, dry_run=False, project_root='.', output_root='outputs', provider=None):
    root = Path(project_root).resolve(); _validate_run_id(run_id); out = root/output_root/run_id
    manifest, prompts, rows, cases = _load(out)
    budget = min(max_requests, manifest['max_http_requests'])
    pending = sum(r['analysis'][m] is None for r in rows for m in ('T', 'I'))
    if dry_run: return {'n_planned': len(rows), 'pending': pending, 'hard_http_cap': budget}
    if not pending: return {'n_planned': len(rows), 'new_steps': 0}
    provider = provider or JudgeProvider.from_env(root=root, run_dir=out, max_requests=budget,
        max_tokens=manifest['max_output_tokens'], visual=True)
    if isinstance(provider, JudgeProvider): provider.max_attempts = 1
    identity = getattr(provider, 'identity', {'model': provider.model})
    if manifest['model_identity'] not in (None, identity): raise ValueError('model changed')
    manifest['model_identity'] = identity; write_json(out/'manifest.json', manifest)
    caller = ModelCaller(provider=provider, prompts=prompts, run_dir=out,
                         protocol_version=PROTOCOL, max_attempts=2)
    steps = 0
    with file_lock(out/'.pipeline.lock'):
        for r in read_jsonl(out/'records.jsonl'):
            c = cases[r['case_id']]
            for removed, retained in (('T', 'I'), ('I', 'T')):
                if r['analysis'][removed] is not None: continue
                text = c['text'] if retained == 'T' else None
                image = str(root/c['image_path']) if retained == 'I' else None
                call = call_model(caller, operation_id=stable_id('finite', r['attempt_id'], removed),
                    attempt_id=r['attempt_id'], role='answer_set_without_'+removed, template=TEMPLATE,
                    variables={'Q': c['question'], 'PROFILE': c['completion_profile'],
                        'COMPONENTS': canonical_json(c['components']),
                        'VISIBLE': '<T>'+text+'</T>' if text else '仅随请求附上的原图；没有文本材料。'},
                    task='assess_quality', question=c['question'], text=text, image_path=image, condition=retained)
                r['analysis'][removed] = {'call_id': call['call_id'], 'status': call['status'],
                    'output': call.get('parsed_output'), 'error': call.get('error')}
                atomic_upsert(out/'records.jsonl', 'attempt_id', r, allow_insert=False); steps += 1
                print(json.dumps({'case_id': r['case_id'], 'removed': removed, 'status': call['status']}, ensure_ascii=False), flush=True)
    report_identifiability(run_id=run_id, project_root=root, output_root=output_root)
    return {'n_planned': len(rows), 'new_steps': steps}


def report_identifiability(*, run_id, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve(); _validate_run_id(run_id); out = root/output_root/run_id
    _, _, rows, cases = _load(out); results = []
    for r in rows:
        c = cases[r['case_id']]; oracle = finite_oracle(c); analyses = {}
        for removed, retained in (('T', 'I'), ('I', 'T')):
            item = r['analysis'][removed]; values = None; error = None
            try:
                if item and item['status'] == 'ok': values = answer_set(item['output']['answers'], len(c['components']))
            except (KeyError, ValueError, TypeError, ZeroDivisionError): error = 'invalid_answer_set'
            expected = oracle['answer_sets'][retained]
            analyses[removed] = {'answers': values, 'exact_set': None if values is None else values == expected,
                'raw_necessary': None if values is None else int(len(values) > 1),
                'verified_necessary': int(len(expected) > 1) if values == expected else None,
                'error': error, 'expected_answers': expected}
        results.append({'case_id': c['case_id'], 'source_kind': c['source_kind'], 'oracle': oracle, 'analyses': analyses})
    counts = {m: {'n_planned': len(rows), 'exact': sum(r['analyses'][m]['exact_set'] is True for r in results),
                 'mismatch': sum(r['analyses'][m]['exact_set'] is False for r in results),
                 'unresolved': sum(r['analyses'][m]['exact_set'] is None for r in results)} for m in ('T', 'I')}
    write_json(out/'results.json', {'n_planned': len(rows), 'cases': results, 'answer_set_checks': counts})
    usage = call_usage(read_jsonl(out/'calls.jsonl')); write_json(out/'usage.json', usage)
    return {'n_planned': len(rows), 'answer_set_checks': counts, 'usage': usage}
