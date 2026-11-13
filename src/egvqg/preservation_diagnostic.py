"""A sealed, evaluation-only preservation diagnostic. Never emits necessity labels."""
from __future__ import annotations

import json
from pathlib import Path

from .assessment_rules import choice_conflicts
from .evidence_verification import CHECKS, verification_gate
from .experiments import call_usage
from .pipeline import _validate_run_id
from .prompts import PromptSet
from .providers import JudgeProvider, ModelCaller, call_model
from .records import (atomic_upsert, canonical_json, file_lock, read_jsonl, sha256_file,
                      sha256_text, stable_id, utc_now, write_json, write_jsonl)

V0 = 'verify-necessity-evidence-v2'
EXTRACT = 'extract-preservation-constraints-diag-v1'
PRESERVE = 'check-preservation-constraints-diag-v1'
TEMPLATES = (V0, EXTRACT, PRESERVE)
READINGS = [
    'Adopt the separated check if the conjunction rejects at least 4 more of the 19 defective proofs than V0 alone and falsely rejects at most 1 of the 8 clean proofs.',
    'Keep the current design if V0 alone already rejects 17 or more of the 19.',
    'H-density has support if V0 misses at least 3 more elaborate than terse planted defects (out of 8 each). If the difference is 1 or less, postpone the density concern for proofs.',
    'Do not adopt if the conjunction falsely rejects 3 or more clean proofs; report what the extraction got wrong.',
    'Anything in between is inconclusive. These counts are directional, not rates or accuracy estimates.',
]


def evidence_content(proof, *, preservation=False):
    witness = proof.get('kind') == 'ambiguous'
    keys = (('scenario_a', 'scenario_b') if preservation else
            ('scenario_a', 'answer_a', 'scenario_b', 'answer_b')) if witness else ('answer', 'reason')
    content = {k: proof[k] for k in keys}
    form = 'two_hypothetical_completions' if witness else 'single_sufficient_solution_no_scenarios_required'
    return form, content


def call_inputs(case, removed, arm, knowledge, *, proof=None, constraints=None):
    """Allowlisted model inputs; labels, provenance and other arms are never accepted."""
    if arm == 'V1':
        form, content = evidence_content(proof, preservation=True)
        return dict(template=PRESERVE, variables={'CONSTRAINTS': canonical_json(constraints),
                    'PROOF_FORM': form, 'CONTENT': canonical_json(content)},
                    text=None, image_path=None, question=None, condition=None)
    retained_text = case['text'] if removed == 'I' else None
    variables = {'Q': case['question'], 'REMOVED': '文本' if removed == 'T' else '图像',
                 'VISIBLE': '<T>'+retained_text+'</T>' if retained_text else '仅附上的原图，没有原始文本。',
                 'KNOWLEDGE': knowledge}
    if arm == 'V0':
        form, content = evidence_content(proof)
        variables['EVIDENCE'] = canonical_json({'proof_form': form, **content})
    return dict(template=V0 if arm == 'V0' else EXTRACT, variables=variables,
                text=retained_text, image_path=case['image_path'] if removed == 'T' else None,
                question=case['question'], condition='I' if removed == 'T' else 'T')


def code_gate(proof, choices, v0_output=None):
    keys = ('scenario_a', 'answer_a', 'scenario_b', 'answer_b', 'reason') if proof.get('kind') == 'ambiguous' else ('answer', 'reason')
    shape = proof.get('kind') in ('unique', 'ambiguous') and all(
        isinstance(proof.get(k), str) and proof[k].strip() for k in keys)
    conflicts = choice_conflicts(proof, choices) + choice_conflicts(v0_output, choices)
    return 'accept' if shape and not conflicts else 'reject'


def conjunction(code, v0, v1):
    return 'accept' if (code, v0, v1) == ('accept', 'accept', 'accept') else 'reject'


def counted_rejection(label, *results):
    # Failure is NOT evidence of defect rejection. For clean items every non-pass
    # counts against the arm. This scoring never grants operational acceptance.
    return any(r == 'reject' for r in results) if label != 'clean' else any(r != 'accept' for r in results)


def strict_output(call):
    if call['status'] != 'ok':
        return None, call.get('error') or call['status']
    try:
        output = json.loads(call['raw_output'])
        if not isinstance(output, dict):
            raise ValueError('expected JSON object')
        return output, None
    except (ValueError, TypeError) as exc:
        return None, 'strict_json_failure: '+str(exc)


def valid_extraction(output):
    if not isinstance(output, dict) or not isinstance(output.get('constraints'), list) or not output['constraints']:
        return False
    if not isinstance(output.get('referring_expressions'), list):
        return False
    rows = output['constraints'] + output['referring_expressions']
    if any(not isinstance(r, dict) for r in rows):
        return False
    ids = [r.get('id') for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        return False
    for r in output['constraints']:
        if r.get('source') not in ('question', 'retained') or r.get('type') not in ('constraint', 'presupposition'):
            return False
        if any(not isinstance(r.get(k), str) or not r[k].strip() for k in ('quote', 'constraint')):
            return False
    for r in output['referring_expressions']:
        if r.get('in_removed_modality') not in ('yes', 'no', 'uncertain') or any(
                not isinstance(r.get(k), str) or not r[k].strip() for k in ('quote', 'referent')):
            return False
    return True


def arm_result(arm, output, constraints=None):
    if not isinstance(output, dict):
        return 'technical_failure'
    if arm == 'V0':
        if any(output.get(k) not in ('pass', 'fail', 'uncertain') for k in CHECKS) or not isinstance(output.get('reason'), str) or not output['reason'].strip():
            return 'technical_failure'
        gate = verification_gate(output, strict=True)
        if gate == 'accept':
            return 'accept'
        if gate == 'reject' or any(output[k] == 'uncertain' for k in CHECKS):
            return 'reject'
        return 'technical_failure'
    if not valid_extraction(constraints):
        return 'technical_failure'
    checks = output.get('checks')
    ids = {r['id'] for r in constraints['constraints'] + constraints['referring_expressions']}
    if not isinstance(checks, list) or any(not isinstance(c, dict) for c in checks):
        return 'technical_failure'
    if len(checks) != len(ids) or {c.get('id') for c in checks} != ids:
        return 'technical_failure'
    if any(c.get('result') not in ('pass', 'fail', 'uncertain') or
           not isinstance(c.get('quote'), str) or not isinstance(c.get('reason'), str) or not c['reason'].strip() for c in checks):
        return 'technical_failure'
    overall = 'fail' if any(c['result'] == 'fail' for c in checks) else 'uncertain' if any(c['result'] == 'uncertain' for c in checks) else 'pass'
    if output.get('overall') != overall or not isinstance(output.get('reason'), str) or not output['reason'].strip():
        return 'technical_failure'
    return 'accept' if overall == 'pass' else 'reject'


def init_diagnostic(*, run_id, construction_path, project_root='.', output_root='outputs', provider=None):
    root = Path(project_root).resolve(); _validate_run_id(run_id)
    out = root / output_root / run_id
    if out.exists():
        raise ValueError('init requires a new run id; resume with run')
    construction = json.loads((root / construction_path).read_text())
    items = construction['items']
    if len(items) != 27 or len({i['item_id'] for i in items}) != 27:
        raise ValueError('fixed 27 unique proofs required')
    expected = {'clean': 8, 'synthetic_planted_defect': 16, 'real_defect': 3}
    if {k: sum(i['sealed_label'] == k for i in items) for k in expected} != expected:
        raise ValueError('fixed construction denominators changed')
    groups = sorted({i['source_item_id'] for i in items})
    if len(groups) != 11:
        raise ValueError('11 extraction groups required')
    source = root / 'outputs/real-scienceqa20-v2'
    cases = {c['case_id']: c for c in read_jsonl(source / 'cases.jsonl')}
    selected_cases = [cases[c] for c in sorted({i['case_id'] for i in items})]
    config = json.loads((root / 'configs/necessity_assessment_v2.json').read_text())
    prompts = PromptSet(root / 'prompts', names=TEMPLATES)
    # Protect all pre-existing runs and scientific definitions, without reading credentials.
    protected = {}
    for directory in (root / output_root).iterdir():
        if directory.is_dir() and directory.name != 'cache':
            for p in directory.rglob('*'):
                if p.is_file() and not p.name.endswith('.lock'):
                    protected[str(p.relative_to(root))] = sha256_file(p)
    for p in [root / 'VQG_Proposal.md', root / 'uv.lock', root / 'pyproject.toml',
              *sorted((root / 'docs').glob('*.md')), *sorted((root / 'configs').glob('*.json'))]:
        protected[str(p.relative_to(root))] = sha256_file(p)
    for c in selected_cases:
        p = root / c['image_path']
        if sha256_file(p) != c['image_sha256']:
            raise ValueError('source image hash mismatch')
        protected[c['image_path']] = c['image_sha256']
    out.mkdir(parents=True)
    provider = provider or JudgeProvider.from_env(root=root, run_dir=out, model='gpt-6-luna',
                 max_requests=70, max_tokens=config['verification_max_tokens'], visual=True)
    identity = getattr(provider, 'identity', {'model': provider.model})
    if isinstance(provider, JudgeProvider) and provider.model != 'gpt-6-luna':
        raise ValueError('this diagnostic requires gpt-6-luna')
    write_jsonl(out / 'proof_set.jsonl', items)
    write_jsonl(out / 'cases.jsonl', selected_cases)
    write_json(out / 'construction.json', {k: v for k, v in construction.items() if k != 'items'})
    (out / 'prompts').mkdir()
    for name, text in prompts.snapshot().items():
        (out / 'prompts' / (name + '.txt')).write_text(text)
    write_json(out / 'assessment.json', config)
    write_json(out / 'protected_sources.json', protected)
    write_jsonl(out / 'records.jsonl', [{'item_id': i['item_id'], 'V0': None, 'V1': None} for i in items])
    write_jsonl(out / 'extractions.jsonl', [{'source_item_id': g, 'extraction': None} for g in groups])
    frozen = {str(p.relative_to(out)): sha256_file(p) for p in out.rglob('*') if p.is_file() and p.name not in ('records.jsonl', 'extractions.jsonl')}
    code_paths = sorted(Path(__file__).resolve().parent.glob('*.py'))
    manifest = {'diagnostic_version': 'necessity-preservation-diagnostic-v1', 'created_at': utc_now(),
        'run_id': run_id, 'proof_count': 27, 'denominators': expected, 'extraction_groups': 11,
        'planned_calls': 65, 'max_http_requests': 70, 'max_output_tokens': config['verification_max_tokens'],
        'model_identity': identity, 'frozen_files': frozen,
        'code_sources': {str(p.resolve()): sha256_file(p) for p in code_paths},
        'template_sources': {str((root / 'prompts' / (n+'.txt')).relative_to(root)): sha256_file(root / 'prompts' / (n+'.txt')) for n in TEMPLATES},
        'readings': READINGS, 'knowledge': config['knowledge'],
        'execution_order': 'V0: proof_set order; extraction: sorted source_item_id; V1: proof_set order',
        'prior_exposure': 'Codex authored the synthetic planted defects, knows all construction labels and has seen historical results/references. Comparison is against sealed construction labels, not independent human judgment.',
        'failure_policy': 'No retries, resampling or prompt edits. Preserve malformed raw output. Technical failures count as not rejected by that arm for defects and rejected for clean proofs. Operational conjunction requires all passes; rejection leaves direction unresolved, never the opposite label.',
        'parsing': 'Strict json.loads of raw completion; no repair, fence removal or normalization. Raw text retained even on malformed/schema failures.',
        'V0_comparison': 'V0 alone is the model check; deterministic checks are logged separately and required by the conjunction, including explicit option conflicts in V0 recomputed answers.',
        'density_manipulation': 'Within each synthetic pair, identical factual cores, answers and violation; elaborate adds only nonfactual explanation/repetition. Witness reason is hidden from both V0 and V1 under the existing V0 input contract; elaboration is placed in scenarios.',
        'scope': 'Diagnostic only; no necessity labels, generation, four-condition answering, BDR, comprehensibility score, established metrics or default pipeline changes.'}
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'seal.json', {'sealed_at': utc_now(), 'manifest_sha256': sha256_file(out / 'manifest.json'),
        'proof_set_sha256': sha256_file(out / 'proof_set.jsonl'), 'frozen_files': frozen})
    return {'run_id': run_id, 'sealed': True, 'proofs': 27, 'planned_calls': 65, 'hard_http_cap': 70}


def load_sealed(root, out):
    manifest = json.loads((out / 'manifest.json').read_text())
    seal = json.loads((out / 'seal.json').read_text())
    if sha256_file(out / 'manifest.json') != seal['manifest_sha256']:
        raise ValueError('sealed manifest changed')
    for path, h in manifest['frozen_files'].items():
        if sha256_file(out / path) != h:
            raise ValueError('sealed file changed: '+path)
    for path, h in {**manifest['code_sources'], **manifest['template_sources']}.items():
        if sha256_file(root / path) != h:
            raise ValueError('sealed implementation/template changed: '+path)
    for path, h in json.loads((out / 'protected_sources.json').read_text()).items():
        if sha256_file(root / path) != h:
            raise ValueError('protected historical/definition file changed: '+path)
    return manifest, read_jsonl(out / 'proof_set.jsonl'), {c['case_id']: c for c in read_jsonl(out / 'cases.jsonl')}


def run_diagnostic(*, run_id, project_root='.', output_root='outputs', max_requests=70, dry_run=False, provider=None):
    root = Path(project_root).resolve(); _validate_run_id(run_id); out = root / output_root / run_id
    manifest, items, cases = load_sealed(root, out)
    if not 1 <= max_requests <= manifest['max_http_requests']:
        raise ValueError('cumulative HTTP cap must be between 1 and 70')
    records = read_jsonl(out / 'records.jsonl'); extractions = read_jsonl(out / 'extractions.jsonl')
    if len(records) != 27 or {r['item_id'] for r in records} != {i['item_id'] for i in items}:
        raise ValueError('fixed record slots changed')
    groups = {i['source_item_id'] for i in items}
    if len(extractions) != 11 or {e['source_item_id'] for e in extractions} != groups:
        raise ValueError('fixed extraction slots changed')
    pending = sum(r[a] is None for r in records for a in ('V0', 'V1')) + sum(e['extraction'] is None for e in extractions)
    if dry_run or not pending:
        return {'pending': pending, 'new_steps': 0, 'hard_http_cap': max_requests}
    provider = provider or JudgeProvider.from_env(root=root, run_dir=out, model='gpt-6-luna', max_requests=max_requests,
                      max_tokens=manifest['max_output_tokens'], visual=True)
    if getattr(provider, 'identity', {'model': provider.model}) != manifest['model_identity']:
        raise ValueError('sealed model identity changed')
    if isinstance(provider, JudgeProvider):
        provider.max_attempts = 1
        provider.max_requests = max_requests
    prompts = PromptSet(out / 'prompts', names=TEMPLATES)
    caller = ModelCaller(provider=provider, prompts=prompts, run_dir=out,
                        protocol_version=manifest['diagnostic_version'], max_attempts=1)
    by_item = {i['item_id']: i for i in items}; steps = 0
    def execute(arm, item, constraints=None):
        nonlocal steps
        c = dict(cases[item['case_id']]); c['image_path'] = str(root / c['image_path'])
        kwargs = call_inputs(c, item['removed'], arm, manifest['knowledge'],
                             proof=item['candidate'], constraints=constraints)
        ident = item['source_item_id'] if arm == 'extract' else item['item_id']
        operation = stable_id('diagnostic', run_id, arm, ident)
        call = call_model(caller, operation_id=operation, attempt_id=operation, role=arm,
                          task='assess_quality', output_format='text', nonce=operation, **kwargs)
        output, error = strict_output(call)
        result = {'call_id': call['call_id'], 'status': call['status'], 'output': output, 'parse_error': error}
        if arm == 'extract':
            result['valid'] = error is None and valid_extraction(output)
        else:
            result['result'] = arm_result(arm, output, constraints) if error is None else 'technical_failure'
        steps += 1
        print(json.dumps({'stage': arm, 'step': steps, 'status': result.get('result', result.get('valid'))}), flush=True)
        return result
    with file_lock(out / '.diagnostic.lock'):
        for r in read_jsonl(out / 'records.jsonl'):
            if r['V0'] is None:
                r['V0'] = execute('V0', by_item[r['item_id']])
                atomic_upsert(out / 'records.jsonl', 'item_id', r, allow_insert=False)
        for e in read_jsonl(out / 'extractions.jsonl'):
            if e['extraction'] is None:
                item = next(i for i in items if i['source_item_id'] == e['source_item_id'])
                e['extraction'] = execute('extract', item)
                atomic_upsert(out / 'extractions.jsonl', 'source_item_id', e, allow_insert=False)
        ext = {e['source_item_id']: e['extraction'] for e in read_jsonl(out / 'extractions.jsonl')}
        for r in read_jsonl(out / 'records.jsonl'):
            if r['V1'] is None:
                item = by_item[r['item_id']]; e = ext[item['source_item_id']]
                constraints = e['output'] if e['valid'] else {'constraints': [], 'referring_expressions': []}
                r['V1'] = execute('V1', item, constraints)
                if not e['valid']:
                    r['V1']['result'] = 'technical_failure'
                    r['V1']['dependency_error'] = 'extraction unavailable or malformed'
                r['V1']['extraction_call_id'] = e['call_id']
                atomic_upsert(out / 'records.jsonl', 'item_id', r, allow_insert=False)
    return {'new_steps': steps, 'pending': 0}


def report_diagnostic(*, run_id, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve(); _validate_run_id(run_id); out = root / output_root / run_id
    manifest, items, cases = load_sealed(root, out)
    records = {r['item_id']: r for r in read_jsonl(out / 'records.jsonl')}
    ext = {e['source_item_id']: e['extraction'] for e in read_jsonl(out / 'extractions.jsonl')}
    if any(records[i['item_id']][a] is None for i in items for a in ('V0', 'V1')) or any(e is None for e in ext.values()):
        raise ValueError('finish all planned slots before final report')
    calls = read_jsonl(out / 'calls.jsonl'); by_call = {c['call_id']: c for c in calls}
    if len(calls) != 65 or len(by_call) != 65:
        raise ValueError('expected exactly 65 logical call records')
    isolation = []
    for item in items:
        c = dict(cases[item['case_id']]); c['image_path'] = str(root / c['image_path'])
        e = ext[item['source_item_id']]
        constraints = e['output'] if e['valid'] else {'constraints': [], 'referring_expressions': []}
        for arm, record in [('V0', records[item['item_id']]['V0']), ('extract', e), ('V1', records[item['item_id']]['V1'])]:
            expected = call_inputs(c, item['removed'], arm, manifest['knowledge'], proof=item['candidate'], constraints=constraints)
            actual = by_call[record['call_id']]['actual_input']
            expected_images = [] if expected['image_path'] is None else [{'path': expected['image_path'], 'sha256': c['image_sha256'], 'detail': 'high'}]
            if (actual['prompt'] != PromptSet(out / 'prompts', names=TEMPLATES).render(expected['template'], **expected['variables']) or
                actual['text'] != expected['text'] or actual['question'] != expected['question'] or actual['images'] != expected_images or
                actual['history'] != [] or actual['condition'] != expected['condition']):
                raise ValueError('logged actual input violates allowlist')
            isolation.append(record['call_id'])
    rows = []
    for i in items:
        r = records[i['item_id']]; code = code_gate(i['candidate'], cases[i['case_id']].get('choices'), r['V0']['output'])
        v0, v1 = r['V0']['result'], r['V1']['result']
        rows.append({'item_id': i['item_id'], 'source_item_id': i['source_item_id'], 'kind': i['candidate']['kind'],
                     'version': i['version'], 'sealed_label': i['sealed_label'], 'code': code, 'V0': v0, 'V1': v1,
                     'conjunction': conjunction(code, v0, v1),
                     'V0_counted_rejected': counted_rejection(i['sealed_label'], v0),
                     'conjunction_counted_rejected': counted_rejection(i['sealed_label'], code, v0, v1)})
    defects = [r for r in rows if r['sealed_label'] != 'clean']; clean = [r for r in rows if r['sealed_label'] == 'clean']
    counts = {'V0_defects_rejected_of_19': sum(r['V0_counted_rejected'] for r in defects),
              'conjunction_defects_rejected_of_19': sum(r['conjunction_counted_rejected'] for r in defects),
              'conjunction_clean_rejected_of_8': sum(r['conjunction_counted_rejected'] for r in clean)}
    for version in ('defect-terse', 'defect-elaborate'):
        counts['V0_missed_'+version+'_of_8'] = sum(not r['V0_counted_rejected'] for r in rows if r['version'] == version)
    gain = counts['conjunction_defects_rejected_of_19'] - counts['V0_defects_rejected_of_19']
    delta = counts['V0_missed_defect-elaborate_of_8'] - counts['V0_missed_defect-terse_of_8']
    reading = ('do not adopt' if counts['conjunction_clean_rejected_of_8'] >= 3 else
               'keep current design' if counts['V0_defects_rejected_of_19'] >= 17 else
               'adoption threshold met; user decides' if gain >= 4 and counts['conjunction_clean_rejected_of_8'] <= 1 else 'inconclusive')
    density = 'H-density has directional support' if delta >= 3 else 'postpone density concern for proofs' if delta <= 1 else 'H-density inconclusive'
    usage = call_usage(calls)
    requests = read_jsonl(out / 'requests.jsonl'); http = sum(r['event'] == 'started' for r in requests)
    if http > 70:
        raise ValueError('cumulative HTTP cap exceeded')
    usage.update(actual_http_requests=http, hard_http_cap=70, planned_calls=65,
                 completed_logical_calls=len(calls), known_tokens_only=True)
    write_json(out / 'usage.json', usage)
    write_json(out / 'results.json', {'rows': rows, 'counts': counts, 'additional_defects_rejected': gain,
               'density_miss_difference': delta, 'reading': reading, 'density_reading': density,
               'warning': 'Diagnostic construction labels only. No necessity labels or established metrics.'})
    write_json(out / 'acceptance.json', {'logged_inputs_reconstructed': len(set(isolation)),
               'extraction_inputs_candidate_free_by_allowlist': True, 'V1_witness_inputs_scenarios_only': True,
               'no_labels_or_V0_outputs_in_any_model_input': True,
               'protected_sources_unchanged': True, 'sealed_files_unchanged': True,
               'http_cap_satisfied': http <= 70, 'raw_outputs_preserved': True})
    lines = ['# Sealed necessity-evidence preservation diagnostic', '',
        'Construction controls only: 8 clean, 16 **synthetic planted defects**, 3 historical real defects. '+manifest['prior_exposure'], '',
        'No necessity labels were produced. Rejection leaves that direction unresolved; it never supplies the opposite label. Existing definitions, protocols, historical runs and metrics are unchanged. The default pipeline has not been changed.', '',
        'Seal: `seal.json`; UTC timestamp '+json.loads((out/'seal.json').read_text())['sealed_at']+'. Proof-set SHA256 `'+sha256_file(out/'proof_set.jsonl')+'`. Frozen prompts and K are in `prompts/` and `assessment.json`; selection, edits and historical checks are in `construction.json` and `proof_set.jsonl`.', '',
        '| item | kind | version | sealed label | V0 result | V1 result | conjunction result |',
        '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| '+' | '.join(str(r[k]) for k in ('item_id','kind','version','sealed_label','V0','V1','conjunction'))+' |')
    lines += ['', 'Pre-registered readings (frozen verbatim before calls):', ''] + ['- '+s for s in manifest['readings']]
    lines += ['', 'The four decision quantities:', '',
        f"- V0 rejected **{counts['V0_defects_rejected_of_19']}/19** defective proofs.",
        f"- Conjunction rejected **{counts['conjunction_defects_rejected_of_19']}/19**, **{gain}** additional defects.",
        f"- Conjunction falsely rejected **{counts['conjunction_clean_rejected_of_8']}/8** sealed-clean proofs.",
        f"- V0 missed terse **{counts['V0_missed_defect-terse_of_8']}/8**, elaborate **{counts['V0_missed_defect-elaborate_of_8']}/8**; elaborate-minus-terse **{delta}**.", '',
        '**Applicable reading: '+reading+'.** '+density+'. These are directional counts, not rates or accuracy estimates; adoption remains the user’s decision.', '',
        'Operational acceptance requires code checks, V0 and V1 all pass. Technical/schema failures are shown as technical_failure; they count as not rejected by that arm for defective proofs and rejected for clean proofs. Other arms can still supply a confirmed rejection. No retries, resampling or output repair were used.', '',
        'V0 accepted defects (exact V0 reason beside the sealed violated constraint):', '']
    for i, r in zip(items, rows):
        if i['sealed_label'] != 'clean' and r['V0'] == 'accept':
            lines += [f"- **{i['item_id']}** ({i['sealed_label']}): violated constraint “{i['violation']['quote']}” — {i['violation']['constraint']}. V0 reason: “{records[i['item_id']]['V0']['output']['reason']}”"]
    if not any(r['V0'] == 'accept' for r in defects):
        lines += ['None.']
    lines += ['', 'Clean rejections and extraction audit:', '']
    for r in clean:
        if r['conjunction_counted_rejected']:
            record = records[r['item_id']]
            lines += [f"- {r['item_id']}: code={r['code']}, V0={r['V0']}, V1={r['V1']}. V0: {canonical_json(record['V0']['output'])}. V1: {canonical_json(record['V1']['output'])}. Extraction: `{r['source_item_id']}` in `extractions.jsonl`."]
    if not any(r['conjunction_counted_rejected'] for r in clean):
        lines += ['No sealed-clean proof rejected. Extraction outputs remain inspectable in `extractions.jsonl`.']
    lines += ['', 'Actual use: **'+str(http)+' HTTP requests**, '+str(len(calls))+' logical calls; model `'+manifest['model_identity']['model']+'`; frozen output limit '+str(manifest['max_output_tokens'])+'. Known tokens: **'+str(usage['known_total_tokens'])+'** (input '+str(usage['known_input_tokens'])+', output '+str(usage['known_output_tokens'])+'); usage_complete='+str(usage['usage_complete'])+'. Request elapsed time '+str(round(usage['elapsed_seconds'],3))+' seconds; returned cost '+str(usage['cost'])+' (unknown is not zero). Full accounting: `usage.json`.',
        '', 'Isolation was reconstructed from every logged actual input: extraction contains Q/options, retained material and K only; V1 witnesses contain extracted constraints and the two scenarios only; no model input includes sealed labels, reference answers, historical reviews or V0 output. Each call has an empty history. See `acceptance.json`.', '',
        'What cannot be concluded: this small exposed construction set does not estimate population accuracy, natural failure frequency, necessity labels or any method’s BDR. The clean labels are inherited supported reviews, not independent human adjudication. The same model underlies both arms; no verifier independence is assumed. Any gain supports the utility of separation on these controls, but cannot isolate framing as its causal mechanism. Density uses only eight matched synthetic pairs, and fluency/length is not independently randomized. A floor in V0 misses cannot rule out density effects elsewhere. Extraction may omit or invent constraints; a conjunction can reduce coverage even when it removes false accepts. No larger batch or pipeline adoption follows from this report.', '']
    (out / 'probe_report.md').write_text('\n'.join(lines))
    return {'counts': counts, 'reading': reading, 'density_reading': density, 'usage': usage}
