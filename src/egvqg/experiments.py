"""Versioned prompt experiments: planned records, resumable steps and human handoff."""
from __future__ import annotations

import copy
import html
import json
import re
import sys
import unicodedata
from pathlib import Path

from .pipeline import _validate_run_id, load_materials
from .prompts import PromptSet
from .providers import JudgeProvider, ModelCaller, call_model
from .records import (CONDITIONS, ExperimentStore, canonical_json, file_lock, read_jsonl,
                      sha256_file, sha256_text, stable_id, utc_now, write_json, write_jsonl, material_metadata)

DIMENSIONS = ('validity', 'relatedness', 'text_necessary', 'image_necessary', 'integration')
SHORTCUTS = ('text_only', 'image_only', 'common_knowledge', 'question_leakage', 'independent_stitching')
TEMPLATES = ('plain-v1', 'strong-v1', 'answer-v1', 'solve-v1', 'score-v1',
             'relatedness-v1', 'necessity-v1', 'shortcut-v1', 'extract-text-v1',
             'extract-image-v1', 'pair-v1', 'generate-anchored-v1',
             'plan-no-image-anchor-v1', 'generate-no-image-anchor-v1', 'verify-answer-v1')
METHODS = ('plain', 'strong', 'anchored', 'no_image_anchor')
STEPS = ('generation', 'candidate_answer', 'independent_answer', 'reference_resolution',
         'answer_TI', 'answer_T', 'answer_I', 'answer_0',
         'score_TI', 'score_T', 'score_I', 'score_0', 'relatedness', 'necessity', 'shortcuts')
# Stage names select steps for partial execution; unselected steps stay pending.
STAGES = {'generation': ('generation',),
          'reference': ('candidate_answer', 'independent_answer', 'reference_resolution'),
          'answers': ('answer_TI', 'answer_T', 'answer_I', 'answer_0'),
          'scores': ('score_TI', 'score_T', 'score_I', 'score_0'),
          'audit': ('relatedness', 'necessity', 'shortcuts')}
# Optional config key reference_policy. Absent = the historical behaviour (a second
# identical same-model request). 'review_gated' = one candidate answer, then an explicit
# reviewer decision before any scoring or reference-dependent audit.
REVIEW_GATED = 'review_gated'
GATED_STEPS = (*STAGES['scores'], 'necessity', 'shortcuts')
REFERENCE_DECISIONS = ('accept_candidate', 'corrected', 'unresolved', 'invalid_question')


def _gated(config):
    policy = config.get('reference_policy')
    if policy not in (None, REVIEW_GATED):
        raise ValueError('unknown reference_policy')
    return policy == REVIEW_GATED


def prompt_names(config):
    """Resolve opt-in template versions without changing historical snapshots."""
    overrides = config.get('prompt_overrides', {})
    if not isinstance(overrides, dict) or not set(overrides) <= set(TEMPLATES):
        raise ValueError('prompt overrides must name existing template roles')
    names = tuple(overrides.get(name, name) for name in TEMPLATES)
    if len(set(names)) != len(names) or any(not isinstance(n, str) or not re.fullmatch(r'[a-z0-9-]+', n) for n in names):
        raise ValueError('prompt versions must be unique safe template names')
    return names


def empty_labels(keys=DIMENSIONS):
    return {key: {'value': None, 'status': 'pending', 'reason': None} for key in keys}


def label(value, reason, *, source='automatic'):
    if value is not None and type(value) not in (int, bool):
        raise ValueError('label must be 0/1/null')
    if value not in (None, 0, 1):
        raise ValueError('label outside 0/1/null')
    return {'value': None if value is None else int(value),
            'status': 'undecided' if value is None else 'decided', 'reason': reason, 'source': source}


def new_record(slot, material, config, prompt_hash, imported_question=None):
    q = normalize_question(imported_question) if imported_question else None
    qid = stable_id('q', slot['attempt_id'], q) if q else None
    return {**slot, 'schema_version': 'experiment-record-v1',
        'protocol_version': config['protocol_version'], 'status': 'pending',
        'material': material, 'model_config_hash': None, 'prompt_snapshot_sha256': prompt_hash,
        'prompt_versions': dict(zip(TEMPLATES, prompt_names(config))),
        'generation': {'status': 'imported' if q else 'pending', 'raw_output': imported_question,
                       'call_ids': [], 'technical_attempts': 0, 'failure_code': None,
                       'text_evidence': None, 'image_evidence': None, 'pairing': None},
        'question': {'question_id': qid, 'normalized_text': q,
                     'sha256': sha256_text(q) if q else None, 'normalization_version': 'q-normalize-v1'},
        'reference': {'question_id': qid, 'question_sha256': sha256_text(q) if q else None,
                      'candidate_answer': None, 'independent_answer': None, 'verified_answer': None,
                      'acceptable_answers': [], 'rubric': [], 'text_evidence': [], 'image_evidence': [],
                      'status': 'pending', 'verification_source': None, 'call_ids': [], 'review_ids': []},
        'responses': {c: None for c in CONDITIONS},
        'labels': {source: empty_labels() for source in ('automatic', 'human', 'adjudicated')},
        'shortcuts': {source: empty_labels(SHORTCUTS) for source in ('automatic', 'human', 'adjudicated')},
        'bdr': {'value': None, 'status': 'pending', 'reason': None},
        'steps': {step: ('done' if step == 'generation' and q else 'pending') for step in STEPS},
        'step_outputs': {}, 'usage': {'remote_requests': 0, 'cache_hits': 0,
            'elapsed_seconds': None, 'input_tokens': None, 'output_tokens': None,
            'total_tokens': None, 'cost': None, 'currency': None, 'usage_complete': False},
        'notes': [], 'created_at': utc_now(), 'updated_at': utc_now()}



def normalize_question(q):
    return unicodedata.normalize('NFC', q).strip()


def init_experiment(*, materials_path, run_id, methods=('plain',), cases_path=None,
                    partition='dev', stage=0, project_root='.', output_root='outputs',
                    config_path='configs/experiments.json', prompt_overrides=None):
    root = Path(project_root).resolve(); out = root / output_root / run_id
    _validate_run_id(run_id)
    config = json.loads((root / config_path).read_text())
    if prompt_overrides:
        config['prompt_overrides'] = {**config.get('prompt_overrides', {}), **prompt_overrides}
    prompts = PromptSet(root / 'prompts', names=prompt_names(config))
    for original, version in zip(TEMPLATES, prompt_names(config)):
        if version != original and set(prompts.variables(version)) != set(
                PromptSet(root / 'prompts', names=(original,)).variables(original)):
            raise ValueError('replacement template variables must match its role')
    materials = load_materials(root, root / materials_path)
    if len(methods) != len(set(methods)) or any(m not in METHODS for m in methods):
        raise ValueError('methods must be unique defined methods')
    # Comparison materials must be frozen by the dedicated admission workflow.
    if partition == 'pilot':
        manifest = json.loads((root / materials_path).with_name('manifest.json').read_text())
        if manifest.get('status') not in ('frozen_human_reviewed', 'frozen_delegated_reviewed'):
            raise ValueError('pilot materials require completed human admission and freeze')
        if manifest['materials_sha256'] != sha256_file(root / materials_path):
            raise ValueError('frozen materials changed')
    by_id = {m['material_id']: m for m in materials}
    cases = read_jsonl(root / cases_path) if cases_path else []
    if cases_path and (not cases or len({c['case_id'] for c in cases}) != len(cases)):
        raise ValueError('import requires unique nonempty cases')
    specs = [(c['material_id'], 'imported', c['case_id'], c['question']) for c in cases] if cases_path else [
        (m['material_id'], method, '', None) for m in materials for method in methods]
    slots, records = [], []
    for index, (mid, method, case_id, q) in enumerate(specs):
        if mid not in by_id or (q is not None and not q.strip()):
            raise ValueError('case material missing or Q empty')
        material = material_metadata(by_id[mid]); material['experiment_partition'] = partition
        material['text_sha256'] = sha256_text(material['text'])
        slot = {'attempt_id': stable_id('attempt', run_id, mid, method, case_id),
                'run_id': run_id, 'stage': 3 if partition == 'pilot' and method in ('anchored','no_image_anchor') else stage,
                'plan_index': index, 'material_id': mid,
                'method': method, 'case_id': case_id or None, 'source_run_id': None,
                'source_attempt_id': None}
        slots.append(slot); records.append(new_record(slot, material, config, prompts.snapshot_sha256(), q))
    identity = {'config': config, 'materials': materials, 'plan': slots,
                'imported_questions': [r['question'] for r in records],
                'partition': partition, 'prompts': prompts.snapshot()}
    fingerprint = sha256_text(canonical_json(identity))
    with file_lock(out / '.init.lock'):
        manifest_path = out / 'manifest.json'
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text())
            if prior['fingerprint'] != fingerprint:
                raise ValueError('run material/config/prompt/plan changed; use a new run_id')
            ExperimentStore(out).rows()
            return {'run_dir': str(out), 'n_planned': len(slots), 'resumed': True}
        # Plan and blank records precede manifest: recovery from partial initialization
        # deterministically rebuilds the same blanks before any model call is allowed.
        write_jsonl(out / 'plan.jsonl', slots); write_jsonl(out / 'records.jsonl', records)
        write_jsonl(out / 'materials.jsonl', materials)
        for name, text in prompts.snapshot().items():
            target = out / 'prompts' / f'{name}.txt'; target.parent.mkdir(exist_ok=True)
            target.write_text(text)
        write_json(out / 'manifest.json', {'fingerprint': fingerprint,
            'protocol_version': config['protocol_version'], 'plan_sha256': sha256_text(canonical_json(slots)),
            'n_planned': len(slots), 'config': config, 'partition': partition,
            'prompt_sha256': prompts.snapshot_sha256(), 'created_at': utc_now(),
            'evaluation_status': 'draft_until_human_review', 'model_identity': None,
            'environment': {'python': sys.version,
                'lock_sha256': sha256_file(root/'uv.lock') if (root/'uv.lock').exists() else None,
                'source_sha256': {p.name: sha256_file(p) for p in sorted((root/'src/egvqg').glob('*.py'))}},
            'model_limitation': 'Greedy settings and fresh-request repeatability require real validation.'})
    export_review(out, root)
    return {'run_dir': str(out), 'n_planned': len(slots), 'resumed': False}


def _json(value):
    return json.dumps(value, ensure_ascii=False)


def answering_input(record, condition, root):
    if condition not in CONDITIONS:
        raise ValueError('unknown condition')
    m = record['material']; q = record['question']['normalized_text']
    visible = '<T>' + m['text'] + '</T>' if condition in ('TI', 'T') else ''
    return {'variables': {'VISIBLE_MATERIALS': visible, 'Q': q},
            'text': m['text'] if condition in ('TI', 'T') else None,
            'image_path': str(root / m['image_path']) if condition in ('TI', 'I') else None,
            'question': q, 'condition': condition}


def _strict_labels(data, keys):
    result = {}
    for key in keys:
        item = data[key]
        if not isinstance(item, dict) or not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError('missing label reason')
        result[key] = {**item, **label(item['value'], item['reason'])}
        if item.get('status') == 'not_applicable' and item['value'] is None:
            result[key]['status'] = 'not_applicable'
    return result


def _validate_answer(data):
    if data.get('status') not in ('answerable', 'ambiguous', 'unsupported', 'unreadable'):
        raise ValueError('invalid answer status')
    if data['status'] == 'answerable':
        if not isinstance(data.get('answer'), str) or not data['answer'].strip():
            raise ValueError('missing answer')
        if not isinstance(data.get('rubric'), list) or not data['rubric'] or not all(
                isinstance(x, str) for x in data['rubric']):
            raise ValueError('missing rubric')
    for key in ('acceptable_answers', 'text_evidence', 'image_evidence'):
        if not isinstance(data.get(key), list):
            raise ValueError(f'missing {key}')
    return data


def _reference_context(record):
    ref = record['reference']
    # Candidate answer and independently solved result are provided only to REVIEW,
    # never to generation or condition answering.
    return _json({k: ref[k] for k in ('status', 'verification_source', 'candidate_answer',
                                    'independent_answer', 'verified_answer', 'rubric')})


def _normalized_answer(value):
    return ' '.join(unicodedata.normalize('NFC', value).split()).casefold()


def _invoke(caller, record, step, template, variables, *, task='assess_quality',
            text=None, image_path=None, question=None, condition=None, output_format='json', nonce=None):
    if nonce is None and record.get('evaluation_revision',0):
        nonce=f"explicit_evaluation_revision:{record['evaluation_revision']}"
    template = record['prompt_versions'].get(template, template)
    return call_model(caller, operation_id=stable_id('operation', record['attempt_id'], step, record.get('evaluation_revision', 0)),
        attempt_id=record['attempt_id'], role=step, template=template, variables=variables,
        task=task, text=text, image_path=image_path, question=record['question']['normalized_text'],
        condition=condition, output_format=output_format, nonce=nonce)


def _generate(record, caller, root):
    m = record['material']; text = m['text']; image = str(root / m['image_path'])
    method = record['method']; results = []
    def ask(step, template, variables, *, include_text=True, include_image=True, fmt='json'):
        result = _invoke(caller, record, step, template, variables, task='generate_question',
                         text=text if include_text else None, image_path=image if include_image else None,
                         output_format=fmt)
        results.append(result)
        if result['status'] != 'ok':
            raise ValueError(result.get('error') or result['status'])
        return result['parsed_output']
    try:
        if method in ('plain', 'strong'):
            q = ask('generation', method+'-v1', {'T': text}, fmt='text')
        else:
            te = ask('extract_text', 'extract-text-v1', {'T': text}, include_image=False)
            if not isinstance(te.get('text_evidence'), list):
                raise ValueError('invalid text evidence')
            record['generation']['text_evidence'] = te
            if method == 'anchored':
                ie = ask('extract_image', 'extract-image-v1', {}, include_text=False)
                if not isinstance(ie.get('image_evidence'), list):
                    raise ValueError('invalid image evidence')
                record['generation']['image_evidence'] = ie
                pair = ask('pair', 'pair-v1', {'T': text, 'TEXT_EVIDENCE': _json(te), 'IMAGE_EVIDENCE': _json(ie)})
                record['generation']['pairing'] = pair
                if pair.get('status') != 'ready' or any(pair.get(k, {}).get('passed') is not True
                        for k in ('redundancy_check', 'common_knowledge_check', 'integration_check')):
                    raise ValueError('no_approved_pair')
                for key, evidence, ids in (('text_ids', te['text_evidence'], 'text_ids'),
                                           ('image_ids', ie['image_evidence'], 'image_ids')):
                    if not pair.get(ids) or not set(pair[ids]) <= {e['id'] for e in evidence}:
                        raise ValueError('pair references missing evidence')
                q = ask('generation', 'generate-anchored-v1', {'T': text,
                    'TEXT_EVIDENCE': _json([e for e in te['text_evidence'] if e['id'] in pair['text_ids']]),
                    'IMAGE_EVIDENCE': _json([e for e in ie['image_evidence'] if e['id'] in pair['image_ids']]),
                    'PAIRING': _json(pair)}, fmt='text')
            else:
                pair = ask('pair', 'plan-no-image-anchor-v1', {'T': text, 'TEXT_EVIDENCE': _json(te)})
                record['generation']['pairing'] = pair
                if pair.get('status') != 'ready' or not pair.get('text_ids') or not set(pair['text_ids']) <= {
                        e['id'] for e in te['text_evidence']}:
                    raise ValueError('no_approved_text_target')
                q = ask('generation', 'generate-no-image-anchor-v1', {'T': text,
                    'TEXT_EVIDENCE': _json([e for e in te['text_evidence'] if e['id'] in pair['text_ids']]),
                    'PAIRING': _json(pair)}, fmt='text')
        q = normalize_question(q)
        if not q or q == 'NO_QUESTION' or '\n' in q or re.search(r'(?im)^(answer|答案)\s*[:：]', q):
            raise ValueError('no_single_question')
        record['question'] = {'question_id': stable_id('q', record['attempt_id'], q),
            'normalized_text': q, 'sha256': sha256_text(q), 'normalization_version': 'q-normalize-v1'}
        record['reference'].update(question_id=record['question']['question_id'], question_sha256=sha256_text(q))
        record['generation'].update(status='ok', raw_output=results[-1]['raw_output'])
    except (ValueError, KeyError, TypeError) as exc:
        record['generation'].update(status='technical_failure' if results and results[-1]['status'] == 'technical_failure'
                                    else 'generation_failure', failure_code=str(exc),
                                    raw_output=results[-1].get('raw_output') if results else None)
    record['generation']['call_ids'] = [r['call_id'] for r in results]
    record['steps']['generation'] = 'done' if record['question']['normalized_text'] else 'failed'


def _awaiting_review(record, gated):
    return bool(gated and record['question']['normalized_text'] and not record['reference']['review_ids'])


def run_experiment(*, run_id, project_root='.', output_root='outputs', max_requests=180,
                   max_steps=None, dry_run=False, provider=None, methods=None, stages=None,
                   attempt_ids=None):
    root = Path(project_root).resolve(); _validate_run_id(run_id); out = root / output_root / run_id
    store = ExperimentStore(out); manifest = store.manifest; config = manifest['config']
    gated = _gated(config)
    if stages is not None and (not stages or any(s not in STAGES for s in stages)):
        raise ValueError('stages must be chosen from ' + ', '.join(STAGES))
    selected = set(STEPS) if stages is None else {step for s in stages for step in STAGES[s]}
    if attempt_ids is not None and (not attempt_ids or not set(attempt_ids) <= store.ids):
        raise ValueError('attempt_ids must be nonempty planned attempts')
    # Unselected attempts keep their pending steps and stay in every denominator as unresolved.
    chosen = store.ids if attempt_ids is None else set(attempt_ids)
    if manifest['partition']=='pilot':
        from .batch import validate_batch_contract
        validate_batch_contract(out,root)
    enabled = set(methods) if methods is not None else (
        {'plain','strong'} if manifest['partition'] == 'pilot' else {r['method'] for r in store.plan})
    if not enabled or not enabled <= {r['method'] for r in store.plan}:
        raise ValueError('enabled methods must belong to the frozen plan')
    prompts = PromptSet(out / 'prompts', names=prompt_names(config))
    if prompts.snapshot_sha256() != manifest['prompt_sha256']:
        raise ValueError('prompt snapshot changed')
    # Validate actual files on every resume, not just the initialization snapshot.
    load_materials(root, out / 'materials.jsonl')
    if dry_run:
        # Upper bound on logical model calls for the selected pending steps. Anchored
        # generation is up to 4 calls (3 without image anchors); gated runs never issue
        # the duplicate reference request. Refusals/unverified references lower the real count.
        pending = {}; calls = 0
        for r in store.rows():
            if r['method'] not in enabled or r['attempt_id'] not in chosen:
                continue
            for step, state in r['steps'].items():
                if state != 'pending' or step not in selected:
                    continue
                pending[step] = pending.get(step, 0) + 1
                if step == 'generation':
                    calls += {'anchored': 4, 'no_image_anchor': 3}.get(r['method'], 1)
                elif not (gated and step in ('independent_answer', 'reference_resolution')):
                    calls += 1
        return {'n_planned': len(store.plan), 'max_http_attempts': max_requests,
                'enabled_methods': sorted(enabled), 'stages': sorted(stages) if stages else 'all',
                'reference_policy': config.get('reference_policy', 'same_model_second_request'),
                'pending_steps': sum(pending.values()), 'pending_by_step': pending,
                'max_logical_calls': calls,
                'max_http_with_technical_retries': calls * config['max_technical_attempts'],
                'awaiting_reference_review': sum(_awaiting_review(r, gated) and r['steps']['candidate_answer'] != 'pending'
                                                 for r in store.rows() if r['method'] in enabled),
                'max_output_tokens': config['max_output_tokens'], 'estimated_cost': None}
    provider = provider or JudgeProvider.from_env(root=root, run_dir=out, visual=True,
        max_requests=max_requests, max_tokens=config['max_output_tokens'], env_prefix=config['env_prefix'])
    if isinstance(provider, JudgeProvider):
        provider.max_attempts = 1
    identity = getattr(provider, 'identity', {'model': provider.model})
    if manifest['partition']=='pilot':
        validate_batch_contract(out,root,identity=identity)
    if manifest['model_identity'] is not None and manifest['model_identity'] != identity:
        raise ValueError('model/effective decoding changed; use a new run_id')
    if manifest['model_identity'] is None:
        manifest['model_identity'] = identity; write_json(out / 'manifest.json', manifest)
    caller = ModelCaller(provider=provider, prompts=prompts, run_dir=out,
                         protocol_version=manifest['protocol_version'], max_attempts=config['max_technical_attempts'],
                         json_repair=config.get('json_repair'))
    completed = 0
    with file_lock(out / '.pipeline.lock'):
        for initial in store.rows():
            if initial['method'] not in enabled or initial['attempt_id'] not in chosen:
                continue
            record = initial
            if record['status'] == 'automatic_complete':
                continue
            if record['steps']['generation'] == 'pending' and 'generation' not in selected:
                continue  # nothing can be evaluated before the planned generation runs
            for step in STEPS:
                if record['steps'][step] != 'pending' or step not in selected:
                    continue
                if max_steps is not None and completed >= max_steps:
                    summarize_experiment(out, root)
                    return {'run_dir': str(out), 'new_steps': completed, 'status': 'paused'}
                previous = copy.deepcopy(record)
                record['status'] = 'running'
                if step == 'generation':
                    _generate(record, caller, root)
                elif not record['question']['normalized_text']:
                    record['steps'][step] = 'not_applicable'
                else:
                    _evaluate_step(record, step, caller, root, **({'gated': True} if gated else {}))
                record['updated_at'] = utc_now()
                record['model_config_hash'] = sha256_text(canonical_json(identity))
                store.put(record, previous=previous); completed += 1
            if any(state == 'pending' for state in record['steps'].values()):
                continue  # partial stage run: finalize only when every step is terminal
            waiting = _awaiting_review(record, gated)
            if waiting and record['status'] == 'awaiting_reference_review':
                continue  # resume before the reviewer decision is a no-op
            if record['status'] != 'automatic_complete':
                previous = copy.deepcopy(record)
                from .metrics import record_states
                value = record_states(record)['bdr']
                record['bdr'] = {'value': value, 'status': 'unresolved' if value is None else ('pass' if value else 'fail'),
                                 'reason': 'V AND C_TI AND NOT C_T AND NOT C_I; refusal remains unresolved',
                                 'source': 'automatic'}
                calls = [c for c in read_jsonl(out/'calls.jsonl') if c['attempt_id'] == record['attempt_id']]
                record['usage'] = call_usage(calls)
                generation_calls = [c for c in calls if c['role'] in ('generation','extract_text','extract_image','pair')]
                record['generation']['call_ids'] = [c['call_id'] for c in generation_calls]
                record['generation']['technical_attempts'] = len(generation_calls)
                record['status'] = 'awaiting_reference_review' if waiting else 'automatic_complete'
                store.put(record, previous=previous)
    summarize_experiment(out, root)
    rows = [r for r in store.rows() if r['method'] in enabled]
    if all(r['status'] == 'automatic_complete' for r in store.rows()):
        status = 'automatic_complete'
    elif any(r['status'] == 'awaiting_reference_review' for r in rows):
        status = 'awaiting_reference_review'
    elif any(state == 'pending' for r in rows for state in r['steps'].values()):
        status = 'selected_stages_complete'
    else:
        status = 'enabled_methods_complete'
    return {'run_dir': str(out), 'new_steps': completed, 'status': status,
            'awaiting_reference_review': sum(r['status'] == 'awaiting_reference_review' for r in rows)}


def _evaluate_step(record, step, caller, root, gated=False):
    m = record['material']; q = record['question']['normalized_text']; ref = record['reference']
    image = str(root / m['image_path']); base = {'T': m['text'], 'Q': q}
    result = None
    try:
        if gated and step in ('independent_answer', 'reference_resolution'):
            # An identical second request to the same model adds no independent evidence.
            record['steps'][step] = 'not_applicable'; return
        if gated and step in GATED_STEPS and not ref['review_ids']:
            # Explicit reference review precedes scoring and reference-dependent audits.
            response = record['responses'].get(step.removeprefix('score_')) if step.startswith('score_') else None
            if response:
                response.update(correct=None, score_status='unresolved', score_reason='awaiting_reference_review')
            record['steps'][step] = 'blocked'; return
        if gated and step == 'candidate_answer':
            result = _invoke(caller, record, step, 'answer-v1', base, task='verify_reference',
                             text=m['text'], image_path=image)
            if result['status'] != 'ok': raise ValueError(result.get('error') or result['status'])
            data = _validate_answer(result['parsed_output']); ref[step] = data
            ref['call_ids'].append(result['call_id'])
            ref.update(status='candidate_unreviewed', verification_source=None)
        elif step in ('candidate_answer', 'independent_answer'):
            result = _invoke(caller, record, step, 'answer-v1', base, task='verify_reference',
                             text=m['text'], image_path=image)
            if result['status'] != 'ok': raise ValueError(result.get('error') or result['status'])
            data = _validate_answer(result['parsed_output']); ref[step] = data
            ref['call_ids'].append(result['call_id'])
            if step == 'independent_answer':
                first = ref['candidate_answer']
                if first and first['status'] == data['status'] == 'answerable' and (
                        _normalized_answer(first['answer']) == _normalized_answer(data['answer'])):
                    ref.update(status='verified_automatic', verified_answer=data['answer'],
                        acceptable_answers=data['acceptable_answers'], rubric=data['rubric'],
                        text_evidence=data['text_evidence'], image_evidence=data['image_evidence'],
                        verification_source='same_model_blind_answer_agreement')
                else:
                    ref.update(status='unresolved', verification_source='automatic_disagreement_or_unanswerable')
        elif step == 'reference_resolution':
            if ref['status'] == 'verified_automatic':
                record['steps'][step] = 'not_applicable'; return
            first, second = ref['candidate_answer'], ref['independent_answer']
            if not first or not second or first['status'] != 'answerable' or second['status'] != 'answerable':
                record['steps'][step] = 'blocked'; return
            result = _invoke(caller, record, step, 'verify-answer-v1', {**base,
                'CANDIDATE': _json(first), 'INDEPENDENT': _json(second)}, task='verify_reference',
                text=m['text'], image_path=image)
            if result['status'] != 'ok': raise ValueError(result.get('error') or result['status'])
            data = result['parsed_output']
            if data.get('status') not in ('verified', 'unresolved'):
                raise ValueError('invalid verification status')
            if data['status'] == 'verified':
                checked = _validate_answer({**data, 'status': 'answerable'})
                ref.update(status='verified_automatic', verified_answer=checked['answer'],
                    acceptable_answers=checked['acceptable_answers'], rubric=checked['rubric'],
                    text_evidence=checked['text_evidence'], image_evidence=checked['image_evidence'],
                    verification_source='same_model_blind_solutions_then_semantic_verification')
            ref['call_ids'].append(result['call_id'])
        elif step.startswith('answer_'):
            c = step.removeprefix('answer_'); payload = answering_input(record, c, root)
            result = _invoke(caller, record, step, 'solve-v1', task='answer_question', **payload)
            if result['status'] != 'ok': raise ValueError(result.get('error') or result['status'])
            data = result['parsed_output']
            if type(data.get('abstain')) is not bool or (not data['abstain'] and not isinstance(data.get('answer'), str)):
                raise ValueError('invalid answer JSON')
            record['responses'][c] = {'response_id': stable_id('response', record['attempt_id'], c),
                'question_id': record['question']['question_id'], 'question_sha256': record['question']['sha256'],
                'condition': c, 'status': 'refused' if data['abstain'] else 'answered',
                'raw_output': result['raw_output'], 'answer': data.get('answer'), 'abstain': data['abstain'],
                'reason_brief': data.get('reason_brief'), 'correct': None, 'score_status': 'pending',
                'score_reason': None, 'answer_call_id': result['call_id'], 'score_call_id': None,
                'input_audit': {'text_included': payload['text'] is not None,
                    'image_included': payload['image_path'] is not None, 'reference_included': False,
                    'generation_evidence_included': False, 'history_included': False}}
        elif step.startswith('score_'):
            c = step.removeprefix('score_'); response = record['responses'][c]
            if ref['status'] not in ('verified', 'verified_automatic') or not response or response['status'] != 'answered':
                if response:
                    response.update(correct=None, score_status='unresolved',
                                    score_reason='reference_unverified_or_refusal')
                record['steps'][step] = 'blocked'; return
            result = _invoke(caller, record, step, 'score-v1', {'Q': q, 'ANSWER': ref['verified_answer'],
                'ACCEPTABLE_ANSWERS': _json(ref['acceptable_answers']), 'RUBRIC': _json(ref['rubric']),
                'RESPONSE': _json({k: response[k] for k in ('answer', 'abstain', 'reason_brief')})}, task='score_response')
            if result['status'] != 'ok': raise ValueError(result.get('error') or result['status'])
            data = result['parsed_output']; checked = label(data['correct'], data['reason'])
            response.update(correct=checked['value'], score_status='scored' if checked['value'] is not None else 'unresolved',
                score_reason=data['reason'], score_call_id=result['call_id'],
                score_reference_sha256=sha256_text(canonical_json({k: ref[k] for k in ('verified_answer', 'acceptable_answers', 'rubric')})))
        elif step == 'relatedness':
            result = _invoke(caller, record, step, 'relatedness-v1', base, text=m['text'], image_path=image)
            if result['status'] != 'ok': raise ValueError(result.get('error') or result['status'])
            data = result['parsed_output']
            record['labels']['automatic']['relatedness'] = {**label(data['relatedness'], data['reason']),
                'text_match': data.get('text_match'), 'image_match': data.get('image_match')}
        elif step == 'necessity':
            result = _invoke(caller, record, step, 'necessity-v1', {**base, 'REFERENCE': _reference_context(record)},
                             text=m['text'], image_path=image)
            if result['status'] != 'ok': raise ValueError(result.get('error') or result['status'])
            record['labels']['automatic'].update(_strict_labels(result['parsed_output'], tuple(k for k in DIMENSIONS if k != 'relatedness')))
        elif step == 'shortcuts':
            result = _invoke(caller, record, step, 'shortcut-v1', {**base, 'REFERENCE': _reference_context(record),
                'FOUR_CONDITION_RESULTS': _json(record['responses'])}, text=m['text'], image_path=image)
            if result['status'] != 'ok': raise ValueError(result.get('error') or result['status'])
            record['shortcuts']['automatic'] = _strict_labels(result['parsed_output'], SHORTCUTS)
        record['steps'][step] = 'done'
    except (ValueError, KeyError, TypeError) as exc:
        record['steps'][step] = 'failed'
        record['notes'].append({'step': step, 'error': str(exc)})
        if step.startswith('score_') and record['responses'][step.removeprefix('score_')]:
            record['responses'][step.removeprefix('score_')].update(score_status='unresolved', correct=None, score_reason=str(exc))
    if result is not None:
        record['step_outputs'][step] = {'call_id': result['call_id'], 'status': result['status'],
                                      'raw_output': result.get('raw_output'), 'parsed_output': result.get('parsed_output')}


def summarize_experiment(out, root):
    from .metrics import summarize_records
    metrics = summarize_records(out / 'records.jsonl', output_dir=out)
    calls = read_jsonl(out/'calls.jsonl')
    write_json(out/'usage.json', call_usage(calls))
    write_jsonl(out/'answer_inputs.jsonl', [
        {'attempt_id': c['attempt_id'], 'condition': c['condition'], 'call_id': c['call_id'],
         'prompt': c['actual_input']['prompt'], 'images': c['actual_input']['images'], 'history': []}
        for c in calls if c['step']=='solve-v1'])
    store = ExperimentStore(out)
    export_cases(out, store.rows())
    export_review(out, root)
    if _gated(store.manifest['config']):
        export_reference_review(out, store.rows())
    return metrics


def export_reference_review(out, rows):
    """Reviewer worksheet: one row per generated Q; edited rows are never overwritten."""
    path = out / 'reference_review_template.jsonl'; existing = read_jsonl(path)
    seen = {r['attempt_id'] for r in existing}; additions = []
    for r in rows:
        if r['attempt_id'] in seen or not r['question']['normalized_text'] or r['steps']['candidate_answer'] == 'pending':
            continue
        candidate = r['reference']['candidate_answer'] or {}
        additions.append({'attempt_id': r['attempt_id'], 'question_sha256': r['question']['sha256'],
            'material_id': r['material_id'], 'image_path': r['material']['image_path'],
            'text': r['material']['text'], 'question': r['question']['normalized_text'],
            'candidate': {k: candidate.get(k) for k in ('status', 'answer', 'acceptable_answers', 'rubric',
                                                         'derivation_brief', 'issues')} if candidate else None,
            'candidate_step': r['steps']['candidate_answer'],
            'decision': None, 'answer': None, 'acceptable_answers': [], 'rubric': [], 'note': None,
            'reviewer': None, 'source': None})
    if additions:
        write_jsonl(path, existing + additions)


def apply_reference_reviews(*, run_id, reviews_path, project_root='.', output_root='outputs'):
    """Bind explicit reviewer decisions to the exact Q, then release the gated steps.

    Rows without a decision are skipped (still awaiting review). Generation, the
    candidate answer and the four condition responses are never re-requested.
    """
    from .records import atomic_upsert, review_source
    root = Path(project_root).resolve(); out = root / output_root / run_id
    store = ExperimentStore(out)
    if not _gated(store.manifest['config']):
        raise ValueError('reference-review applies only to review_gated runs; use experiment-review for others')
    reviews = [r for r in read_jsonl(root / reviews_path) if r.get('decision') is not None]
    if len({r['attempt_id'] for r in reviews}) != len(reviews):
        raise ValueError('duplicate attempt in reference reviews')
    for review in reviews:  # validate the whole file before changing anything
        record = store.get(review['attempt_id'])
        review_source(review)
        if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
            raise ValueError('reviewer identity required')
        if not record['question']['normalized_text'] or review.get('question_sha256') != record['question']['sha256']:
            raise ValueError('reference review belongs to another Q')
        if review['decision'] not in REFERENCE_DECISIONS:
            raise ValueError('decision must be one of ' + ', '.join(REFERENCE_DECISIONS))
        if not isinstance(review.get('note'), str) or not review['note'].strip():
            raise ValueError('reference decision requires a checkable note')
        candidate = record['reference']['candidate_answer']
        if review['decision'] == 'accept_candidate' and not (candidate and candidate.get('status') == 'answerable'):
            raise ValueError('no answerable candidate to accept')
        if review['decision'] == 'corrected' and not (
                isinstance(review.get('answer'), str) and review['answer'].strip() and review.get('rubric')):
            raise ValueError('corrected reference requires answer and rubric')
    changed = 0
    for review in reviews:
        record = store.get(review['attempt_id']); previous = copy.deepcopy(record); ref = record['reference']
        source = review_source(review)
        decision = {k: review.get(k) for k in ('attempt_id', 'question_sha256', 'decision', 'answer',
            'acceptable_answers', 'rubric', 'note', 'reviewer', 'source', 'user_authorized', 'authorization')}
        review_id = stable_id('refreview', review['attempt_id'], sha256_text(canonical_json(decision)))
        if review_id in ref['review_ids']:
            continue
        atomic_upsert(out / 'reviews.jsonl', 'review_id', {'review_id': review_id, 'kind': 'reference',
            'review': decision, 'candidate_answer': ref['candidate_answer'], 'created_at': utc_now()})
        def basis():
            return canonical_json({k: ref[k] for k in ('status', 'verified_answer', 'acceptable_answers', 'rubric')})
        old = basis(); tag = f"{source}:{review['reviewer']}:{review['decision']}"
        if review['decision'] == 'accept_candidate':
            c = ref['candidate_answer']
            ref.update(status='verified', verified_answer=c['answer'], acceptable_answers=c['acceptable_answers'],
                       rubric=c['rubric'], text_evidence=c['text_evidence'], image_evidence=c['image_evidence'])
        elif review['decision'] == 'corrected':
            ref.update(status='verified', verified_answer=review['answer'],
                       acceptable_answers=review.get('acceptable_answers') or [], rubric=review['rubric'])
        else:
            ref.update(status=review['decision'], verified_answer=None, acceptable_answers=[], rubric=[])
        ref['verification_source'] = tag; ref['review_ids'].append(review_id)
        # A later decision that changes the reference is a new logged evaluation event.
        redo = basis() != old and any(record['steps'][s] in ('done', 'failed') for s in GATED_STEPS)
        if redo:
            record['evaluation_revision'] = record.get('evaluation_revision', 0) + 1
        for step in GATED_STEPS:
            if record['steps'][step] == 'blocked' or redo:
                record['steps'][step] = 'pending'
                response = record['responses'].get(step.removeprefix('score_')) if step.startswith('score_') else None
                if response:
                    response.update(correct=None, score_status='pending', score_reference_sha256=None)
        record['notes'].append({'event': 'reference_review', 'review_id': review_id,
                                'decision': review['decision'], 'created_at': utc_now()})
        record['status'] = 'running'; record['updated_at'] = utc_now()
        store.put(record, previous=previous); changed += 1
    summarize_experiment(out, root)
    return {'updated_records': changed, 'undecided_rows_skipped': len(read_jsonl(root / reviews_path)) - len(reviews)}


def export_review(out, root):
    rows = ExperimentStore(out).rows(); review_rows = []
    fragments = ['<!doctype html><meta charset="utf-8"><title>Blind review</title>',
                 '<p>原始材料审阅：自动结论默认隐藏。最终裁定见adjudicated，审核来源保留human或user_delegated_ai，不相互冒充。</p>']
    for r in rows:
        if r['steps']['generation'] == 'pending':
            continue
        m = r['material']; q = r['question']
        review_rows.append({'attempt_id': r['attempt_id'], 'question_sha256': q['sha256'],
            'reviewer': None, 'source': 'human', 'labels': empty_labels(), 'shortcuts': empty_labels(SHORTCUTS),
            'reference': {'status': 'pending', 'answer': None, 'acceptable_answers': [], 'rubric': []}, 'notes': None})
        image = (root / m['image_path']).resolve().as_uri()
        fragments.append('<article><h2>'+html.escape(r['attempt_id'])+'</h2><pre>'+html.escape(m['text'])+
            '</pre><img style="max-width:650px" src="'+html.escape(image, quote=True)+'"><p>'+html.escape(q['normalized_text'] or '(无问题)')+
            '</p><details><summary>候选答案与自动结果（完成独立标注后查看）</summary><pre>'+html.escape(_json(
                {'reference': r['reference'], 'labels': r['labels']['automatic'], 'responses': r['responses']}))+'</pre></details></article>')
    (out / 'review.html').write_text('\n'.join(fragments))
    # Preserve all edited rows; add blanks only when another method becomes ready.
    path = out / 'review_template.jsonl'; existing = read_jsonl(path)
    seen = {r['attempt_id'] for r in existing}
    additions = [r for r in review_rows if r['attempt_id'] not in seen]
    if additions:
        write_jsonl(path, existing + additions)


def apply_human_reviews(*, run_id, reviews_path, adjudicate=False, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve(); out = root / output_root / run_id
    store = ExperimentStore(out); reviews = read_jsonl(root / reviews_path)
    if not reviews or len({r['attempt_id'] for r in reviews}) != len(reviews):
        raise ValueError('review file must have unique nonempty attempts')
    # Validate the entire handoff before making any change.
    for review in reviews:
        record = store.get(review['attempt_id'])
        from .records import review_source
        review_source(review)
        if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
            raise ValueError('reviewer identity required')
        if review['question_sha256'] != record['question']['sha256']:
            raise ValueError('review belongs to another Q')
        _strict_labels(review['labels'], DIMENSIONS)
        if any(v.get('status') != 'pending' for v in review.get('shortcuts', {}).values()):
            _strict_labels(review['shortcuts'], SHORTCUTS)
        ref = review.get('reference', {})
        if ref.get('status') == 'verified' and (not ref.get('answer') or not ref.get('rubric')):
            raise ValueError('verified reference requires answer and rubric')
        if 'condition_scores' in review:
            for c in CONDITIONS:
                label(review['condition_scores'][c]['value'], review['condition_scores'][c]['reason'])
    from .records import atomic_upsert
    changed = 0
    for review in reviews:
        record = store.get(review['attempt_id']); previous = copy.deepcopy(record)
        source = review_source(review)
        bank = 'human' if source == 'human' else 'delegated'
        digest = sha256_text(canonical_json(review))
        review_id = stable_id('review', review['attempt_id'], digest, adjudicate)
        audit_path = out / 'reviews.jsonl'
        applied = all(item.get('review_id') == review_id for item in record['labels'][
            'adjudicated' if adjudicate else bank].values()) if (adjudicate or bank in record['labels']) else False
        if applied:
            continue
        if not any(r['review_id'] == review_id for r in read_jsonl(audit_path)):
            atomic_upsert(audit_path, 'review_id', {'review_id': review_id, 'review': review,
                                                   'adjudicated': adjudicate, 'created_at': utc_now()})
        for family, keys in (('labels', DIMENSIONS), ('shortcuts', SHORTCUTS)):
            if family == 'shortcuts' and not any(v.get('status') != 'pending' for v in review.get(family, {}).values()):
                continue
            checked = _strict_labels(review[family], keys)
            for item in checked.values():
                item.update(source=source, reviewer=review['reviewer'], review_id=review_id)
            record[family][bank] = checked
            if adjudicate:
                record[family]['adjudicated'] = copy.deepcopy(checked)
        incoming = review.get('reference', {})
        if incoming.get('status') == 'verified':
            ref = record['reference']
            old_score_basis = canonical_json({k: ref[k] for k in ('verified_answer', 'acceptable_answers', 'rubric')})
            ref.update(status='verified', verified_answer=incoming['answer'],
                acceptable_answers=incoming.get('acceptable_answers', []), rubric=incoming['rubric'],
                verification_source=source + ':' + review['reviewer'])
            if review_id not in ref['review_ids']:
                ref['review_ids'].append(review_id)
            if old_score_basis != canonical_json({k: ref[k] for k in ('verified_answer', 'acceptable_answers', 'rubric')}):
                record['evaluation_revision'] = record.get('evaluation_revision', 0) + 1
                for c in CONDITIONS:
                    record['steps']['score_'+c] = 'pending'
                    if record['responses'][c]:
                        record['responses'][c].update(correct=None, score_status='pending', score_reference_sha256=None)
                if 'condition_scores' not in review:
                    record['steps']['necessity'] = record['steps']['shortcuts'] = 'pending'
                    record['status'] = 'running'
        elif incoming.get('status') == 'invalid_question':
            record['reference'].update(status='invalid_question', verification_source=source+':'+review['reviewer'])
        if 'condition_scores' in review:
            ref = record['reference']
            basis = sha256_text(canonical_json({k: ref[k] for k in ('verified_answer','acceptable_answers','rubric')}))
            for c in CONDITIONS:
                response = record['responses'][c]
                if response:
                    decision = review['condition_scores'][c]
                    response.setdefault('pre_review_score', {k:response.get(k) for k in (
                        'correct','score_status','score_reason','score_reference_sha256','score_call_id')})
                    response.update(correct=decision['value'],score_status='scored' if decision['value'] is not None else 'unresolved',
                        score_reason=decision['reason'],score_source=source,score_review_id=review_id,score_reference_sha256=basis)
                record['steps']['score_'+c] = 'done'
        record['updated_at'] = utc_now(); store.put(record, previous=previous); changed += 1
    summarize_experiment(out, root)
    return {'updated_records': changed, 'adjudicated': adjudicate}


def freeze_evaluation(*, run_id, project_root='.', output_root='outputs', protocol_path=None):
    root = Path(project_root).resolve(); out = root / output_root / run_id
    store = ExperimentStore(out); rows = store.rows()
    if len(rows) != 12 or any(r['method'] != 'imported' for r in rows):
        raise ValueError('evaluation freeze requires all 12 imported controls')
    examples = 0
    for r in rows:
        final = r['labels']['adjudicated']
        if any(item['status'] not in ('decided','undecided','not_applicable') or not item.get('review_id') for item in final.values()):
            raise ValueError('all five dimensions require explicit final review before freeze')
        if final['validity']['value'] == 1 and r['reference']['status'] != 'verified':
            raise ValueError('valid Q requires a human/program-verified reference')
        if any(s == 'pending' for s in r['steps'].values()):
            raise ValueError('finish pending model steps before freeze')
        if final['validity']['value'] == final['relatedness']['value'] == 1 and any(
                final[k]['value'] == 0 for k in ('text_necessary', 'image_necessary')):
            examples += 1
    if examples < 2:
        raise ValueError('fewer than 2 human-confirmed relevant-but-unnecessary examples; report result without forcing the requested conclusion')
    manifest = store.manifest
    delegated = any(r['labels']['adjudicated']['validity'].get('source')=='user_delegated_ai' for r in rows)
    manifest.update(evaluation_status='frozen_delegated_reviewed' if delegated else 'frozen_human_reviewed',
                    review_confirmed_examples=examples, human_confirmed_examples=0 if delegated else examples,
                    frozen_records_sha256=sha256_file(out / 'records.jsonl'),
                    frozen_prompt_sha256=manifest['prompt_sha256'], frozen_at=utc_now())
    if protocol_path:
        path=root/protocol_path
        manifest.update(evaluation_protocol_path=str(path),evaluation_protocol_sha256=sha256_file(path))
    write_json(out / 'manifest.json', manifest)
    return {'status': manifest['evaluation_status'], 'review_confirmed_examples': examples,
            'human_confirmed_examples':manifest['human_confirmed_examples']}


def probe_reproducibility(*, run_id, project_root='.', output_root='outputs', max_requests=3):
    root = Path(project_root).resolve(); out = root / output_root / run_id
    store = ExperimentStore(out); record = store.rows()[0]
    if record['material']['experiment_partition'] != 'dev':
        raise ValueError('repeatability probe is restricted to dev materials')
    provider = JudgeProvider.from_env(root=root, run_dir=out, visual=True, max_requests=max_requests,
                                      max_tokens=store.manifest['config']['max_output_tokens'])
    provider.max_attempts = 1
    caller = ModelCaller(provider=provider, prompts=PromptSet(out / 'prompts', names=TEMPLATES),
                         run_dir=out, protocol_version=store.manifest['protocol_version'], max_attempts=1)
    calls = []
    for i in range(3):
        calls.append(_invoke(caller, record, f'fresh_repeat_{i}', 'plain-v1', {'T': record['material']['text']},
            task='generate_question', text=record['material']['text'],
            image_path=str(root / record['material']['image_path']), output_format='text', nonce=str(i)))
    succeeded = all(c['status'] == 'ok' for c in calls)
    report = {'fresh_request_outputs_equal': succeeded and len({c['raw_output'] for c in calls}) == 1,
              'n_calls': len(calls), 'call_ids': [c['call_id'] for c in calls],
              'payload_sha256': [(c.get('usage') or {}).get('input_audit', {}).get('payload_sha256') for c in calls],
              'raw_output_sha256': [sha256_text(c['raw_output']) if c.get('raw_output') is not None else None for c in calls],
              'remote_requests': sum(c.get('remote_requests', 0) for c in calls),
              'model_identity': provider.identity,
              'strict_greedy_verified': False,
              'limitation': 'API does not expose do_sample=False; equality of three responses is an empirical check, not a deterministic guarantee.'}
    write_json(out / 'reproducibility.json', report)
    return report


def call_usage(calls):
    active = [r for r in calls if r['status'] not in ('started', 'budget_paused')]
    totals = {}
    for output, key in (('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),('total_tokens','total_tokens'),('cost','cost')):
        values = [(r.get('usage') or {}).get(key) for r in active]
        totals[output] = sum(values) if all(type(v) in (int,float) for v in values) else None
        totals['known_'+output] = sum(v for v in values if type(v) in (int,float))
    return {**totals, 'remote_requests': sum(r.get('remote_requests',
                (r.get('usage') or {}).get('network_requests', 0 if r.get('cache_hit') else 1)) for r in active),
            'cache_hits': sum(bool(r.get('cache_hit')) for r in active),
            'elapsed_seconds': sum(r.get('elapsed_seconds',0) for r in active),
            'usage_complete': totals['total_tokens'] is not None,
            'cost_complete': totals['cost'] is not None,
            'incomplete_calls': sum(r['status'] in ('started','budget_paused') for r in calls),
            'currency': None}


def export_cases(out, rows):
    from .metrics import record_states
    source = 'adjudicated' if all(r['labels']['adjudicated']['validity']['status']=='decided' for r in rows) else 'automatic'
    def priority(r):
        s=record_states(r,source); l=s['labels']
        counter = l['validity']==l['relatedness']==1 and (l['text_necessary']==0 or l['image_necessary']==0)
        return (not counter, r['plan_index'])
    ordered=sorted(rows,key=priority); chosen=[]
    predicates = (
        lambda s: s['labels']['validity']==s['labels']['relatedness']==1 and s['labels']['text_necessary']==0 and s['labels']['image_necessary']==1,
        lambda s: s['labels']['validity']==s['labels']['relatedness']==1 and s['labels']['text_necessary']==1 and s['labels']['image_necessary']==0,
        lambda s: s['shortcuts'].get('question_leakage')==1,
        lambda s: all(s['labels'][k]==1 for k in ('validity','text_necessary','image_necessary','integration')) and s['bdr'] is None,
        lambda s: all(s['labels'][k]==1 for k in ('validity','text_necessary','image_necessary')) and s['labels']['integration']==0,
    )
    for predicate in predicates:
        match=next((r for r in ordered if r not in chosen and predicate(record_states(r,source))),None)
        if match: chosen.append(match)
    chosen += [r for r in ordered if r not in chosen][:5-len(chosen)]
    lines=['# 典型案例', '', f'标签来源：{source}。automatic案例仅供人工复核，不是已确认实证。', '']
    for r in chosen:
        s=record_states(r,source)
        lines += [f"## {r.get('case_id') or r['attempt_id']}", '',
                  f"材料：{r['material_id']}；图像：{r['material']['image_path']}", '',
                  'T：'+r['material']['text'], '', 'Q：'+str(r['question']['normalized_text']), '',
                  '参考答案：'+str(r['reference']['verified_answer']),
                  '核验来源：'+str(r['reference']['verification_source']), '',
                  '五维审核：', '', '```json', json.dumps(r['labels'][source],ensure_ascii=False,indent=2), '```', '',
                  '四条件：', '', '| 条件 | 回答 | 正确性 |', '|---|---|---|']
        for c in CONDITIONS:
            response=r['responses'][c] or {}
            answer=str(response.get('answer')).replace('|','\\|').replace('\n',' ')
            lines.append(f"| {c} | {answer} | {response.get('correct')} |")
        lines += ['', 'BDR：'+str(s['bdr']), '', '捷径审核：', '', '```json',
                  json.dumps(r['shortcuts'][source],ensure_ascii=False,indent=2), '```', '',
                  '未解决问题：'+json.dumps(r['notes'],ensure_ascii=False), '']
    (out/'cases.md').write_text('\n'.join(lines))


def probe_repeatability(*, run_id, project_root='.', output_root='outputs', max_requests=15):
    """Three preselected dev questions, five consecutive fresh requests each."""
    from difflib import SequenceMatcher
    root=Path(project_root).resolve(); out=root/output_root/run_id
    store=ExperimentStore(out); rows=store.rows()
    if len(rows)!=3 or any(r['material']['experiment_partition']!='dev' for r in rows):
        raise ValueError('repeatability validation requires exactly three dev questions')
    p=JudgeProvider.from_env(root=root,run_dir=out,visual=True,max_requests=max_requests,
        max_tokens=store.manifest['config']['max_output_tokens'])
    p.max_attempts=1
    caller=ModelCaller(provider=p,prompts=PromptSet(out/'prompts',names=TEMPLATES),
                       run_dir=out,protocol_version=store.manifest['protocol_version'],max_attempts=1)
    groups=[]
    for r in rows:
        calls=[]
        for i in range(5):
            payload=answering_input(r,'TI',root)
            calls.append(_invoke(caller,r,f'repeatability_{i}','solve-v1',task='answer_question',
                                 nonce=f"{r['case_id']}:{i}",**payload))
        raw=[c.get('raw_output') for c in calls]
        hashes=[(c.get('usage') or {}).get('input_audit',{}).get('payload_sha256') for c in calls]
        passed=all(c['status']=='ok' for c in calls) and len(set(raw))==1 and None not in hashes and len(set(hashes))==1
        groups.append({'case_id':r['case_id'],'question_sha256':r['question']['sha256'],
            'call_ids':[c['call_id'] for c in calls], 'payload_sha256':hashes,
            'raw_output_sha256':[sha256_text(s) if s is not None else None for s in raw],
            'raw_outputs_identical':passed,
            'character_difference_from_first':[None if s is None or raw[0] is None else
                {'equal':s==raw[0], 'similarity':SequenceMatcher(None,raw[0],s).ratio()} for s in raw]})
    report={'design':'3 fixed dev questions x 5 fresh requests; no history; raw character equality',
        'groups':groups,'passed':all(g['raw_outputs_identical'] for g in groups),
        'operational_term':'等效贪心（仅指本次3×5输入上的字符级重复性）',
        'decoding_mechanism_verified':False,'model_identity':p.identity,
        'effective_top_p':None,'max_output_tokens':p.max_tokens,
        'limitation':'Finite repeated outputs do not establish a greedy decoding algorithm or guarantee future determinism.',
        'usage':call_usage(read_jsonl(out/'calls.jsonl'))}
    write_json(out/'reproducibility.json',report)
    return report


def export_source_review(*, run_id, project_root='.', output_root='outputs', seed=20261111):
    """Source-only blind packet: T, image and exact Q under anonymous shuffled IDs.

    Method, candidate reference, automatic labels, solver answers and BDR are not
    written here; key.jsonl maps IDs back and is opened only after labels are sealed.
    Attempts without a generated Q stay in the plan and are listed in the manifest.
    """
    import random
    root = Path(project_root).resolve(); out = root / output_root / run_id
    store = ExperimentStore(out); target = out / 'source_review'
    if (target / 'manifest.json').exists():
        raise ValueError('source review packet already exists; it is never regenerated or reordered')
    rows = store.rows()
    if any(r['steps']['generation'] == 'pending' for r in rows):
        raise ValueError('finish the planned generation stage before building the packet')
    with_q = [r for r in rows if r['question']['normalized_text']]
    random.Random(seed).shuffle(with_q)
    packet, template, key = [], [], []
    for index, r in enumerate(with_q, 1):
        rid = f'S{index:02d}'; m = r['material']
        packet.append({'review_id': rid, 'text': m['text'], 'image_path': str((root / m['image_path']).resolve()),
                       'image_sha256': m['image_sha256'], 'question': r['question']['normalized_text'],
                       'question_sha256': r['question']['sha256']})
        template.append({'review_id': rid, 'question_sha256': r['question']['sha256'],
                         'independently_derived_answer': None, 'acceptable_answers': [], 'rubric': [],
                         'labels': empty_labels(), 'shortcuts': empty_labels(SHORTCUTS),
                         'unresolved_assumptions': None, 'reviewer': None, 'source': None,
                         'user_authorized': None, 'authorization': None, 'prior_exposure': None})
        key.append({'review_id': rid, 'attempt_id': r['attempt_id']})
    target.mkdir()
    write_jsonl(target / 'source_inputs.jsonl', packet); write_jsonl(target / 'labels_template.jsonl', template)
    write_jsonl(target / 'key.jsonl', key)
    fragments = ['<!doctype html><meta charset="utf-8"><title>Source-only review</title>']
    for item in packet:
        fragments.append('<article><h2>' + item['review_id'] + '</h2><pre style="white-space:pre-wrap">' +
            html.escape(item['text']) + '</pre><img style="max-width:650px" src="' +
            html.escape(Path(item['image_path']).as_uri(), quote=True) + '"><p><b>Q:</b> ' +
            html.escape(item['question']) + '</p><p><code>' + item['question_sha256'] + '</code></p></article>')
    (target / 'packet.html').write_text('\n'.join(fragments))
    protocol = root / 'docs/evaluation_protocol_v1.0.md'
    manifest = {'run_id': run_id, 'n_planned': len(rows), 'n_in_packet': len(packet),
        'attempts_without_question': [r['attempt_id'] for r in rows if not r['question']['normalized_text']],
        'shuffle_seed': seed, 'source_inputs_sha256': sha256_file(target / 'source_inputs.jsonl'),
        'key_sha256': sha256_file(target / 'key.jsonl'),
        'protocol_sha256': sha256_file(protocol) if protocol.exists() else None,
        'hidden_until_sealed': ['method', 'candidate reference', 'automatic labels', 'solver answers', 'BDR',
                                'generation evidence and pairing'],
        'do_not_open_before_seal': ['source_review/key.jsonl', 'records.jsonl', 'calls.jsonl', 'cases.md',
                                    'review.html', 'review_template.jsonl', 'reference_review_template.jsonl',
                                    'summary.md', 'metrics.json'],
        'created_at': utc_now()}
    write_json(target / 'manifest.json', manifest)
    return {'packet_dir': str(target), 'n_in_packet': len(packet), 'n_planned': len(rows)}
