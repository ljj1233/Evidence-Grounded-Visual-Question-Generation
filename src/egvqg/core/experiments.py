"""Helpers of the baseline and evaluation workflow (the remaining part of an older module; everything else was removed)."""
from __future__ import annotations

import re
import unicodedata
from .records import CONDITIONS, sha256_text, stable_id, utc_now


DIMENSIONS = ('validity', 'relatedness', 'text_necessary', 'image_necessary', 'integration')

SHORTCUTS = ('text_only', 'image_only', 'common_knowledge', 'question_leakage', 'independent_stitching')

TEMPLATES = ('plain-v2', 'solve-v1')

STEPS = ('generation', 'candidate_answer', 'independent_answer', 'reference_resolution',
         'answer_TI', 'answer_T', 'answer_I', 'answer_0',
         'score_TI', 'score_T', 'score_I', 'score_0', 'relatedness', 'necessity', 'shortcuts')

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

def answering_input(record, condition, root):
    if condition not in CONDITIONS:
        raise ValueError('unknown condition')
    m = record['material']; q = record['question']['normalized_text']
    visible = '<T>' + m['text'] + '</T>' if condition in ('TI', 'T') else ''
    return {'variables': {'VISIBLE_MATERIALS': visible, 'Q': q},
            'text': m['text'] if condition in ('TI', 'T') else None,
            'image_path': str(root / m['image_path']) if condition in ('TI', 'I') else None,
            'question': q, 'condition': condition}

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
