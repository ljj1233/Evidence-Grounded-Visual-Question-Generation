"""Read the Phase 2.1 protocol literally; parse failures are not support labels."""
import hashlib
import json
from pathlib import Path

from vvqg.rewards.support.calibration import LABELS, aggregate_requirements, overall_support

ROOT = Path(__file__).resolve().parents[4]
PROMPTS = ROOT / 'configs/new_vqg/visual_support_prompts.json'


def frozen_prompts():
    prompts = json.loads(PROMPTS.read_text())
    return {'holistic': prompts['holistic'], 'flat': prompts['flat']}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_prediction(raw, variant):
    text = (raw or '').strip()
    if text.startswith('```'):
        text = text.partition('\n')[2].rsplit('```', 1)[0].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {'status': 'parser_failure'}
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict):
        return {'status': 'malformed_structured_output'}
    if variant == 'holistic':
        fields = ('premise_consistency', 'reference_clarity', 'visual_answerability')
        if any(data.get(k) not in LABELS[k] for k in fields):
            return {'status': 'malformed_structured_output', 'structure': data}
        labels = {k: data[k] for k in fields}
        labels['overall_visual_support'] = overall_support(*(labels[k] for k in fields))
    else:
        p, r, t = (data.get(k) for k in ('presuppositions', 'reference_constraints', 'query_target'))
        if not isinstance(p, list) or not isinstance(r, list) or not isinstance(t, dict):
            return {'status': 'malformed_structured_output', 'structure': data}
        groups = ((p, LABELS['premise_consistency']), (r, ('clear', 'ambiguous', 'uncertain')),
                  ([t], LABELS['visual_answerability']))
        if any(not isinstance(n, dict) or not isinstance(n.get('text'), str) or not n['text'].strip()
               or n.get('label') not in vocabulary for nodes, vocabulary in groups for n in nodes):
            return {'status': 'malformed_structured_output', 'structure': data}
        labels = aggregate_requirements(data, [n['label'] for n in p + r + [t]])
    return {'status': 'ok', **labels, 'structure': data}
