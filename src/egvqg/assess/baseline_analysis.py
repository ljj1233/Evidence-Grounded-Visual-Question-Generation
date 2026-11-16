"""Helpers of the baseline and evaluation workflow (the remaining part of an older module; everything else was removed)."""
from __future__ import annotations

from ..core.records import canonical_json


FAITH_TEMPLATE = 'review-question-faithfulness-v1'

def proof_shape(proof):
    if not isinstance(proof, dict) or not isinstance(proof.get('reason'), str) or not proof['reason'].strip():
        return False
    keys = (('answer',) if proof.get('kind') == 'unique' else
            ('scenario_a','answer_a','scenario_b','answer_b') if proof.get('kind') == 'ambiguous' else ())
    return bool(keys) and all(isinstance(proof.get(k), str) and proof[k].strip() for k in keys)

def analysis_input(row, stage, removed, config, proof=None):
    """Only Q/T/I/K; proof checker sees one candidate, never old decisions or reference."""
    q = row['question']['normalized_text']; m = row['material']
    if stage == 'faithfulness':
        return {'template': FAITH_TEMPLATE,
                'variables': {'Q': q, 'T': m['text'], 'KNOWLEDGE': config['knowledge'],
                              'REFERENCE': 'No reference answer provided. Independently check whether the original Q is supported by the material; fill reference_supported with null.'},
                'text': m['text'], 'image_path': m['image_path'], 'question': q, 'condition': 'TI'}
    retained = 'I' if removed == 'T' else 'T'
    if stage == 'screen':      # the retained modality alone: is the full answer already determined?
        from .necessity_ensemble import NO_IMAGE_TEXT
        return {'template': 'ensemble-sufficiency-v1',
                'variables': {'Q': q, 'MATERIAL': '<T>' + m['text'] + '</T>' if retained == 'T' else NO_IMAGE_TEXT,
                              'KNOWLEDGE': config['knowledge']},
                'question': q, 'condition': retained, 'text': m['text'] if retained == 'T' else None,
                'image_path': m['image_path'] if retained == 'I' else None}
    text = m['text'] if retained == 'T' else None
    variables = {'Q': q, 'REMOVED': 'text' if removed == 'T' else 'image',
                 'VISIBLE': '<T>'+text+'</T>' if text else 'the attached original image only; there is no text material.',
                 'KNOWLEDGE': config['knowledge']}
    template = config['analysis_template']
    if stage == 'verification':
        template = config['verification_template']
        keys = ('scenario_a','answer_a','scenario_b','answer_b') if proof['kind'] == 'ambiguous' else ('answer','reason')
        variables['EVIDENCE'] = canonical_json({'proof_form': (
            'two_hypothetical_completions' if proof['kind']=='ambiguous' else 'single_sufficient_solution_no_scenarios_required'),
            **{k: proof[k] for k in keys}})
    return {'template': template, 'variables': variables, 'question': q,
            'condition': retained, 'text': text, 'image_path': m['image_path'] if retained == 'I' else None}
