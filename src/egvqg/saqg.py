"""Schema-Anchored Question Generation (SAQG).

Pipeline: extract a text rule and an image-specific fact; decide whether they are complementary; route to a
synergistic question schema; generate anchored questions.  Materials without complementary evidence are abstained
(optionally forced for measurement).  This module only generates candidates; labels come from the existing
single-modality screen / consensus label repair.
"""
from __future__ import annotations

import json
from pathlib import Path

from .candidates import candidate_seed
from .llm import json_object
from .records import append_jsonl, utc_now, write_json

VERSION = 'saqg-v1'
SCHEMA_EXAMPLE = {
    'probability': 'What is the probability that an offspring from this cross has the recessive phenotype?',
    'exchange': 'What could the two students trade so that each gets an item they want?',
    'rule-comparison': 'Which object has the higher value when the stated rule is applied to the measurements in the image?',
}
SCHEMA_INSTRUCTION = {
    'probability': 'Ask for a probability or ratio that requires applying the rule to the genotypes shown in the image.',
    'exchange': 'Ask what trade or exchange is possible, using the preferences in the context and what each person has in the image.',
    'rule-comparison': 'Ask a comparison that requires applying the rule to the values shown in the image.',
}
SEED = 20261113


def extract_evidence(backend, prompts, material, root, *, max_new_tokens=512):
    from .providers import CompletionRequest
    request = CompletionRequest(task='assess_quality', prompt=prompts.render('saqg-extract-v1', T=material['text']),
                                image_path=str(root / material['image_path']))
    old = backend.max_new_tokens
    backend.max_new_tokens = max_new_tokens
    try:
        text = backend.generate([request])[0].text
    finally:
        backend.max_new_tokens = old
    parsed = json_object(text) or {}
    rule = parsed.get('rule') if isinstance(parsed.get('rule'), str) else 'the general rule stated in the context'
    fact = parsed.get('fact') if isinstance(parsed.get('fact'), str) else 'the specific value shown in the image'
    complementary = parsed.get('complementary') if isinstance(parsed.get('complementary'), bool) else None
    return {'rule': rule, 'fact': fact, 'complementary': complementary,
            'reason': parsed.get('reason'), 'raw': text[:1200], 'parsed': bool(parsed)}


def run_saqg(*, run_id, materials_file, model_path, project_root='.', output_root='outputs', device='cuda:0',
             n_candidates=6, modes=('saqg', 'control'), force_negative=False, limit=None, prompt_name='saqg-generate-v1'):
    from .local_vision import QwenVLBackend
    from .prompts import PromptSet
    from .providers import CompletionRequest
    from .records import read_jsonl
    from .whitebox import Capturer

    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    materials = {m['material_id']: m for m in read_jsonl(root / materials_file)}
    prompts = PromptSet(root / 'prompts', names=('plain-v2', 'saqg-extract-v1', 'saqg-generate-v1', 'saqg-generate-v2'))
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=1024)
    capturer = Capturer(backend, k=1)
    backend._load()
    sampling = {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}
    decisions, counts = [], {m: 0 for m in modes}
    for n, material in enumerate(materials.values(), 1):
        if limit is not None and n > limit:
            break
        mid = material['material_id']
        text = material['text']
        evidence = extract_evidence(backend, prompts, material, root)
        schema = material.get('schema_hint')
        eligible = material.get('affordance_class') == 'positive'
        abstain = not eligible   # affordance class gates; extraction only supplies rule/fact
        decision = {'material_id': mid, 'affordance_class': material.get('affordance_class'), 'schema_hint': schema,
                    'evidence': evidence, 'abstain': abstain, 'generated': {}, 'created_at': utc_now()}
        for mode in modes:
            if mode == 'saqg':
                if abstain and not force_negative:
                    continue
                if not schema:
                    if not force_negative:
                        continue
                    schema = 'rule-comparison'
                prompt = prompts.render(prompt_name, T=text, R=evidence['rule'], F=evidence['fact'],
                                        SCHEMA=SCHEMA_INSTRUCTION[schema], EXAMPLE=SCHEMA_EXAMPLE.get(schema, ''))
            else:
                prompt = prompts.render('plain-v2', T=text)
            request = CompletionRequest(task='generate_question', prompt=prompt, image_path=str(root / material['image_path']))
            prepared = capturer.prepare(request, text)
            made = []
            for i in range(n_candidates):
                seed = candidate_seed(SEED, mid, i)
                _, meta = capturer.capture(request, text, prepared=prepared, sample={**sampling, 'seed': seed})
                question = meta['question'].strip()
                if not question:
                    continue
                append_jsonl(out / f'candidates_{mode}.jsonl', {
                    'cand_id': f'{mid}~{mode}~c{i}', 'material_id': mid, 'parent_id': mid,
                    'skill': material.get('skill'), 'condition': mode, 'question': question,
                    'label_old': {'redundant': None, 'direction': None}, 'label_new': None,
                    'valid': 1, 'evaluated': True, 'features': {'text_chars': len(question)},
                    'seed': seed, 'created_at': utc_now()})
                made.append(question)
                counts[mode] += 1
            decision['generated'][mode] = made
        append_jsonl(out / 'decisions.jsonl', decision)
        print(mid, material.get('affordance_class'), 'abstain' if abstain else schema, {k: len(v) for k, v in decision['generated'].items()}, flush=True)
    write_json(out / 'summary.json', {'run_id': run_id, 'version': VERSION, 'modes': list(modes),
                                      'counts': counts, 'n_materials': len(materials), 'n_candidates': n_candidates,
                                      'force_negative': force_negative})
    return {'run_id': run_id, 'counts': counts, 'n_materials': len(materials)}
