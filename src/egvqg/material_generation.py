"""Generate N questions per material from an arbitrary materials JSONL.

This is the generic form of the composition generator: materials only need the shared material contract
(`material_id`, `text`, `image_path`). Used for the MMQA `plain-refer-v2` feasibility run. Sampling config,
prompt snapshot and model identity are frozen in `manifest_generation.json`; rerunning the same `run_id`
resumes only missing (material, sample_index) slots.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .local_vision import BatchingProvider, QwenVLBackend, qwen_identity
from .prompts import PromptSet
from .providers import CompletionRequest
from .records import (append_jsonl, canonical_json, read_jsonl, sha256_file, sha256_text,
                      stable_id, utc_now, write_json, write_jsonl)


def _plan(run_id, materials, n_questions):
    return [{'attempt_id': stable_id('mat-gen', run_id, m['material_id'], i),
             'run_id': run_id, 'material_id': m['material_id'],
             'category': m.get('category') or m.get('group') or 'unknown',
             'sample_index': i}
            for m in materials for i in range(n_questions)]


def run_material_generation(*, run_id, materials_path, model_path, device='cuda:0', n_questions=6,
                            temperature=0.7, top_p=0.8, top_k=20, max_batch=8, max_new_tokens=96,
                            limit=None, template='plain-v2', project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    materials = read_jsonl(root / materials_path)
    if not materials:
        raise ValueError('no materials')
    missing = [m['material_id'] for m in materials if not m.get('text') or not m.get('image_path')]
    if missing:
        raise ValueError(f'materials missing text/image_path: {missing[:3]}')
    by_id = {m['material_id']: m for m in materials}
    plan = _plan(run_id, materials, n_questions)
    if limit:
        plan = plan[:limit]
    prompts = PromptSet(root / 'prompts', names=(template,))
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens,
                            do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    identity = qwen_identity(backend, 'material-generation-v1', do_sample=True, temperature=temperature,
                             top_p=top_p, top_k=top_k, template=template)
    fingerprint = sha256_text(canonical_json({
        'config': {'n_questions': n_questions, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k,
                   'max_new_tokens': max_new_tokens, 'template': template},
        'plan': plan, 'materials': sha256_file(root / materials_path),
        'prompts': prompts.snapshot(), 'identity': identity}))
    manifest_path = out / 'manifest_generation.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('generation run configuration changed; use a new run_id')
    else:
        write_json(manifest_path, {
            'created_at': utc_now(), 'fingerprint': fingerprint,
            'config': {'n_questions': n_questions, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k,
                       'max_new_tokens': max_new_tokens, 'template': template},
            'materials_path': materials_path, 'model_identity': identity, 'prompt_snapshot': prompts.snapshot()})
        (out / 'prompts').mkdir(exist_ok=True)
        for name, text in prompts.snapshot().items():
            (out / 'prompts' / f'{name}.txt').write_text(text)
    write_jsonl(out / 'generation_plan.jsonl', plan)
    prior = {row['attempt_id'] for row in read_jsonl(out / 'generated.jsonl')}
    pending = [slot for slot in plan if slot['attempt_id'] not in prior]
    if not pending:
        return {'run_id': run_id, 'planned': len(plan), 'already_done': len(prior), 'new': 0}
    provider = BatchingProvider(backend, model=backend.path.name, identity=identity, max_batch=max_batch)
    lock = threading.Lock()
    done = 0

    def work(slot):
        material = by_id[slot['material_id']]
        rendered = prompts.render(template, T=material['text'])
        request = CompletionRequest(task='material-generation', prompt=rendered, text=material['text'],
                                    image_path=str(root / material['image_path']), condition='TI',
                                    metadata={'material_id': slot['material_id'], 'sample_index': str(slot['sample_index'])})
        completion = provider.complete(request)
        raw = completion.text
        question = ' '.join(raw.strip().split())
        row = {**slot, 'question': question, 'raw_output': raw, 'finish_reason': completion.finish_reason,
               'usage': completion.usage, 'prompt_sha256': sha256_text(rendered),
               'question_sha256': sha256_text(question), 'completed_at': utc_now()}
        with lock:
            append_jsonl(out / 'generated.jsonl', row)
        return row

    with ThreadPoolExecutor(max_workers=max(2, max_batch * 2)) as pool:
        futures = [pool.submit(work, slot) for slot in pending]
        for future in as_completed(futures):
            future.result()
            done += 1
    return {'run_id': run_id, 'planned': len(plan), 'already_done': len(prior), 'new': done}

