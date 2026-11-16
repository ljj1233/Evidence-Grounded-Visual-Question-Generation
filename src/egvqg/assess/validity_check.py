"""Local validity check for generated questions (Qwen3-VL-8B, not the official cascade evaluator)."""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..core.local_vision import BatchingProvider, QwenVLBackend, qwen_identity
from ..core.prompts import PromptSet
from ..core.providers import CompletionRequest
from ..core.records import canonical_json, read_jsonl, sha256_file, sha256_text, stable_id, utc_now, write_json, write_jsonl


def _parse_valid(text):
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        return None, 'no_json'
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, 'bad_json'
    if not isinstance(value, dict) or type(value.get('valid')) is not int or value.get('valid') not in (0, 1):
        return None, 'bad_contract'
    return int(value['valid']), str(value.get('reason', ''))[:300]


def _programmatic_valid(question):
    words = question.split()
    return bool(question.strip().endswith('?') and 3 <= len(words) <= 60 and 'output only' not in question.lower())


def _append_jsonl(path, row, lock):
    with lock:
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def run_local_validity(*, run_id, source_run, model_path, device='cuda:0', max_batch=8,
                       max_new_tokens=96, limit=None, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    source_dir = root / output_root / source_run
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(source_dir / 'generated.jsonl')
    if not rows:
        raise ValueError('source run has no generated.jsonl')
    manifest = json.loads((source_dir / 'manifest_generation.json').read_text())
    from ..studies.synthetic_composition import load_composition_materials
    materials = {m['material_id']: m for m in load_composition_materials(root, manifest['materials_path'])}
    prompts = PromptSet(root / 'prompts', names=('local-validity-v1',))
    plan = []
    for row in rows:
        plan.append({'attempt_id': stable_id('validity', run_id, row['material_id'], row['sample_index']),
                     'material_id': row['material_id'], 'category': row['category'],
                     'sample_index': row['sample_index'], 'question': row['question']})
    if limit:
        plan = plan[:limit]
    prior = {row['attempt_id'] for row in read_jsonl(out / 'validity.jsonl')} if (out / 'validity.jsonl').exists() else set()
    pending = [slot for slot in plan if slot['attempt_id'] not in prior]
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens)
    identity = qwen_identity(backend, 'local-validity-v1')
    manifest_path = out / 'manifest_validity.json'
    fingerprint = sha256_text(canonical_json({'config': {'max_new_tokens': max_new_tokens}, 'source_run': source_run,
        'plan': plan, 'prompts': prompts.snapshot(), 'identity': identity}))
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('validity run configuration changed')
    else:
        write_json(manifest_path, {'created_at': utc_now(), 'fingerprint': fingerprint, 'source_run': source_run,
            'config': {'max_new_tokens': max_new_tokens}, 'model_identity': identity, 'prompt_snapshot': prompts.snapshot()})
    write_jsonl(out / 'validity_plan.jsonl', plan)
    if not pending:
        return {'run_id': run_id, 'planned': len(plan), 'already_done': len(prior), 'new': 0}
    provider = BatchingProvider(backend, model=backend.path.name, identity=identity, max_batch=max_batch)
    lock = threading.Lock()
    done = 0

    def work(slot):
        material = materials[slot['material_id']]
        rendered = prompts.render('local-validity-v1', T=material['text'], Q=slot['question'])
        request = CompletionRequest(task='local-validity', prompt=rendered, question=slot['question'],
                                    text=material['text'], image_path=str(root / material['image_path']), condition='TI',
                                    metadata={'material_id': slot['material_id']})
        completion = provider.complete(request)
        valid, reason = _parse_valid(completion.text)
        row = {**slot, 'raw_output': completion.text, 'valid': valid, 'reason': reason,
               'programmatic_valid': _programmatic_valid(slot['question']),
               'finish_reason': completion.finish_reason, 'usage': completion.usage, 'completed_at': utc_now()}
        _append_jsonl(out / 'validity.jsonl', row, lock)
        return valid

    with ThreadPoolExecutor(max_workers=max(2, max_batch * 2)) as pool:
        futures = [pool.submit(work, slot) for slot in pending]
        for future in as_completed(futures):
            future.result()
            done += 1
    return {'run_id': run_id, 'planned': len(plan), 'already_done': len(prior), 'new': done}


def analyze_validity(*, run_id, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    rows = read_jsonl(out / 'validity.jsonl')
    if not rows:
        raise ValueError('no validity rows')
    manifest = json.loads((out / 'manifest_validity.json').read_text())
    by_category = {}
    for category in sorted({row['category'] for row in rows}):
        subset = [row for row in rows if row['category'] == category]
        parsed = [row for row in subset if row['valid'] is not None]
        by_category[category] = {
            'n': len(subset), 'parsed': len(parsed),
            'valid_rate': (sum(row['valid'] for row in parsed) / len(parsed)) if parsed else None,
            'programmatic_valid_rate': sum(bool(row['programmatic_valid']) for row in subset) / len(subset),
        }
    parsed = [row for row in rows if row['valid'] is not None]
    report = {'run_id': run_id, 'source_run': manifest['source_run'], 'model': manifest['model_identity']['model'],
              'n': len(rows), 'parsed': len(parsed),
              'valid_rate': (sum(row['valid'] for row in parsed) / len(parsed)) if parsed else None,
              'programmatic_valid_rate': sum(bool(row['programmatic_valid']) for row in rows) / len(rows),
              'by_category': by_category}
    write_json(out / 'validity_report.json', report)
    lines = [f'# Local validity check: {run_id}', '',
             f'- source: `{manifest["source_run"]}`',
             f'- model: `{report["model"]}` (local Qwen, not the official cascade evaluator)',
             f'- rows: {len(rows)}; parsed: {len(parsed)}',
             f'- valid rate: {report["valid_rate"]:.3f}' if report['valid_rate'] is not None else '- valid rate: n/a',
             f'- programmatic well-formed rate: {report["programmatic_valid_rate"]:.3f}', '',
             '| category | n | parsed | model valid rate | programmatic well-formed |', '|---|---:|---:|---:|---:|']
    for category, values in by_category.items():
        valid = 'n/a' if values['valid_rate'] is None else f'{values["valid_rate"]:.3f}'
        lines.append(f'| {category} | {values["n"]} | {values["parsed"]} | {valid} | {values["programmatic_valid_rate"]:.3f} |')
    lines += ['', 'The model check uses `local-validity-v1`; it is a local AI review, not the project evaluator. '
              'The programmatic check requires a question mark, 3-60 words, and no prompt leakage.', '']
    (out / 'validity_report.md').write_text('\n'.join(lines))
    return report
