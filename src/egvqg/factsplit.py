"""1a / P2: split a parent material's fact set into A/B and build four-condition materials.

Condition = (text_facts, image_facts), both drawn from A / B / F (all facts). Defaults:
  (A,B), (B,A), (F,F), (F,none), (none,F)
The LLM only extracts and splits facts (one cached call per material); no answer/question is generated here.
Image backends:
  existing  use the original image when it is supposed to carry B (assumption recorded); unsupported otherwise;
  render    draw the visible facts as text on a blank canvas (a simple implementation, flagged as an OCR shortcut).
The output is a materials file for `candidates-capture`; every condition keeps its parent id for pairing.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .candidates import load_materials_any
from .llm import CachedCaller, json_object
from .prompts import PromptSet
from .records import write_json, write_jsonl

PROMPT = 'factsplit-v1'
DEFAULT_CONDITIONS = 'A,B;B,A;F,F;F,none;none,F'


def parse_conditions(text):
    out = []
    for part in text.split(';'):
        if not part.strip():
            continue
        t, i = part.split(',')
        out.append((t.strip(), i.strip()))
    return out


def split_facts(reply):
    """Validate the LLM split: facts non-empty, A/B a disjoint partition; returns (facts, A, B, ok, note)."""
    obj = json_object(reply or '')
    if not obj or not isinstance(obj.get('facts'), list):
        return [], [], [], False, 'unparsable'
    facts = [str(f).strip() for f in obj['facts'] if str(f).strip()]
    ids = list(range(len(facts)))
    a, b = obj.get('A'), obj.get('B')
    if not isinstance(a, list) or not isinstance(b, list):
        return facts, [], [], False, 'missing A/B'
    try:
        a, b = sorted({int(x) for x in a}), sorted({int(x) for x in b})
    except (TypeError, ValueError):
        return facts, [], [], False, 'non-integer fact indices'
    if set(a) & set(b) or set(a) | set(b) != set(ids) or not a or not b:
        return facts, a, b, False, 'A/B not a disjoint non-empty partition'
    return facts, a, b, True, 'ok'


def facts_text(facts, ids):
    return ' '.join(facts[i] for i in ids) if ids else ''


def render_image(facts, out_path, size=768):
    """Simple programmatic image: the visible facts written as text (OCR shortcut by construction)."""
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (size, size), (245, 245, 240))
    draw = ImageDraw.Draw(image)
    lines, line = [], ''
    for word in ' '.join(facts).split():
        if len(line) + len(word) + 1 > 60:
            lines.append(line); line = word
        else:
            line = (line + ' ' + word).strip()
    if line:
        lines.append(line)
    draw.multiline_text((24, 24), '\n'.join(lines[:30]), fill=(20, 20, 20), spacing=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return str(out_path)


def image_for(*, material, facts, ids, side, backend, out_dir, tag):
    """Returns (image_path or None, risk flags). `side` is the requested image fact set (A/B/F/none)."""
    if side == 'none':
        if backend == 'render':
            path = render_image([], Path(out_dir) / f'{tag}_blank.png')
            return path, {'image': 'blank', 'no_fact_content': True}
        return None, {'image': 'unsupported', 'reason': 'existing backend has no blank image; use render'} 
    if not ids:
        return None, {'image': 'none'}
    if backend == 'existing':
        if side == 'B':
            return material['image_path'], {'image': 'existing', 'assumption': 'original image carries B'}
        return None, {'image': 'unsupported', 'reason': 'existing backend only supports image_facts=B'}
    if backend == 'render':
        path = render_image([facts[i] for i in ids], Path(out_dir) / f'{tag}.png')
        return path, {'image': 'render', 'ocr_shortcut_risk': True}
    raise ValueError(f'unknown image backend: {backend}')


def run_factsplit(*, project_root='.', output_root='outputs', run_id=None, reference_run='qwen3vl8b-baseline-test1257-simple-v1',
                  materials_file=None, n_materials=5, conditions=DEFAULT_CONDITIONS, image_backend='render', model='gpt-6-luna',
                  dry_run=False, max_calls=0):
    root = Path(project_root).resolve()
    pool = load_materials_any(root, output_root, reference_run, materials_file)
    parents = sorted(pool)[:n_materials]
    prompts = PromptSet(root / 'prompts', names=(PROMPT,))
    caller = CachedCaller(project_root=project_root, run_id=run_id, output_root=output_root, dry_run=dry_run, max_calls=max_calls)
    conds = parse_conditions(conditions)
    if dry_run:
        example = prompts.render(PROMPT, TEXT=pool[parents[0]]['text']) if parents else '(no materials)'
        gen = len(parents) * len(conds) * 8
        print(f'[dry-run] factsplit: {len(parents)} materials x 1 extraction call = {len(parents)} API calls '
              f'(+1 retry each -> <= {2 * len(parents)})')
        print(f'[dry-run] generation estimate: {len(parents)} materials x {len(conds)} conditions x 8 candidates = {gen} local generations; '
              f'label evaluation ~{round(gen * 0.2 * 2.4)} API calls')
        print(f'[dry-run] image backend={image_backend}; conditions={conds}')
        print(f'[dry-run] example extraction prompt:\n{example[:600]}')
        return {'dry_run': True, 'materials': len(parents), 'conditions': conds, **caller.summary()}
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for n, mid in enumerate(parents, 1):
        material = pool[mid]
        reply = caller.call(model=model, prompt=prompts.render(PROMPT, TEXT=material['text']), task='assess_quality')
        facts, a, b, ok, note = split_facts((reply or {}).get('text'))
        if not ok:
            failures.append({'material_id': mid, 'note': note})
            continue
        sets = {'A': a, 'B': b, 'F': list(range(len(facts))), 'none': []}
        for t_key, i_key in conds:
            text_ids, image_ids = sets[t_key], sets[i_key]
            tag = f'{mid}~{t_key}_{i_key}'
            image, risk = image_for(material=material, facts=facts, ids=image_ids, side=i_key, backend=image_backend,
                                    out_dir=out / 'images', tag=tag)
            if risk.get('image') == 'unsupported':
                failures.append({'material_id': mid, 'condition': f'{t_key},{i_key}', 'note': risk['reason']})
                continue
            rows.append({'material_id': tag, 'parent_material_id': mid, 'group': material.get('group') or 'factsplit',
                         'text': facts_text(facts, text_ids), 'image_path': image, 'n_text_choices': None,
                         'condition': f'{t_key},{i_key}', 'facts': facts, 'A': a, 'B': b,
                         'image_backend': image_backend, 'risk': risk})
        print(f'[factsplit] {n}/{len(parents)} {mid}: {len(facts)} facts, A={len(a)} B={len(b)}', flush=True)
    mats_path = out / 'conditions_materials.jsonl'
    keep = [{k: v for k, v in r.items() if k not in ('facts', 'A', 'B', 'condition', 'risk', 'image_backend')} for r in rows]
    write_jsonl(mats_path, keep)
    write_json(out / 'factsplit.json', {'run_id': run_id, 'materials': len(parents), 'conditions': conds,
                                        'rows': [{k: r[k] for k in ('material_id', 'parent_material_id', 'condition', 'risk')} for r in rows],
                                        'failures': failures, 'provider': {**caller.summary()},
                                        'note': 'render images contain text -> OCR shortcut; existing backend assumes the original image carries B'})
    return {'run_id': run_id, 'conditions': len(rows), 'materials_file': str(mats_path), 'failures': failures, **caller.summary()}
