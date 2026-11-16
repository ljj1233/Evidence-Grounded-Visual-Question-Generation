"""Controlled synthetic composition benchmark: derived (serial), bridging (serial), stitching (parallel).

Every material ships the full question and two single-hop questions with programmatic answers, so
composition gaps and dual-generation rates can be scored without a judge.
"""
from __future__ import annotations

import json
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .local_vision import BatchingProvider, QwenVLBackend, qwen_identity
from .prompts import PromptSet
from .providers import CompletionRequest
from .records import canonical_json, read_jsonl, sha256_file, sha256_text, stable_id, utc_now, write_json, write_jsonl

CATEGORIES = ('derived', 'bridging', 'stitching')
KNOWN_CATEGORIES = ('derived', 'bridging', 'stitching', 'interval')
COLUMNS = ('left', 'middle', 'right')
ANSWER_PROMPTS = {'direct': 'composition-answer-v1', 'two-step': 'composition-two-step-v1'}
MATERIALS_DIR = 'data/processed/synthetic-composition-v1'

_COLUMN_ALIASES = {
    'left': ('left', 'leftmost', 'first', '1', 'column 1', 'column one'),
    'middle': ('middle', 'center', 'centre', 'central', 'second', '2', 'column 2', 'column two'),
    'right': ('right', 'rightmost', 'third', 'last', '3', 'column 3', 'column three'),
}


def _respond(question, kind, expected):
    return {'question': question, 'answer_kind': kind, 'expected': expected}


def _base_material(category, index, values, text, q, full, text_hop, image_hop, seed, extra=None):
    material = {
        'schema_version': 'synthetic-composition-v1',
        'material_id': f'syn-{category}-{index:03d}',
        'category': category,
        'text': text,
        'q': q,
        'values': values,
        'seed': seed,
        'evaluation': {'full': full, 'text_hop': text_hop, 'image_hop': image_hop},
    }
    if extra:
        material.update(extra)
    return material


def _draw_table(path, headers, rows, width=520, height=300):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    ncols = len(headers)
    left, top, right, bottom = 24, 24, width - 24, height - 24
    col_width = (right - left) / ncols
    nrows = 1 + len(rows)
    row_height = (bottom - top) / nrows
    for r in range(nrows + 1):
        y = top + r * row_height
        draw.line([(left, y), (right, y)], fill='black', width=2)
    for c in range(ncols + 1):
        x = left + c * col_width
        draw.line([(x, top), (x, bottom)], fill='black', width=2)
    for r, row in enumerate([headers] + rows):
        for c, value in enumerate(row):
            text = str(value)
            box = draw.textbbox((0, 0), text, font=font)
            tw, th = box[2] - box[0], box[3] - box[1]
            x = left + c * col_width + (col_width - tw) / 2
            y = top + r * row_height + (row_height - th) / 2 - box[1]
            draw.text((x, y), text, fill='black', font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def build_synthetic_composition(*, project_root='.', output_dir=MATERIALS_DIR, per_category=60, seed=20261115,
                                derived_full='sum', categories=('derived', 'bridging', 'stitching'),
                                value_min=2, value_max=21):
    if derived_full not in ('sum', 'count'):
        raise ValueError('derived_full must be sum or count')
    allowed = {'derived', 'bridging', 'stitching', 'interval'}
    if not categories or not set(categories) <= allowed:
        raise ValueError('categories must be a non-empty subset of derived/bridging/stitching/interval')
    if value_max - value_min < 6:
        raise ValueError('value range too narrow')
    root = Path(project_root).resolve()
    out = root / output_dir
    images = out / 'images'
    rng = random.Random(seed)
    materials = []
    for cat_index, category in enumerate(categories):
        for index in range(per_category):
            material_seed = seed + cat_index * 100003 + index
            if category == 'derived':
                while True:
                    values = rng.sample(range(value_min, value_max), 4)
                    ordered = sorted(values)
                    three_count = rng.random() < 0.5
                    candidates = (list(range(ordered[0] + 1, ordered[1])) if three_count
                                  else list(range(ordered[1] + 1, ordered[2])))
                    if candidates:
                        break
                threshold = rng.choice(candidates)
                text = f'Only items with value at least {threshold} count.'
                if derived_full == 'sum':
                    q = 'What is the total value of the items that count?'
                    full_answer = sum(v for v in values if v >= threshold)
                else:
                    q = 'How many items count?'
                    full_answer = sum(v >= threshold for v in values)
                text_question = 'What is the minimum value an item must have to count?'
                image_question = 'What are the values of items A, B, C, and D?'
                material = _base_material(category, index, values, text, q,
                    _respond(q, 'numeric', full_answer),
                    _respond(text_question, 'numeric', threshold),
                    _respond(image_question, 'item_values', values),
                    material_seed, {'threshold': threshold, 'derived_full': derived_full})
                _draw_table(images / f'{material["material_id"]}.png', ('item', 'value'),
                            [('A', values[0]), ('B', values[1]), ('C', values[2]), ('D', values[3])])
            elif category == 'bridging':
                values = rng.sample(range(value_min, value_max), 3)
                column = rng.choice(COLUMNS)
                text = f'The value in the {column} column is called alpha.'
                q = 'What is the value of alpha?'
                expected = values[COLUMNS.index(column)]
                text_question = 'Which column holds the value called alpha?'
                image_question = f'What is the value in the {column} column?'
                material = _base_material(category, index, values, text, q,
                    _respond(q, 'numeric', expected),
                    _respond(text_question, 'column', column),
                    _respond(image_question, 'numeric', expected),
                    material_seed, {'column': column})
                _draw_table(images / f'{material["material_id"]}.png', COLUMNS, [tuple(values)])
            elif category == 'stitching':
                values = rng.sample(range(value_min, value_max), 3)
                while True:
                    threshold = rng.randint(3, 7)
                    if threshold != values[2]:
                        break
                text = f'Only items with value at least {threshold} count.'
                q = 'What is the rule threshold, and what is the value of item C?'
                text_question = 'What is the rule threshold?'
                image_question = 'What is the value of item C?'
                material = _base_material(category, index, values, text, q,
                    _respond(q, 'both_numbers', [threshold, values[2]]),
                    _respond(text_question, 'numeric', threshold),
                    _respond(image_question, 'numeric', values[2]),
                    material_seed, {'threshold': threshold})
                _draw_table(images / f'{material["material_id"]}.png', ('item', 'value'),
                            [('A', values[0]), ('B', values[1]), ('C', values[2])])
            else:
                while True:
                    values = rng.sample(range(value_min, value_max), 4)
                    ordered = sorted(values)
                    lower_candidates = list(range(ordered[0] + 1, ordered[1]))
                    if rng.random() < 0.5:
                        upper_candidates = list(range(ordered[2] + 1, ordered[3]))
                    else:
                        upper_candidates = list(range(ordered[1] + 1, ordered[2]))
                    if lower_candidates and upper_candidates:
                        break
                lower, upper = rng.choice(lower_candidates), rng.choice(upper_candidates)
                text = f'Only items with value from {lower} to {upper} count.'
                if derived_full == 'sum':
                    q = 'What is the total value of the items that count?'
                    full_answer = sum(v for v in values if lower <= v <= upper)
                else:
                    q = 'How many items count?'
                    full_answer = sum(lower <= v <= upper for v in values)
                text_question = 'What are the lower and upper limits for an item to count?'
                image_question = 'What are the values of items A, B, C, and D?'
                material = _base_material(category, index, values, text, q,
                    _respond(q, 'numeric', full_answer),
                    _respond(text_question, 'both_numbers', [lower, upper]),
                    _respond(image_question, 'item_values', values),
                    material_seed, {'lower': lower, 'upper': upper, 'derived_full': derived_full})
                _draw_table(images / f'{material["material_id"]}.png', ('item', 'value'),
                            [('A', values[0]), ('B', values[1]), ('C', values[2]), ('D', values[3])])
            image_path = images / f'{material["material_id"]}.png'
            material['image_path'] = str(image_path.relative_to(root))
            material['image_sha256'] = sha256_file(image_path)
            materials.append(material)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'materials.jsonl', materials)
    write_json(out / 'provenance.json', {
        'created_at': utc_now(), 'seed': seed, 'per_category': per_category, 'derived_full': derived_full,
        'categories': list(categories), 'value_min': value_min, 'value_max': value_max, 'n_materials': len(materials),
        'design': {
            'derived': 'text gives the numeric condition; image gives four values; threshold determines whether the second-smallest value counts (serial)',
            'bridging': 'text maps a value to a column name; image gives the column values; full answer needs both (serial)',
            'stitching': 'text gives the threshold and image gives item C; the two hops are independent (parallel)',
            'interval': 'text gives a two-sided inclusive interval; image gives four values; both limits are needed to decide which values count (serial)',
        },
        'programmatic_only': True,
    })
    return {'n_materials': len(materials), 'path': str(out / 'materials.jsonl'), 'images': str(images)}


def load_composition_materials(project_root, materials_path):
    root = Path(project_root).resolve()
    rows = read_jsonl(root / materials_path)
    if not rows:
        raise ValueError('no synthetic materials')
    return rows


def _extract_json(raw):
    match = re.search(r'\{.*\}', raw, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _answer_text(raw):
    obj = _extract_json(raw)
    if obj is not None:
        if obj.get('abstain') is True:
            return None
        answer = obj.get('answer')
        return None if answer is None else str(answer)
    return raw


def _ints(text):
    return [int(x) for x in re.findall(r'(?<!\d)-?\d+(?!\d)', text)]


def _item_values(text):
    found = {}
    for match in re.finditer(r'\b([ABCD])\b[^0-9\-]{0,15}(-?\d+)', text, re.I):
        found.setdefault(match.group(1).upper(), int(match.group(2)))
    for match in re.finditer(r'(-?\d+)[^0-9]{0,15}\b([ABCD])\b', text, re.I):
        found.setdefault(match.group(2).upper(), int(match.group(1)))
    return found


def score_answer(raw, kind, expected):
    answer = _answer_text(raw)
    if answer is None:
        return False, None, 'abstain_or_missing'
    if kind == 'numeric':
        numbers = _ints(answer)
        return (expected in numbers, numbers, '' if expected in numbers else 'number_not_found')
    if kind == 'item_values':
        mapping = _item_values(answer)
        letters = [chr(65 + i) for i in range(len(expected))]
        if all(letter in mapping for letter in letters):
            ok = [mapping[letter] for letter in letters] == list(expected)
            return ok, mapping, '' if ok else 'item_mapping_mismatch'
        numbers = _ints(answer)
        ok = sorted(numbers) == sorted(expected)
        return ok, numbers, '' if ok else 'value_multiset_mismatch'
    if kind == 'both_numbers':
        numbers = set(_ints(answer))
        ok = all(int(x) in numbers for x in expected)
        return ok, sorted(numbers), '' if ok else 'missing_required_number'
    if kind == 'column':
        text = answer.lower()
        aliases = _COLUMN_ALIASES[expected]
        ok = any(re.search(r'(?<![a-z])' + re.escape(alias) + r'(?![a-z])', text) for alias in aliases)
        return ok, answer, '' if ok else 'column_mismatch'
    raise ValueError(f'unknown answer kind: {kind}')


def _append_jsonl(path, row, lock):
    with lock:
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def _answer_plan(run_id, materials, condition):
    plan = []
    for material in materials:
        for qtype in ('full', 'text_hop', 'image_hop'):
            evaluation = material['evaluation'][qtype]
            plan.append({
                'attempt_id': stable_id('comp-answer', run_id, material['material_id'], qtype),
                'run_id': run_id, 'material_id': material['material_id'], 'category': material['category'],
                'question_type': qtype, 'condition': condition, 'question': evaluation['question'],
                'answer_kind': evaluation['answer_kind'], 'expected': evaluation['expected'],
            })
    return plan


def run_composition_answers(*, run_id, materials_path=MATERIALS_DIR, model_path, device='cuda:0',
                            prompt='direct', max_batch=8, max_new_tokens=256, limit=None,
                            project_root='.', output_root='outputs'):
    if prompt not in ANSWER_PROMPTS:
        raise ValueError('prompt must be direct or two-step')
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    template = ANSWER_PROMPTS[prompt]
    prompts = PromptSet(root / 'prompts', names=(template,))
    materials = load_composition_materials(root, materials_path)
    by_id = {m['material_id']: m for m in materials}
    plan = _answer_plan(run_id, materials, 'TI' if prompt == 'direct' else 'TI-two-step')
    if limit:
        plan = plan[:limit]
    prior = {row['attempt_id'] for row in read_jsonl(out / 'answers.jsonl')} if (out / 'answers.jsonl').exists() else set()
    pending = [slot for slot in plan if slot['attempt_id'] not in prior]
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens)
    identity = qwen_identity(backend, f'composition-answer-{prompt}-v1')
    manifest_path = out / 'manifest_answer.json'
    fingerprint = sha256_text(canonical_json({'config': {'prompt': prompt, 'max_new_tokens': max_new_tokens},
        'plan': plan, 'materials': sha256_file(root / materials_path), 'prompts': prompts.snapshot(), 'identity': identity}))
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        if stored['fingerprint'] != fingerprint:
            raise ValueError('answer run configuration changed')
    else:
        write_json(manifest_path, {'created_at': utc_now(), 'fingerprint': fingerprint,
            'config': {'prompt': prompt, 'max_new_tokens': max_new_tokens, 'device': device},
            'materials_path': materials_path, 'model_identity': identity, 'prompt_snapshot': prompts.snapshot()})
    write_jsonl(out / 'answer_plan.jsonl', plan)
    if not pending:
        return {'run_id': run_id, 'planned': len(plan), 'already_done': len(prior), 'new': 0}
    provider = BatchingProvider(backend, model=backend.path.name, identity=identity, max_batch=max_batch)
    lock = threading.Lock()
    done = 0

    def work(slot):
        material = by_id[slot['material_id']]
        rendered = prompts.render(template, T=material['text'], Q=slot['question'])
        request = CompletionRequest(task=f'composition-answer-{prompt}', prompt=rendered, question=slot['question'],
                                    text=material['text'], image_path=str(root / material['image_path']), condition='TI',
                                    metadata={'material_id': slot['material_id'], 'question_type': slot['question_type']})
        completion = provider.complete(request)
        correct, parsed, reason = score_answer(completion.text, slot['answer_kind'], slot['expected'])
        row = {**slot, 'raw_output': completion.text, 'parsed_answer': parsed, 'correct': correct,
               'score_reason': reason, 'finish_reason': completion.finish_reason,
               'usage': completion.usage, 'completed_at': utc_now()}
        _append_jsonl(out / 'answers.jsonl', row, lock)
        return correct

    with ThreadPoolExecutor(max_workers=max(2, max_batch * 2)) as pool:
        futures = [pool.submit(work, slot) for slot in pending]
        for future in as_completed(futures):
            future.result()
            done += 1
    return {'run_id': run_id, 'planned': len(plan), 'already_done': len(prior), 'new': done}


def classify_generated_question(question, material):
    q = question.lower()
    has_value = 'value' in q
    has_item = re.search(r'item\s+[abc]\b|\b[abc]\s+(?:is|=|equals)', q) is not None
    has_threshold = re.search(r'threshold|at least|minimum value|greater than|more than|above|exceed|rule', q) is not None
    asks_threshold = re.search(r'minimum value|threshold|what is the (?:rule|minimum)', q) is not None
    asks_which_items = re.search(r'which item|what items|how many items|items? (?:have|meet|satisfy|with|whose|count)|item[s]? that', q) is not None
    has_count = re.search(r'count|qualif', q) is not None
    has_total = 'total' in q
    has_alpha = 'alpha' in q
    has_explicit_number = re.search(r'\d', q) is not None or re.search(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty(?:-\w+)?|thirty(?:-\w+)?|forty(?:-\w+)?)\b', q) is not None
    refers_rule = re.search(r'condition|rule|criteria|criterion|satisf|require', q) is not None
    if has_alpha and has_value:
        return 'dual_bridging'
    if has_threshold and has_item and has_value:
        return 'dual_stitching'
    if has_threshold and asks_which_items:
        return 'single_image' if has_explicit_number else 'dual_derived'
    if asks_which_items and refers_rule and not has_explicit_number:
        return 'dual_derived'
    if has_total and has_count:
        return 'dual_derived'
    if has_count and asks_which_items and not has_value:
        return 'dual_derived'
    if has_count and has_value and not has_threshold:
        return 'dual_derived'
    if has_threshold and asks_threshold and not asks_which_items:
        return 'single_text'
    if has_item or (has_value and 'column' in q):
        return 'single_image'
    if has_total or has_value:
        return 'single_image'
    return 'other'


def _generation_plan(run_id, materials, n_questions):
    plan = []
    for material in materials:
        for index in range(n_questions):
            plan.append({'attempt_id': stable_id('comp-gen', run_id, material['material_id'], index),
                         'run_id': run_id, 'material_id': material['material_id'], 'category': material['category'],
                         'sample_index': index})
    return plan


def run_composition_generation(*, run_id, materials_path=MATERIALS_DIR, model_path, device='cuda:0',
                               n_questions=4, temperature=0.7, top_p=0.8, top_k=20,
                               max_batch=8, max_new_tokens=96, limit=None, template='plain-v2',
                               suppress_numbers=False, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    prompts = PromptSet(root / 'prompts', names=(template,))
    materials = load_composition_materials(root, materials_path)
    by_id = {m['material_id']: m for m in materials}
    plan = _generation_plan(run_id, materials, n_questions)
    if limit:
        plan = plan[:limit]
    prior = {row['attempt_id'] for row in read_jsonl(out / 'generated.jsonl')} if (out / 'generated.jsonl').exists() else set()
    pending = [slot for slot in plan if slot['attempt_id'] not in prior]
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens,
                            do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                            suppress_text_numbers=suppress_numbers)
    identity = qwen_identity(backend, 'composition-generation-v1', do_sample=True, temperature=temperature,
                             top_p=top_p, top_k=top_k, template=template, suppress_text_numbers=suppress_numbers)
    manifest_path = out / 'manifest_generation.json'
    fingerprint = sha256_text(canonical_json({'config': {'n_questions': n_questions, 'temperature': temperature,
        'top_p': top_p, 'top_k': top_k, 'max_new_tokens': max_new_tokens, 'template': template, 'suppress_numbers': suppress_numbers}, 'plan': plan,
        'materials': sha256_file(root / materials_path), 'prompts': prompts.snapshot(), 'identity': identity}))
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('generation run configuration changed')
    else:
        write_json(manifest_path, {'created_at': utc_now(), 'fingerprint': fingerprint,
            'config': {'n_questions': n_questions, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k, 'template': template, 'suppress_numbers': suppress_numbers},
            'materials_path': materials_path, 'model_identity': identity, 'prompt_snapshot': prompts.snapshot()})
    write_jsonl(out / 'generation_plan.jsonl', plan)
    if not pending:
        return {'run_id': run_id, 'planned': len(plan), 'already_done': len(prior), 'new': 0}
    provider = BatchingProvider(backend, model=backend.path.name, identity=identity, max_batch=max_batch)
    lock = threading.Lock()
    done = 0

    def work(slot):
        material = by_id[slot['material_id']]
        rendered = prompts.render(template, T=material['text'])
        request = CompletionRequest(task='composition-generation', prompt=rendered, text=material['text'],
                                    image_path=str(root / material['image_path']), condition='TI',
                                    metadata={'material_id': slot['material_id'], 'sample_index': str(slot['sample_index'])})
        completion = provider.complete(request)
        question = completion.text.strip().replace('\n', ' ')
        label = classify_generated_question(question, material)
        row = {**slot, 'question': question, 'generated_label': label, 'finish_reason': completion.finish_reason,
               'usage': completion.usage, 'completed_at': utc_now()}
        _append_jsonl(out / 'generated.jsonl', row, lock)
        return label

    with ThreadPoolExecutor(max_workers=max(2, max_batch * 2)) as pool:
        futures = [pool.submit(work, slot) for slot in pending]
        for future in as_completed(futures):
            future.result()
            done += 1
    return {'run_id': run_id, 'planned': len(plan), 'already_done': len(prior), 'new': done}


def _present_categories(rows):
    seen = [category for category in KNOWN_CATEGORIES if any(row['category'] == category for row in rows)]
    extra = sorted({row['category'] for row in rows} - set(seen))
    return seen + extra


def _material_stats(answers):
    grouped = defaultdict(dict)
    for row in answers:
        grouped[row['material_id']][row['question_type']] = row
    stats = []
    for material_id, rows in grouped.items():
        material = next(iter(rows.values()))
        stats.append({
            'material_id': material_id, 'category': material['category'],
            'full': bool(rows.get('full', {}).get('correct')),
            'text_hop': bool(rows.get('text_hop', {}).get('correct')),
            'image_hop': bool(rows.get('image_hop', {}).get('correct')),
            'full_row': rows.get('full'), 'text_row': rows.get('text_hop'), 'image_row': rows.get('image_hop'),
        })
    return stats


def _metrics(stats):
    n = len(stats)
    if not n:
        return {'n': 0, 'acc_full': None, 'acc_text_hop': None, 'acc_image_hop': None,
                'product': None, 'gap': None, 'deficit': None, 'both_hops_correct': 0, 'both_hops_full_wrong': 0,
                'full_wrong_given_both_hops': None}
    acc_full = sum(s['full'] for s in stats) / n
    acc_text = sum(s['text_hop'] for s in stats) / n
    acc_image = sum(s['image_hop'] for s in stats) / n
    both_ok = [s for s in stats if s['text_hop'] and s['image_hop']]
    product = acc_text * acc_image
    return {'n': n, 'acc_full': acc_full, 'acc_text_hop': acc_text, 'acc_image_hop': acc_image,
            'product': product, 'gap': acc_full - product, 'deficit': product - acc_full, 'both_hops_correct': len(both_ok),
            'both_hops_full_wrong': sum(1 for s in both_ok if not s['full']),
            'full_wrong_given_both_hops': (sum(1 for s in both_ok if not s['full']) / len(both_ok)) if both_ok else None}


def _qualifying_values(material):
    values = material.get('values') or []
    if material.get('lower') is not None and material.get('upper') is not None:
        return [v for v in values if material['lower'] <= v <= material['upper']]
    threshold = material.get('threshold')
    if threshold is None:
        return None
    return [v for v in values if v >= threshold]


def _derived_error_mode(row, material):
    values = material.get('values') or []
    qualifying = _qualifying_values(material)
    if qualifying is None or len(values) < 3:
        return None
    parsed = row.get('parsed_answer')
    numbers = parsed if isinstance(parsed, list) else ([parsed] if isinstance(parsed, int) else [])
    if material.get('derived_full') == 'count' or (material.get('category') and row.get('question', '').startswith('How many')):
        if len(numbers) == 1:
            number = numbers[0]
            if number == len(values):
                return 'counted_all_items'
            if number == 0:
                return 'counted_nothing'
            return 'wrong_count'
        return 'list_not_count'
    target = sum(qualifying)
    if len(numbers) == 1:
        number = numbers[0]
        if number == sum(values):
            return 'included_all_values'
        if number == target:
            return 'correct'
        matches = []
        for mask in range(1, 1 << len(values)):
            subset = [values[i] for i in range(len(values)) if mask & (1 << i)]
            if sum(subset) == number:
                matches.append(set(subset))
        for subset in matches:
            if subset == set(values):
                return 'included_all_values'
            if subset == set(qualifying):
                return 'correct'
            if subset > set(qualifying):
                return 'included_extra_value'
            if subset < set(qualifying):
                return 'omitted_qualifying_value'
        if matches:
            return 'mixed_subset'
        return 'other_numeric'
    if sorted(numbers) == sorted(qualifying):
        return 'listed_qualifying_not_summed'
    if sorted(numbers) == sorted(values):
        return 'listed_all_values_not_summed'
    return 'other_list'


def _bootstrap_gap(stats, B=10000, seed=20261115):
    if not stats:
        return None
    rng = random.Random(seed)
    n = len(stats)
    gaps, fulls, products = [], [], []
    for _ in range(B):
        sample = [stats[rng.randrange(n)] for _ in range(n)]
        m = _metrics(sample)
        gaps.append(m['gap'])
        fulls.append(m['acc_full'])
        products.append(m['product'])
    gaps.sort()
    return {'bootstrap': B, 'gap_ci_95': [gaps[int(0.025 * B)], gaps[int(0.975 * B) - 1]],
            'full_ci_95': [sorted(fulls)[int(0.025 * B)], sorted(fulls)[int(0.975 * B) - 1]],
            'product_ci_95': [sorted(products)[int(0.025 * B)], sorted(products)[int(0.975 * B) - 1]],
            'deficit_ci_95': [-gaps[int(0.975 * B) - 1], -gaps[int(0.025 * B)]]}


def _fmt(value, pattern='{:.3f}'):
    return 'n/a' if value is None else pattern.format(value)


def analyze_composition_answers(*, run_id, B=10000, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    rows = read_jsonl(out / 'answers.jsonl')
    if not rows:
        raise ValueError('no answers to analyze')
    stats = _material_stats(rows)
    manifest = json.loads((out / 'manifest_answer.json').read_text())
    categories = {}
    for category in _present_categories(stats):
        subset = [s for s in stats if s['category'] == category]
        metrics = _metrics(subset)
        metrics.update(_bootstrap_gap(subset, B=B) or {})
        metrics['ci_includes_zero'] = bool(metrics.get('gap_ci_95') and metrics['gap_ci_95'][0] <= 0 <= metrics['gap_ci_95'][1])
        categories[category] = metrics
    pooled = {'n': len(stats), 'metrics': _metrics(stats)}
    stop = {'derived_ci_includes_zero': categories['derived']['ci_includes_zero'],
            'bridging_ci_includes_zero': categories['bridging']['ci_includes_zero']}
    stop['stop_mechanism'] = stop['derived_ci_includes_zero'] and stop['bridging_ci_includes_zero']
    if 'interval' in categories:
        stop['interval_ci_includes_zero'] = categories['interval']['ci_includes_zero']
    report = {'run_id': run_id, 'model': manifest['model_identity']['model'],
              'prompt': manifest['config']['prompt'], 'n_answers': len(rows),
              'categories': categories, 'pooled': pooled, 'stop_loss': stop}
    write_json(out / 'composition_report.json', report)
    mismatch = [s for s in stats if s['text_hop'] and s['image_hop'] and not s['full']]
    materials = {m['material_id']: m for m in load_composition_materials(root, manifest['materials_path'])}
    mismatch_rows = [{
        'material_id': s['material_id'], 'category': s['category'],
        'full_question': s['full_row']['question'], 'full_expected': s['full_row']['expected'],
        'full_answer': s['full_row']['parsed_answer'], 'full_raw': s['full_row']['raw_output'],
        'text_hop_question': s['text_row']['question'], 'text_hop_answer': s['text_row']['parsed_answer'],
        'image_hop_question': s['image_row']['question'], 'image_hop_answer': s['image_row']['parsed_answer'],
        'error_mode': _derived_error_mode(s['full_row'], materials[s['material_id']]) if s['category'] in ('derived', 'interval') else None,
    } for s in mismatch]
    write_jsonl(out / 'both_hops_correct_full_wrong.jsonl', mismatch_rows)
    error_modes = {}
    error_modes_by_category = {}
    for row in mismatch_rows:
        if row['error_mode']:
            error_modes[row['error_mode']] = error_modes.get(row['error_mode'], 0) + 1
            bucket = error_modes_by_category.setdefault(row['category'], {})
            bucket[row['error_mode']] = bucket.get(row['error_mode'], 0) + 1
    report['error_modes'] = error_modes
    report['error_modes_by_category'] = error_modes_by_category
    write_json(out / 'composition_report.json', report)
    lines = [f'# Composition gap: {run_id}', '',
             f'- model: `{report["model"]}`', f'- prompt: `{report["prompt"]}`',
             f'- answers: {len(rows)}; materials: {len(stats)}; bootstrap: {B}', '',
             'Deficit = acc(text hop) * acc(image hop) - acc(full question); positive means the full question is harder than both hops together. 95% CI from material bootstrap.', '',
             '| category | n | acc full | acc text hop | acc image hop | product | deficit (product - full) | deficit 95% CI | both hops ok | both ok full wrong | P(full wrong \\| both ok) |',
             '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for category in list(categories):
        m = categories[category]
        ci = m.get('deficit_ci_95')
        ci_text = f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else 'n/a'
        p = m['full_wrong_given_both_hops']
        lines.append(f'| {category} | {m["n"]} | {_fmt(m["acc_full"])} | {_fmt(m["acc_text_hop"])} | {_fmt(m["acc_image_hop"])} | {_fmt(m["product"])} | {_fmt(m["deficit"], "{:+.3f}")} | {ci_text} | {m["both_hops_correct"]} | {m["both_hops_full_wrong"]} | {_fmt(p)} |')
    lines += ['', f'Stop-loss: derived CI includes 0 = {stop["derived_ci_includes_zero"]}; '
              f'bridging CI includes 0 = {stop["bridging_ci_includes_zero"]}; '
              f'stop mechanism direction = {stop["stop_mechanism"]}; interval CI includes 0 = {stop.get("interval_ci_includes_zero", "n/a")}.', '',
              f'Sample list: `both_hops_correct_full_wrong.jsonl` ({len(mismatch)} rows).', '',
              f'Error modes by category among both-hops-correct-but-full-wrong: {error_modes_by_category}.', '']
    (out / 'composition_report.md').write_text('\n'.join(lines))
    return report


def analyze_composition_generation(*, run_id, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    rows = read_jsonl(out / 'generated.jsonl')
    if not rows:
        raise ValueError('no generated questions to analyze')
    labels = ('dual_bridging', 'dual_stitching', 'dual_derived', 'single_text', 'single_image', 'other')
    manifest = json.loads((out / 'manifest_generation.json').read_text())
    materials = {m['material_id']: m for m in load_composition_materials(root, manifest['materials_path'])}
    changed = False
    for row in rows:
        label = classify_generated_question(row['question'], {})
        if row.get('generated_label') != label:
            row['generated_label'] = label
            changed = True
    if changed:
        write_jsonl(out / 'generated.jsonl', rows)
    by_category = {}
    for category in _present_categories(rows):
        subset = [r for r in rows if r['category'] == category]
        counts = {label: sum(r['generated_label'] == label for r in subset) for label in labels}
        dual = sum(counts[k] for k in ('dual_bridging', 'dual_stitching', 'dual_derived'))
        parsed = dual + counts['single_text'] + counts['single_image']
        threshold_copies = 0
        if category in ('derived', 'stitching', 'interval'):
            for row in subset:
                material = materials[row['material_id']]
                numbers = [int(x) for x in re.findall(r'\d+', row['question'])]
                if category == 'interval':
                    if material.get('lower') in numbers and material.get('upper') in numbers:
                        threshold_copies += 1
                elif material.get('threshold') is not None and material['threshold'] in numbers:
                    threshold_copies += 1
        by_category[category] = {'n': len(subset), 'counts': counts, 'dual': dual,
                                 'dual_rate_all': dual / len(subset) if subset else None,
                                 'dual_rate_parsed': dual / parsed if parsed else None,
                                 'other_rate': counts['other'] / len(subset) if subset else None,
                                 'threshold_copy': threshold_copies if category in ('derived', 'stitching', 'interval') else None,
                                 'threshold_copy_rate': (threshold_copies / len(subset)) if subset and category in ('derived', 'stitching', 'interval') else None}
    dual = sum(1 for r in rows if r['generated_label'].startswith('dual_'))
    parsed = sum(1 for r in rows if r['generated_label'] != 'other')
    report = {'run_id': run_id, 'n': len(rows), 'dual': dual,
              'dual_rate_all': dual / len(rows), 'dual_rate_parsed': dual / parsed if parsed else None,
              'by_category': by_category}
    write_json(out / 'generation_report.json', report)
    lines = [f'# Dual-generation rate: {run_id}', '',
             f'- generated questions: {len(rows)}', f'- overall dual rate (all): {report["dual_rate_all"]:.3f}',
             f'- overall dual rate (parsed): {_fmt(report["dual_rate_parsed"])}', '',
             '| category | n | dual | dual rate (all) | dual rate (parsed) | other rate | exact threshold in Q |', '|---|---:|---:|---:|---:|---:|---:|']
    for category in list(by_category):
        c = by_category[category]
        lines.append(f'| {category} | {c["n"]} | {c["dual"]} | {c["dual_rate_all"]:.3f} | '
                     f'{_fmt(c["dual_rate_parsed"])} | {c["other_rate"]:.3f} | {_fmt(c["threshold_copy_rate"])} |')
    lines += ['', 'Labels are programmatic grammar labels over the synthetic materials; `other` questions are excluded '
              'from the parsed rate and reported separately.', '']
    (out / 'generation_report.md').write_text('\n'.join(lines))
    return report


def compare_composition_prompts(*, run_id, direct_run, two_step_run, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    direct = json.loads((root / output_root / direct_run / 'composition_report.json').read_text())
    two_step = json.loads((root / output_root / two_step_run / 'composition_report.json').read_text())
    categories = {}
    for category in [c for c in KNOWN_CATEGORIES if c in direct['categories'] and c in two_step['categories']]:
        d, t = direct['categories'][category], two_step['categories'][category]
        product_direct = None if d['product'] is None else d['product']
        gap_mixed = None if t['acc_full'] is None or product_direct is None else t['acc_full'] - product_direct
        categories[category] = {
            'direct_acc_full': d['acc_full'], 'two_step_acc_full': t['acc_full'],
            'delta_acc_full': None if t['acc_full'] is None or d['acc_full'] is None else t['acc_full'] - d['acc_full'],
            'direct_deficit': d['deficit'], 'two_step_deficit': t['deficit'],
            'deficit_change': None if t['deficit'] is None or d['deficit'] is None else t['deficit'] - d['deficit'],
            'two_step_full_minus_direct_hop_product': gap_mixed,
            'direct_hops': {'text': d['acc_text_hop'], 'image': d['acc_image_hop']},
            'two_step_hops': {'text': t['acc_text_hop'], 'image': t['acc_image_hop']},
            'two_step_both_hops_full_wrong': t['both_hops_full_wrong'],
        }
    report = {'run_id': run_id, 'direct_run': direct_run, 'two_step_run': two_step_run, 'categories': categories}
    write_json(out / 'prompt_comparison.json', report)
    lines = [f'# Two-step prompting comparison: {run_id}', '',
             f'- direct: `{direct_run}`', f'- two-step: `{two_step_run}`', '',
             'Primary readout is the change in full-question accuracy. The mixed deficit uses two-step full accuracy '
             'with the direct run\'s single-hop product, so it isolates the full-question effect from the prompt\'s '
             'effect on hop extraction.', '',
             '| category | full direct | full two-step | delta full | deficit direct | deficit two-step | deficit change | mixed deficit |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for category in CATEGORIES:
        c = categories[category]
        lines.append(f'| {category} | {_fmt(c["direct_acc_full"])} | {_fmt(c["two_step_acc_full"])} | '
                     f'{_fmt(c["delta_acc_full"], "{:+.3f}")} | {_fmt(c["direct_deficit"], "{:+.3f}")} | '
                     f'{_fmt(c["two_step_deficit"], "{:+.3f}")} | {_fmt(c["deficit_change"], "{:+.3f}")} | '
                     f'{_fmt(c["two_step_full_minus_direct_hop_product"], "{:+.3f}")} |')
    lines += ['', '`deficit change` recomputes both-hop product under the two-step prompt; `mixed deficit` uses direct hop '
              'accuracy. Report both because two-step extraction can itself change single-hop accuracy.', '']
    (out / 'prompt_comparison.md').write_text('\n'.join(lines))
    return report


def compare_composition_runs(*, run_id, runs, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for label, source in runs.items():
        source_dir = root / output_root / source
        answer = json.loads((source_dir / 'composition_report.json').read_text())
        generation = None
        if (source_dir / 'generation_report.json').exists():
            generation = json.loads((source_dir / 'generation_report.json').read_text())
        entries.append({'label': label, 'source_run': source, 'model': answer['model'],
                        'deficit': {category: answer['categories'][category]['deficit'] for category in KNOWN_CATEGORIES if category in answer['categories']},
                        'dual_rate': None if generation is None else generation['dual_rate_all'],
                        'dual_rate_parsed': None if generation is None else generation['dual_rate_parsed']})
    serial_deficit = [(e['deficit']['derived'] + e['deficit'].get('bridging', e['deficit']['derived'])) / 2 for e in entries]
    report = {'run_id': run_id, 'entries': entries, 'serial_deficit': serial_deficit,
              'note': 'Descriptive only: three model sizes, no repeated seeds; no correlation statistic is reported.'}
    write_json(out / 'comparison.json', report)
    lines = [f'# Cross-size comparison: {run_id}', '',
             '| label | model | deficit derived | deficit bridging | deficit stitching | dual rate (all) | dual rate (parsed) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for e in entries:
        lines.append(f'| {e["label"]} | `{e["model"]}` | {e["deficit"]["derived"]:+.3f} | {e["deficit"]["bridging"]:+.3f} | '
                     f'{e["deficit"]["stitching"]:+.3f} | '
                     f'{_fmt(e["dual_rate"])} | '
                     f'{_fmt(e["dual_rate_parsed"])} |')
    lines += ['', 'No correlation statistic is reported (three model sizes, no repeated seeds; the generation rates are template-driven).', '']
    (out / 'comparison.md').write_text('\n'.join(lines))
    return report
