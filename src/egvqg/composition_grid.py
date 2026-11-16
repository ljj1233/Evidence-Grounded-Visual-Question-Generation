"""E2 modality grid: does the composition gap depend on *where* the two hops live?

Four cells place the same two facts in different modalities:

    TT  rule in text, values in text   (text only, no image)
    TI  rule in text, values in image  (the existing synthetic `derived` design)
    IT  rule in image, values in text
    II  rule in image, values in image (image only)

Every cell holds the same values, the same threshold and the same question, so the cells are paired by
material. Three questions are asked per material and cell: the full derivation ("total value of the items
that count"), the rule hop (threshold) and the value hop (the four item values). That separates a perception
failure from a composition failure: the composition gap is the full-question accuracy conditioned on both
hops being read correctly.

Everything here is programmatic (no judge): the answers and the expected values are exact integers.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .local_vision import BatchingProvider, QwenVLBackend, qwen_identity
from .prompts import PromptSet
from .providers import CompletionRequest
from .records import (canonical_json, read_jsonl, sha256_file, sha256_text, stable_id, utc_now, write_json,
                      write_jsonl)
from .stats import paired_bootstrap, wilson
from .synthetic_composition import _answer_text, _draw_table, score_answer

CELLS = ('TT', 'TI', 'IT', 'II')
SAME_MODALITY = ('TT', 'II')
CROSS_MODALITY = ('TI', 'IT')
GRID_DIR = 'data/processed/composition-grid-v1'
PROMPT = 'grid-answer-v1'
HOP_QUESTIONS = {
    'full': 'Which items count?',
    'count': 'How many items count?',
    'sum': 'What is the total value of the items that count?',
    'rule': 'What is the minimum value an item must have to count?',
    'values': 'What are the values of items A, B, C, and D?',
}
QUESTION_TYPES = ('full', 'count', 'sum', 'rule', 'values')


def score_items(raw, expected):
    """Order-insensitive scoring of "which items count?" (expected = list of letters)."""
    answer = _answer_text(raw)
    if answer is None:
        return False, None, 'abstain_or_missing'
    text = answer.upper()
    letters = set('ABCD')
    if re.fullmatch(r'[\s.]*(NONE|NO ITEMS?|NOTHING|0)[\s.]*', text):
        found = set()
    elif re.search(r'\b(ALL|EVERY)\b', text):
        found = letters
    else:
        found = set(re.findall(r'\b([ABCD])\b', text))
        found &= letters
    ok = found == (set(expected) & letters)
    return ok, sorted(found), '' if ok else 'item_set_mismatch'


def _write_rule_image(path, text, width=560, height=120, font_size=24):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=font_size)
    # wrap on word boundaries against the usable width
    usable = width - 40
    words = text.split()
    lines, current = [], ''
    for word in words:
        trial = f'{current} {word}'.strip()
        box = draw.textbbox((0, 0), trial, font=font)
        if box[2] - box[0] <= usable or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    line_height = (height - 20) / max(1, len(lines))
    for row, line in enumerate(lines):
        box = draw.textbbox((0, 0), line, font=font)
        x = (width - (box[2] - box[0])) / 2
        y = 10 + row * line_height + (line_height - (box[3] - box[1])) / 2 - box[1]
        draw.text((x, y), line, fill='black', font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _stack(rule_path, table_path, path):
    from PIL import Image
    rule = Image.open(rule_path)
    table = Image.open(table_path)
    width = max(rule.width, table.width)
    canvas = Image.new('RGB', (width, rule.height + table.height + 10), 'white')
    canvas.paste(rule, ((width - rule.width) // 2, 0))
    canvas.paste(table, ((width - table.width) // 2, rule.height + 10))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_grid_materials(*, project_root='.', output_dir=GRID_DIR, n_materials=30, seed=20261115,
                         value_min=2, value_max=21, cells=CELLS):
    root = Path(project_root).resolve()
    out = root / output_dir
    rng = random.Random(seed)
    rows = []
    for index in range(n_materials):
        while True:
            values = rng.sample(range(value_min, value_max), 4)
            ordered = sorted(values)
            candidates = list(range(ordered[0] + 1, ordered[1]))
            if candidates:
                break
        threshold = rng.choice(candidates)
        rule_text = f'Only items with value at least {threshold} count.'
        value_text = 'The item values are ' + ', '.join(
            f'{chr(65 + i)} = {value}' for i, value in enumerate(values)) + '.'
        material_id = f'grid-{index:03d}'
        rule_image = out / 'images' / f'{material_id}-rule.png'
        table_image = out / 'images' / f'{material_id}-values.png'
        combined_image = out / 'images' / f'{material_id}-both.png'
        _write_rule_image(rule_image, rule_text)
        _draw_table(table_image, ('item', 'value'),
                    [(chr(65 + i), value) for i, value in enumerate(values)])
        _stack(rule_image, table_image, combined_image)
        evaluate = lambda kind, qtype, expected: {'question': HOP_QUESTIONS[qtype], 'answer_kind': kind,
                                                  'expected': expected}
        counting = [chr(65 + i) for i, value in enumerate(values) if value >= threshold]
        for cell in cells:
            rule_text_side = cell[0] == 'T'
            value_text_side = cell[1] == 'T'
            text = ' '.join(part for part, keep in ((rule_text, rule_text_side), (value_text, value_text_side)) if keep)
            if rule_text_side and value_text_side:
                image = None
            elif rule_text_side:
                image = table_image
            elif value_text_side:
                image = rule_image
            else:
                image = combined_image
            rows.append({
                'schema_version': 'composition-grid-v1',
                'material_id': material_id,
                'cell': cell,
                'rule_modality': cell[0],
                'value_modality': cell[1],
                'text': text,
                'image_path': str(image.relative_to(root)) if image is not None else None,
                'q': HOP_QUESTIONS['full'],
                'values': values,
                'threshold': threshold,
                'seed': seed + index,
                'evaluation': {
                    'full': evaluate('items', 'full', counting),
                    'count': evaluate('numeric', 'count', len(counting)),
                    'sum': evaluate('numeric', 'sum', sum(v for v in values if v >= threshold)),
                    'rule': evaluate('numeric', 'rule', threshold),
                    'values': evaluate('item_values', 'values', values),
                },
            })
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'materials.jsonl', rows)
    write_json(out / 'provenance.json', {
        'created_at': utc_now(), 'seed': seed, 'n_materials': n_materials, 'cells': list(cells),
        'value_min': value_min, 'value_max': value_max, 'prompt': PROMPT,
        'design': {
            'cell_notation': 'first letter = modality holding the rule, second = modality holding the four values',
            'TT': 'text only: rule sentence and item values in the context, no image',
            'TI': 'rule in the context, item values in the image table',
            'IT': 'rule rendered into the image, item values in the context',
            'II': 'rule and item values both rendered into one image, empty context',
            'questions': HOP_QUESTIONS,
        },
        'programmatic_only': True,
    })
    return {'materials': len(rows), 'path': str(out / 'materials.jsonl')}


def _plan(run_id, rows):
    plan = []
    for row in rows:
        for qtype in QUESTION_TYPES:
            evaluation = row['evaluation'][qtype]
            plan.append({
                'attempt_id': stable_id('grid-answer', run_id, row['material_id'], row['cell'], qtype),
                'material_id': row['material_id'], 'cell': row['cell'], 'question_type': qtype,
                'question': evaluation['question'], 'answer_kind': evaluation['answer_kind'],
                'expected': evaluation['expected'],
            })
    return plan


def run_grid_answers(*, run_id, materials_path=GRID_DIR + '/materials.jsonl', model_path, device='cuda:0',
                     max_batch=8, max_new_tokens=192, limit=None, prompt=PROMPT, project_root='.',
                     output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (root / materials_path).read_text(encoding='utf-8').splitlines() if line]
    by_key = {(row['material_id'], row['cell']): row for row in rows}
    prompts = PromptSet(root / 'prompts', names=(prompt,))
    plan = _plan(run_id, rows)
    if limit:
        plan = plan[:limit]
    answers_path = out / 'answers.jsonl'
    prior = {row['attempt_id'] for row in read_jsonl(answers_path)} if answers_path.exists() else set()
    pending = [slot for slot in plan if slot['attempt_id'] not in prior]
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens)
    identity = qwen_identity(backend, 'composition-grid-answer-v1')
    manifest_path = out / 'manifest_answer.json'
    fingerprint = sha256_text(canonical_json({
        'config': {'prompt': prompt, 'max_new_tokens': max_new_tokens}, 'plan': plan,
        'materials': sha256_file(root / materials_path), 'prompts': prompts.snapshot(), 'identity': identity}))
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('grid answer configuration changed')
    else:
        write_json(manifest_path, {'created_at': utc_now(), 'fingerprint': fingerprint,
            'materials_path': materials_path, 'model_identity': identity,
            'config': {'prompt': prompt, 'max_new_tokens': max_new_tokens, 'device': device},
            'prompt_snapshot': prompts.snapshot()})
    write_jsonl(out / 'answer_plan.jsonl', plan)
    if not pending:
        return {'run_id': run_id, 'planned': len(plan), 'already_done': len(prior), 'new': 0}
    provider = BatchingProvider(backend, model=backend.path.name, identity=identity, max_batch=max_batch)
    import threading
    lock = threading.Lock()
    done = 0

    def work(slot):
        material = by_key[(slot['material_id'], slot['cell'])]
        rendered = prompts.render(prompt, T=material['text'], Q=slot['question'])
        image_path = str(root / material['image_path']) if material['image_path'] else None
        request = CompletionRequest(task='composition-grid-answer', prompt=rendered, question=slot['question'],
                                    text=material['text'], image_path=image_path, condition=slot['cell'],
                                    metadata={'material_id': slot['material_id'], 'question_type': slot['question_type']})
        completion = provider.complete(request)
        if slot['answer_kind'] == 'items':
            correct, parsed, reason = score_items(completion.text, slot['expected'])
        else:
            correct, parsed, reason = score_answer(completion.text, slot['answer_kind'], slot['expected'])
        row = {**slot, 'condition': slot['cell'], 'raw_output': completion.text, 'parsed_answer': parsed,
               'correct': bool(correct), 'score_reason': reason, 'finish_reason': completion.finish_reason,
               'usage': completion.usage, 'completed_at': utc_now()}
        with lock:
            with answers_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        return correct

    with ThreadPoolExecutor(max_workers=max(2, max_batch * 2)) as pool:
        futures = [pool.submit(work, slot) for slot in pending]
        for future in as_completed(futures):
            future.result()
            done += 1
    return {'run_id': run_id, 'planned': len(plan), 'already_done': len(prior), 'new': done}


def _rescore(row):
    """Re-score from the raw output so a scoring fix does not require regenerating answers."""
    kind = row.get('answer_kind')
    if kind == 'items':
        correct, parsed, reason = score_items(row['raw_output'], row['expected'])
    else:
        correct, parsed, reason = score_answer(row['raw_output'], kind, row['expected'])
    return {**row, 'correct': bool(correct), 'parsed_answer': parsed, 'score_reason': reason,
            'correct_as_logged': row['correct']}


SIGNED_BUCKETS = ('<=-4', '-3', '-2', '-1', '+1', '+2', '+3', '>=+4')


def _signed_bucket(margin):
    """Signed distance of an item's value from the rule threshold (negative = does not count)."""
    if margin <= -4:
        return '<=-4'
    if margin >= 4:
        return '>=+4'
    return f'{margin:+d}'


def _item_level(answers, materials):
    """Per-item correctness and error shape for the "which items count?" question.

    Items are the unit of the conditional judgment, so this is the psychometric view: accuracy by the margin
    between an item's value and the rule threshold, split by where the values were rendered.
    """
    by_key = {(row['material_id'], row['cell']): row for row in materials}
    margins, letters, shapes = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), defaultdict(int)
    by_modality = defaultdict(lambda: [0, 0])
    signed, signed_by_cell = defaultdict(lambda: [0, 0]), defaultdict(lambda: defaultdict(lambda: [0, 0]))
    direction = defaultdict(lambda: {'missed': 0, 'extra': 0, 'correct': 0})
    for row in answers:
        if row['question_type'] != 'full' or not isinstance(row['parsed_answer'], list):
            continue
        material = by_key.get((row['material_id'], row['cell']))
        if material is None:
            continue
        answered = set(row['parsed_answer'])
        truth = set(row['expected'])
        for index, value in enumerate(material['values']):
            letter = chr(65 + index)
            margin = value - material['threshold']
            bucket = '0' if margin < 0 else str(margin) if margin <= 3 else '4+'
            correct = (letter in answered) == (letter in truth)
            signed_bucket = _signed_bucket(margin)
            for table in (margins[bucket], letters[letter], by_modality[row['cell'][1]],
                          signed[signed_bucket], signed_by_cell[row['cell']][signed_bucket]):
                table[0] += int(correct)
                table[1] += 1
            direction[signed_bucket]['missed' if letter in truth and letter not in answered else
                                     'extra' if letter in answered and letter not in truth else 'correct'] += 1
        missed, extra = truth - answered, answered - truth
        if not missed and not extra:
            shapes['correct'] += 1
        elif len(missed) == 1 and not extra:
            shapes[f'missed_one_{sorted(missed)[0]}'] += 1
        elif len(extra) == 1 and not missed:
            shapes[f'extra_one_{sorted(extra)[0]}'] += 1
        else:
            shapes['other'] += 1

    def table(store):
        return {key: {'k': value[0], 'n': value[1], 'p': value[0] / value[1]} for key, value in sorted(store.items())}

    sum_shapes = defaultdict(int)
    for row in answers:
        if row['question_type'] != 'sum':
            continue
        material = by_key.get((row['material_id'], row['cell']))
        if material is None:
            continue
        if row['correct']:
            sum_shapes['correct'] += 1
            continue
        counting = [v for v in material['values'] if v >= material['threshold']]
        parsed = row['parsed_answer'] if isinstance(row['parsed_answer'], list) else []
        if parsed and sorted(parsed) == sorted(counting):
            sum_shapes['listed_values_not_summed'] += 1
        elif parsed and len(parsed) == 1:
            sum_shapes['arithmetic_error'] += 1
        else:
            sum_shapes['abstain_or_unparsed'] += 1

    return {'by_margin': table(margins), 'by_signed_margin': table(signed),
            'signed_margin_by_cell': {cell: table(store) for cell, store in sorted(signed_by_cell.items())},
            'by_letter': table(letters), 'by_value_modality': table(by_modality),
            'signed_margin_direction': {key: dict(direction[key]) for key in SIGNED_BUCKETS if key in direction},
            'error_shape': dict(sorted(shapes.items())), 'sum_error_shape': dict(sorted(sum_shapes.items())), 'note':
            'margin = item value minus the rule threshold; negative = the item does not count, positive = it counts; '
            'the legacy by_margin key collapses every below-threshold item into bucket 0'}


def _cell_stats(answers, cell):
    subset = [row for row in answers if row['cell'] == cell]
    by_material = defaultdict(dict)
    for row in subset:
        by_material[row['material_id']][row['question_type']] = row['correct']
    full = [row['correct'] for row in subset if row['question_type'] == 'full']
    count = [row['correct'] for row in subset if row['question_type'] == 'count']
    total = [row['correct'] for row in subset if row['question_type'] == 'sum']
    rule = [row['correct'] for row in subset if row['question_type'] == 'rule']
    values = [row['correct'] for row in subset if row['question_type'] == 'values']
    both = [flags.get('rule', False) and flags.get('values', False) for flags in by_material.values()]
    conditional = [flags['full'] for flags in by_material.values()
                   if flags.get('rule', False) and flags.get('values', False) and 'full' in flags]
    return {
        'cell': cell,
        'n': len(full),
        'full': wilson(sum(full), len(full)) if full else None,
        'count': wilson(sum(count), len(count)) if count else None,
        'sum': wilson(sum(total), len(total)) if total else None,
        'rule_hop': wilson(sum(rule), len(rule)) if rule else None,
        'value_hop': wilson(sum(values), len(values)) if values else None,
        'both_hops': wilson(sum(both), len(both)) if both else None,
        'full_given_both_hops': wilson(sum(conditional), len(conditional)) if conditional else None,
    }


def analyze_grid_answers(*, run_id, materials_path=GRID_DIR + '/materials.jsonl', project_root='.',
                         output_root='outputs', bootstrap=10000, seed=20261115):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    answers = [_rescore(row) for row in read_jsonl(out / 'answers.jsonl')]
    materials = [json.loads(line) for line in (root / materials_path).read_text(encoding='utf-8').splitlines() if line]
    cells = sorted({row['cell'] for row in materials})
    stats = [_cell_stats(answers, cell) for cell in cells]
    by_cell = {entry['cell']: entry for entry in stats}

    full_by_material = defaultdict(dict)
    for row in answers:
        if row['question_type'] == 'full':
            full_by_material[row['material_id']][row['cell']] = row['correct']

    def contrast(group_a, group_b):
        material_ids, values = [], []
        for material_id in sorted(full_by_material):
            flags = full_by_material[material_id]
            a = [flags[c] for c in group_a if c in flags]
            b = [flags[c] for c in group_b if c in flags]
            if a and b:
                material_ids.append(material_id)
                values.append(sum(a) / len(a) - sum(b) / len(b))
        if len(values) < 2:
            return None
        values_array = np.asarray(values, dtype=float)
        return paired_bootstrap(lambda idx: float(np.mean(values_array[idx])), material_ids,
                                n=bootstrap, seed=seed)

    comparisons = {
        'cross_minus_same': contrast(CROSS_MODALITY, SAME_MODALITY),
        'TI_minus_TT': contrast(('TI',), ('TT',)),
        'IT_minus_TT': contrast(('IT',), ('TT',)),
        'II_minus_TT': contrast(('II',), ('TT',)),
        'IT_minus_TI': contrast(('IT',), ('TI',)),
    }
    summary = {
        'created_at': utc_now(), 'run_id': run_id, 'n_answers': len(answers),
        'n_materials': len({row['material_id'] for row in materials}), 'cells': cells,
        'cell_order_note': 'cell = rule modality + value modality (T=text, I=image)',
        'stats': stats, 'comparisons': comparisons,
        'item_level': _item_level(answers, materials),
    }
    write_json(out / 'grid_summary.json', summary)
    (out / 'grid_report.md').write_text(
        _render(run_id, stats, comparisons, answers, summary['item_level']), encoding='utf-8')
    return summary


def _pct(entry, key):
    value = entry.get(key)
    if not value:
        return '—'
    return f'{value["p"]:.3f} ({value["k"]}/{value["n"]})'


def _render(run_id, stats, comparisons, answers, item_level):
    lines = [f'# Composition grid {run_id}', '',
             'Cell notation: first letter = modality holding the rule, second = modality holding the four item '
             'values (T = text, I = image). All cells share the same values and threshold, so they are paired by '
             'material. Programmatic scoring; no judge.', '',
             '| cell | full (which items) | count | sum | rule hop | value hop | both hops | full \\| both hops |',
             '|---|---|---|---|---|---|---|---|']
    for entry in stats:
        lines.append(f'| {entry["cell"]} | {_pct(entry, "full")} | {_pct(entry, "count")} | '
                     f'{_pct(entry, "sum")} | {_pct(entry, "rule_hop")} | {_pct(entry, "value_hop")} | '
                     f'{_pct(entry, "both_hops")} | {_pct(entry, "full_given_both_hops")} |')
    lines += ['', 'Paired material bootstrap (10000 resamples of materials); difference in full-question accuracy:', '',
              '| contrast | diff | CI |', '|---|---:|---|']
    for name, result in comparisons.items():
        if not result:
            lines.append(f'| {name} | — | — |')
            continue
        lines.append(f'| {name} | {result["point"]:+.3f} | [{result["lo"]:+.3f}, {result["hi"]:+.3f}] |')
    lines += ['', 'Item level (which items count, item = one value/threshold comparison):', '',
              'Psychometric curve: per-item accuracy by signed distance from the rule threshold '
              '(negative = the item does not count):', '', '| signed margin | accuracy |', '|---|---|']
    for key in SIGNED_BUCKETS:
        entry = item_level['by_signed_margin'].get(key)
        lines.append(f'| {key} | {entry["k"]}/{entry["n"]} = {entry["p"]:.3f} |' if entry else f'| {key} | — |')
    direction = item_level.get('signed_margin_direction', {})
    if direction:
        lines += ['', '| signed margin | accuracy | missed (counts, omitted) | extra (does not count, included) |',
                  '|---|---|---|---|']
        for key in SIGNED_BUCKETS:
            entry = item_level['by_signed_margin'].get(key)
            miss = direction.get(key, {})
            lines.append(f'| {key} | ' + (f'{entry["k"]}/{entry["n"]} = {entry["p"]:.3f}' if entry else '—') +
                         f' | {miss.get("missed", 0)} | {miss.get("extra", 0)} |')
    cells = [cell for cell in CELLS if cell in item_level['signed_margin_by_cell']]
    if cells:
        lines += ['', '| signed margin | ' + ' | '.join(cells) + ' |', '|---|' + '---|' * len(cells)]
        for key in SIGNED_BUCKETS:
            parts = []
            for cell in cells:
                entry = item_level['signed_margin_by_cell'].get(cell, {}).get(key)
                parts.append(f'{entry["k"]}/{entry["n"]} = {entry["p"]:.3f}' if entry else '—')
            lines.append(f'| {key} | ' + ' | '.join(parts) + ' |')
    for letter, entry in item_level['by_letter'].items():
        lines.append(f'| item {letter} | {entry["k"]}/{entry["n"]} = {entry["p"]:.3f} |')
    shapes = ', '.join(f'{name} {count}' for name, count in item_level['error_shape'].items())
    lines += ['', f'Error shape of the full question: {shapes}.']
    sum_shapes = ', '.join(f'{name} {count}' for name, count in item_level['sum_error_shape'].items())
    lines += [f'Error shape of the sum question: {sum_shapes}.']
    failed = [row for row in answers if row['question_type'] == 'full' and not row['correct']]
    lines += ['', f'Full-question failures: {len(failed)} of '
                  f'{sum(1 for row in answers if row["question_type"] == "full")}.']
    lines.append('')
    return '\n'.join(lines)
