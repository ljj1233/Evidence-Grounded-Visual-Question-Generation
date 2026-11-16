"""E1: where does single-modality redundancy come from?

Every question already adjudicated redundant (one side alone determines the answer) is classified by the
source of that redundancy:

- ``leak``: the question restates information that exists only on the redundant side, so the redundant side
  is no longer needed. The generator copied the other side's content into the question.
- ``anchor``: the question does not restate the redundant side; it simply asks for something the retained
  side already contains (a text fact lookup when the image is redundant, a visual readout when the text is
  redundant).
- ``other``: degenerate or ambiguous questions, outside knowledge, or a judgment that looks wrong.

Two passes run on the same units. The programmatic screen uses string evidence only: for a redundant text
side, a number or 5-word span shared with the text (a *necessary* condition, so ``screen=leak`` is an upper
bound); for a redundant image side, a number in the question that does not occur in the text (no OCR is
available, so this is a *lower* bound). The optional vision judge re-adjudicates with the material and the
attached image, which stands in for OCR. Neither pass is human review: the manual sheet is written with the
human columns empty and stays pending until a person fills it in.

Unit of analysis is the unique (material, question) pair; duplicate generations of the same question are
carried along as a weight and never inflate the rates. Rates are therefore question-level, and the
per-material block reports how concentrated a source is inside a material.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .inline_analysis import contains_number, lexical_inline, number_tokens
from .llm import CachedCaller, json_object
from .records import read_jsonl, stable_id, utc_now, write_json, write_jsonl
from .stats import wilson
from .prompts import PromptSet, prompt_path

SOURCES = ('leak', 'anchor', 'other')
VERDICT_DIRECTION = {'text_suff': 'text', 'image_suff': 'image', 'both_suff': 'both'}
SYNTHETIC_DIRECTION = {'single_image': 'image', 'single_text': 'text'}
DEICTIC = ('in the image', 'in the picture', 'in the figure', 'in the diagram', 'in the chart',
           'in the graph', 'shown in', 'as shown', 'the image shows', 'the figure shows')
PROMPTS = ('redundancy-source-text-v1', 'redundancy-source-image-v1')


def redundant_sides(direction):
    """Sides that are unnecessary. `direction` names the side that alone suffices."""
    return {'image': ('text',), 'text': ('image',), 'both': ('text', 'image')}.get(direction, ())


# ----------------------------------------------------------------------------- unit loading

def _material_index(materials_path, root):
    materials = {}
    for row in read_jsonl(root / materials_path):
        materials[row['material_id']] = row
    return materials


def load_units(*, kind, labels_path, materials_path, dataset, root):
    """Read one dataset's labels into the shared unit schema; keep only decided and redundant units."""
    root = Path(root)
    rows = read_jsonl(root / labels_path)
    if not rows:
        raise ValueError(f'no rows in {labels_path}')
    materials = _material_index(materials_path, root)
    units = {}
    for row in rows:
        if kind == 'scienceqa':
            direction = row.get('direction') if row.get('redundant') == 1 else None
            if direction not in ('image', 'text', 'both'):
                direction = None
            redundant = 1 if direction else (0 if row.get('redundant') == 0 else None)
            decided = bool(row.get('decided'))
            category = None
        elif kind == 'mmqa':
            verdict = (row.get('label_consensus') or {}).get('verdict')
            direction = VERDICT_DIRECTION.get(verdict)
            redundant = 1 if direction else (0 if verdict == 'dual_necessary' else None)
            decided = verdict is not None and verdict != 'undetermined'
            category = None
        elif kind == 'synthetic':
            label = row.get('generated_label')
            direction = SYNTHETIC_DIRECTION.get(label)
            redundant = 1 if direction else (0 if str(label).startswith('dual') else None)
            decided = label is not None and label != 'other'
            category = row.get('category')
        else:
            raise ValueError(f'unknown dataset kind: {kind}')
        material = materials.get(row['material_id'])
        if material is None:
            continue
        key = (row['material_id'], row['question'])
        if key in units:
            units[key]['duplicates'] += 1
            continue
        units[key] = {
            'unit_id': stable_id('redsrc', dataset, row['material_id'], row['question']),
            'dataset': dataset, 'kind': kind, 'material_id': row['material_id'],
            'question': row['question'], 'direction': direction, 'redundant': redundant,
            'decided': decided, 'valid': row.get('valid'), 'category': category,
            'text': material['text'], 'image_path': material['image_path'], 'duplicates': 1,
        }
    return list(units.values()), len(rows)


# ----------------------------------------------------------------------------- programmatic screen

def screen_side(question, text, side, *, min_digits=1, lexical_n=5):
    """String evidence that the question copies the (redundant) side it names."""
    if side == 'text':
        numbers = contains_number(question, text, min_digits=min_digits)
        span = lexical_inline(question, text, n=lexical_n)
        flag = bool(numbers or span)
        return {'side': 'text', 'source': 'leak' if flag else 'anchor',
                'copy_numbers': numbers, 'copy_span': span, 'novel_numbers': [],
                'deictic': None}
    text_numbers = set(number_tokens(text))
    novel = [token for token in number_tokens(question) if token not in text_numbers]
    lowered = question.lower()
    deictic = next((cue for cue in DEICTIC if cue in lowered), None)
    return {'side': 'image', 'source': 'leak' if novel else 'anchor',
            'copy_numbers': [], 'copy_span': None, 'novel_numbers': novel, 'deictic': deictic}


def screen_unit(unit, *, min_digits=1, lexical_n=5):
    return {side: screen_side(unit['question'], unit['text'], side, min_digits=min_digits,
                              lexical_n=lexical_n)
            for side in redundant_sides(unit['direction'])}


# ----------------------------------------------------------------------------- vision judge

def _normalize_source(value):
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if value in SOURCES else None


def judge_prompt(unit, side, prompts, knowledge):
    template = f'redundancy-source-{side}-v1'
    return prompts.render(template, Q=unit['question'], TEXT=unit['text'], KNOWLEDGE=knowledge)


def judge_unit(unit, *, caller, model, prompts, knowledge, sides):
    outcome = {}
    for side in sides:
        prompt = judge_prompt(unit, side, prompts, knowledge)
        image_path = unit['image_path'] if side == 'image' else None
        result = caller.call(model=model, prompt=prompt, image_path=image_path, task='audit_question')
        verdict = None if result is None else json_object(result['text'])
        source = _normalize_source((verdict or {}).get('source'))
        outcome[side] = {
            'source': source,
            'copied_span': (verdict or {}).get('copied_span'),
            'copied_info': (verdict or {}).get('copied_info'),
            'needed': (verdict or {}).get('needed'),
            'why': (verdict or {}).get('why'),
            'raw': None if result is None else result['text'],
            'error': None if source else 'unparsable or failed call',
        }
    return outcome


def combine_source(direction, sides):
    """One source label per question: a leak on either redundant side wins, then anchor, then other."""
    wanted = redundant_sides(direction)
    available = [sides[side]['source'] for side in wanted if side in sides]
    if not available or any(value is None for value in available):
        return 'other'
    if 'leak' in available:
        return 'leak'
    if all(value == 'anchor' for value in available):
        return 'anchor'
    return 'other'


# ----------------------------------------------------------------------------- aggregation

def _rate(rows, predicate):
    values = [predicate(row) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return wilson(0, 0)
    return wilson(sum(1 for value in values if value), len(values))


def _source_rates(units, key):
    rows = [{'source': unit.get(key)} for unit in units]
    return {source: _rate(rows, lambda row, s=source: row['source'] == s) for source in SOURCES}


def _examples(units, key, source, detail_key, n=2):
    picked = [unit for unit in units if unit.get(key) == source]
    return [{'material_id': unit['material_id'], 'question': unit['question'],
             'detail': (unit.get(detail_key) or {}), 'text': unit['text'][:400]}
            for unit in picked[:n]]


def _per_material(redundant_units, key):
    grouped = defaultdict(list)
    for unit in redundant_units:
        grouped[unit['material_id']].append(unit[key])
    materials = len(grouped)
    with_leak = sum(1 for values in grouped.values() if 'leak' in values)
    shares = [sum(1 for v in values if v == 'leak') / len(values) for values in grouped.values()]
    mixes = Counter(tuple(sorted(Counter(values).items(), key=lambda kv: str(kv[0]))) for values in grouped.values())
    return {'materials': materials,
            'materials_with_any_leak': wilson(with_leak, materials),
            'mean_leak_share': None if not shares else round(sum(shares) / len(shares), 4),
            'dominant_mix': [{'sources': list(mix), 'count': count}
                             for mix, count in mixes.most_common(6)]}


def summarize_dataset(dataset, units, n_rows):
    redundant = [unit for unit in units if unit['redundant'] == 1]
    decided = [unit for unit in units if unit['decided']]
    mix = {
        'text_redundant': _rate(decided, lambda u: u['direction'] in ('image', 'both')),
        'image_redundant': _rate(decided, lambda u: u['direction'] in ('text', 'both')),
        'both_sides_redundant': _rate(decided, lambda u: u['direction'] == 'both'),
        'no_single_side': _rate(decided, lambda u: u['redundant'] == 0),
    }
    by_direction = {}
    for direction in ('image', 'text', 'both'):
        subset = [unit for unit in redundant if unit['direction'] == direction]
        by_direction[direction] = {
            'n': len(subset),
            'n_valid': sum(1 for unit in subset if unit['valid'] == 1),
            'screen': _source_rates(subset, 'screen_source'),
            'judge': _source_rates(subset, 'judge_source'),
            'agreement': _rate([unit for unit in subset if unit.get('judge_source')],
                               lambda u: u['screen_source'] == u['judge_source']),
        }
    valid_redundant = [unit for unit in redundant if unit['valid'] == 1]
    validity_tracked = any(unit['valid'] is not None for unit in redundant)
    return {
        'kind': units[0]['kind'] if units else None, 'n_rows': n_rows, 'n_units': len(units),
        'n_redundant': len(redundant), 'n_redundant_valid': len(valid_redundant) if validity_tracked else None,
        'redundancy_mix': mix,
        'by_direction': by_direction,
        'overall': {'screen': _source_rates(redundant, 'screen_source'),
                    'judge': _source_rates(redundant, 'judge_source'),
                    'judge_valid_only': _source_rates(valid_redundant, 'judge_source'),
                    'combined': _source_rates(redundant, 'source')},
        'per_material_judge': _per_material([u for u in redundant if u.get('judge_source')], 'judge_source'),
        'per_material_screen': _per_material(redundant, 'screen_source'),
        'examples': {source: _examples(redundant, 'judge_source', source, 'judge_detail') for source in SOURCES},
        'screen_note': ('text side: any shared number or 5-word span is necessary but not sufficient for a '
                        'leak, so screen=leak is an upper bound; image side: a question number absent from '
                        'the text is a lower bound because image-only text content is not OCR-able'),
    }


# ----------------------------------------------------------------------------- manual sheet

def manual_sheet(units, n, seed=20261115):
    strata = defaultdict(list)
    for unit in units:
        if unit['redundant'] == 1:
            strata[(unit['dataset'], unit['direction'], unit['source'])].append(unit)
    rng = random.Random(seed)
    keys = sorted(strata, key=lambda k: (k[0], str(k[1]), k[2]))
    picked = []
    while len(picked) < n and any(strata[key] for key in keys):
        for key in keys:
            if not strata[key] or len(picked) >= n:
                continue
            items = sorted(strata[key], key=lambda u: u['unit_id'])
            picked.append(items.pop(rng.randrange(len(items))))
    return picked


def write_manual_sheet(units, n, out, seed=20261115):
    picked = manual_sheet(units, n, seed=seed)
    rows = [{
        'dataset': unit['dataset'], 'material_id': unit['material_id'], 'question': unit['question'],
        'redundant_side': '+'.join(redundant_sides(unit['direction'])), 'evaluator_direction': unit['direction'],
        'evaluator_valid': unit['valid'], 'text_excerpt': unit['text'][:1200],
        'image_path': unit['image_path'], 'screen_source': unit['screen_source'],
        'screen_detail': unit.get('screen_detail'), 'judge_source': unit.get('judge_source'),
        'judge_detail': unit.get('judge_detail'), 'human_source': None, 'human_notes': None,
    } for unit in picked]
    write_jsonl(out / 'manual_sheet.jsonl', rows)
    return {'n': len(rows), 'by_source': dict(Counter(row['judge_source'] or row['screen_source'] or 'unclassified'
                                                        for row in rows)),
            'path': str((out / 'manual_sheet.jsonl').relative_to(out.parent.parent))}


# ----------------------------------------------------------------------------- report

def _fmt(value):
    if not value or value['n'] == 0:
        return 'n/a'
    return f"{value['p']:.3f} ({value['k']}/{value['n']})"


def render_report(report):
    lines = ['# E1: where does single-modality redundancy come from?', '',
             f"Created {report['created_at']}. Unit = unique (material, question) pair. `direction` names the "
             'side that alone suffices, so `direction=image` means the **text is redundant**.',
             '', '`leak` = the question restates information only the redundant side holds. `anchor` = the '
             'question asks something the retained side already contains. `other` = degenerate, outside '
             'knowledge, or a judgment that looks wrong. Judge labels are AI, not human review.', '']
    judge = report['judge']
    if judge:
        lines += [f"Judge: `{judge['model']}`, prompts {', '.join(f'`{name}`' for name in PROMPTS)}. "
                  f"{judge['summary']['api_calls']} API calls, {judge['summary']['cache_hits']} cache hits, "
                  f"{judge['summary'].get('total_tokens', 'unknown')} tokens (see calls).", '']
    for dataset, item in report['datasets'].items():
        lines += [f"## {dataset} ({item['kind']})", '',
                  f"{item['n_rows']} candidate rows, {item['n_units']} unique questions, "
                  f"{item['n_redundant']} redundant ("
                  + ('validity not tracked' if item['n_redundant_valid'] is None
                     else f"{item['n_redundant_valid']} valid") + ').', '',
                  'Redundancy mix (decided units): text redundant '
                  f"{_fmt(item['redundancy_mix']['text_redundant'])}, image redundant "
                  f"{_fmt(item['redundancy_mix']['image_redundant'])}, both sides redundant "
                  f"{_fmt(item['redundancy_mix']['both_sides_redundant'])}.", '',
                  '| direction (side that suffices) | n | leak (judge) | anchor (judge) | other (judge) | '
                  'leak (screen) | agreement |', '|---|---:|---:|---:|---:|---:|---:|']
        for direction in ('image', 'text', 'both'):
            row = item['by_direction'][direction]
            lines.append(f"| {direction} | {row['n']} | {_fmt(row['judge']['leak'])} | "
                         f"{_fmt(row['judge']['anchor'])} | {_fmt(row['judge']['other'])} | "
                         f"{_fmt(row['screen']['leak'])} | {_fmt(row['agreement'])} |")
        overall = item['overall']['judge']
        lines += ['', f"All redundant units: leak {_fmt(overall['leak'])}, anchor {_fmt(overall['anchor'])}, "
                      f"other {_fmt(overall['other'])}. Valid only: leak "
                      f"{_fmt(item['overall']['judge_valid_only']['leak'])}.", '',
                  f"Materials with at least one leak: {_fmt(item['per_material_judge']['materials_with_any_leak'])} "
                  f"of {item['per_material_judge']['materials']} materials "
                  f"(mean within-material leak share {item['per_material_judge']['mean_leak_share']}).", '']
        for source in SOURCES:
            for example in item['examples'][source][:1]:
                lines.append(f"- {source}: \"{example['question']}\" | material: {example['text'][:160]!r}")
        lines.append('')
    manual = report.get('manual')
    if manual:
        lines += ['## Manual sheet', '',
                  f"{manual['n']} rows in `{manual['path']}` with `human_source`/`human_notes` empty; "
                  f"strata {manual['by_source']}. **Pending human review.**", '']
    return '\n'.join(lines)


# ----------------------------------------------------------------------------- driver

def run_redundancy_source(*, run_id, datasets, judge_model, knowledge, project_root='.', output_root='outputs',
                          use_judge=True, concurrency=8, max_calls=0, min_digits=1, lexical_n=5,
                          manual=50, dry_run=False, seed=20261115):
    """`datasets` maps a label to {'kind', 'labels_path', 'materials_path'}."""
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    loaded, counts = {}, {}
    for label, spec in datasets.items():
        units, n_rows = load_units(kind=spec['kind'], labels_path=spec['labels_path'],
                                   materials_path=spec['materials_path'], dataset=label, root=root)
        loaded[label] = units
        counts[label] = n_rows
    for label, units in loaded.items():
        for unit in units:
            screened = screen_unit(unit, min_digits=min_digits, lexical_n=lexical_n)
            unit['screen_detail'] = screened
            unit['screen_source'] = ('leak' if any(value['source'] == 'leak' for value in screened.values())
                                     else ('anchor' if screened else None))

    caller = CachedCaller(project_root=root, run_id=run_id, output_root=output_root, dry_run=dry_run,
                          max_calls=max_calls)
    if use_judge:
        from .prompts import PromptSet
        prompts = PromptSet(root / 'prompts', names=PROMPTS)
        tasks = [(label, index, side) for label, units in loaded.items() for index, unit in enumerate(units)
                 if unit['redundant'] == 1 for side in redundant_sides(unit['direction'])]

        def work(task):
            label, index, side = task
            unit = loaded[label][index]
            return task, judge_unit(unit, caller=caller, model=judge_model, prompts=prompts,
                                    knowledge=knowledge, sides=(side,))

        results = {}
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            for task, outcome in pool.map(work, tasks):
                results[task[:2]] = results.get(task[:2], {})
                results[task[:2]].update(outcome)
        for (label, index), sides in results.items():
            unit = loaded[label][index]
            unit['judge_detail'] = sides
            unit['judge_source'] = combine_source(unit['direction'], sides)
    for units in loaded.values():
        for unit in units:
            unit['source'] = unit.get('judge_source') or unit['screen_source']
    call_summary = caller.summary()
    call_summary['total_tokens'] = _token_total(out)
    judge = {'model': judge_model, 'summary': call_summary,
             'prompts': {name: prompt_path(root / 'prompts', name).read_text() for name in PROMPTS}} if use_judge else None

    report = {'created_at': utc_now(), 'run_id': run_id, 'judge': judge,
              'datasets': {label: summarize_dataset(label, units, counts[label]) for label, units in loaded.items()}}
    units_out = []
    for label, units in loaded.items():
        for unit in units:
            units_out.append({key: unit.get(key) for key in (
                'unit_id', 'dataset', 'kind', 'material_id', 'question', 'direction', 'redundant', 'decided',
                'valid', 'category', 'duplicates', 'screen_source', 'screen_detail', 'judge_source',
                'judge_detail', 'source')})
    write_jsonl(out / 'source_units.jsonl', units_out)
    report['manual'] = write_manual_sheet([unit for units in loaded.values() for unit in units], manual, out,
                                          seed=seed)
    write_json(out / 'source_summary.json', report)
    (out / 'source_report.md').write_text(render_report(report))
    return {label: {'n_units': item['n_units'], 'n_redundant': item['n_redundant'],
                    'leak': item['overall']['combined']['leak']['p'],
                    'anchor': item['overall']['combined']['anchor']['p']}
            for label, item in report['datasets'].items()} | {'calls': call_summary}


def _token_total(out):
    total = 0
    path = out / 'calls_vqag.jsonl'
    if not path.exists():
        return total
    for row in read_jsonl(path):
        usage = row.get('usage') or {}
        total += int(usage.get('total_tokens') or 0)
    return total


def parse_dataset_specs(specs):
    """`label=kind:labels_path:materials_path` -> dict, as used by the CLI."""
    datasets = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError(f'bad dataset spec (need label=kind:labels:materials): {spec}')
        label, rest = spec.split('=', 1)
        parts = rest.split(':')
        if len(parts) != 3:
            raise ValueError(f'bad dataset spec (need label=kind:labels:materials): {spec}')
        datasets[label] = {'kind': parts[0], 'labels_path': parts[1], 'materials_path': parts[2]}
    return datasets
