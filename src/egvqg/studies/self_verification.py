"""E3: local self-verification of candidate questions, and its repair tiers.

The selection rule mirrors API PVS but uses the local answerer instead of API screens:

    select iff the answerer is CORRECT with the full material (TI) and WRONG with text alone (T) and
    with the image alone (I).

A genuine dual question should satisfy it; a question that one side alone answers fails it because that side's
condition is correct. The measured quantity on a pool with known labels is the selected-set precision/recall and
the cost in answerer calls, so it is directly comparable with the API PVS numbers.

Two repair tiers are supported. `direct` uses `composition-answer-v1`. `two-step` uses
`composition-two-step-v1`, which asks the model to state the rule, then the values, then combine (the behavioural
repair). The mechanistic patching tier is not part of this module.

Gold answers are required to score correctness. For the synthetic composition families they are computable from the
material (exact designed questions first, then pattern rules for threshold / which-items / count / sum / item value),
so no judge calls are needed and the scoring is deterministic; questions without a computable gold are dropped and
the coverage is reported rather than guessed.
"""
from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..core.prompts import PromptSet
from ..core.records import append_jsonl, canonical_json, read_jsonl, sha256_file, sha256_text, stable_id, utc_now, write_json, write_jsonl
from ..core.stats import paired_bootstrap, wilson
from .synthetic_composition import _answer_text, _extract_json, _item_values, _ints, score_answer

CONDITIONS = ('TI', 'T', 'I')
TIERS = ('direct', 'two-step', 'delegated', 'per-item', 'delegated-compare')
PROMPTS = {'direct': 'composition-answer-v1', 'two-step': 'composition-two-step-v1',
           'delegated': 'composition-delegated-v1', 'per-item': 'composition-per-item-v1',
           'delegated-compare': 'composition-delegated-compare-v1'}
DUAL_LABELS = ('dual_derived', 'dual_stitching', 'dual_bridging')
SELECT_RULE = 'TI correct and T wrong and I wrong'


def normalize_text(text):
    return re.sub(r'\s+', ' ', str(text or '').strip().lower())


def _letters(text):
    return {letter.upper() for letter in re.findall(r'\b([A-Da-d])\b', str(text or ''))}


def _combine(question, kind):
    """How the answer is built from the qualifying values; drives the delegated-aggregation tier."""
    if kind == 'item_set':
        return 'set'
    if re.search(r'\btotal\b|\bsum\b|\baltogether\b|\bcombined\b', question):
        return 'sum'
    if re.search(r'\bhow many\b|\bnumber of\b', question):
        return 'count'
    return 'identity'


def parse_gold(question, material):
    """Deterministic gold for a question over one synthetic composition material; None when not computable."""
    q = normalize_text(question)

    def entry(kind, **fields):
        return {'kind': kind, 'combine': _combine(q, kind), **fields}

    evaluation = material.get('evaluation') or {}
    for key in ('full', 'image_hop', 'text_hop'):
        item = evaluation.get(key)
        if item and normalize_text(item.get('question')) == q:
            return entry('designed', answer_kind=item['answer_kind'], expected=item['expected'], source=key)
    values = list(material.get('values') or [])
    threshold = material.get('threshold')
    explicit = re.search(r'(?:at least|minimum|greater than|more than|above|exceeds?|below)\s+(\d+)', q)
    cutoff = int(explicit.group(1)) if explicit else threshold
    if re.search(r'\btotal\b|\bsum\b', q):
        if cutoff is None:
            return None
        return entry('designed', answer_kind='numeric',
                     expected=sum(v for v in values if v >= cutoff), source='sum_rule')
    if re.search(r'\bhow many\b', q):
        if cutoff is None:
            return None
        return entry('designed', answer_kind='numeric',
                     expected=sum(1 for v in values if v >= cutoff), source='count_rule')
    if re.search(r'\b(?:which|what)\s+items?\b|\bitems?\s+that\s+count|items?\s+(?:have|meet|with|whose|satisf)', q):
        if cutoff is None:
            return None
        return entry('item_set', expected=[chr(65 + i) for i, v in enumerate(values) if v >= cutoff],
                     source='which_items_rule')
    item = re.search(r'\bitem\s+([A-Da-d])\b', q)
    if item and not re.search(r'\bhow many\b', q):
        index = ord(item.group(1).upper()) - 65
        if index < len(values):
            return entry('designed', answer_kind='numeric', expected=values[index], source='item_value_rule')
        return None
    if 'alpha' in q:
        full = evaluation.get('full')
        if full and isinstance(full.get('expected'), int):
            return entry('designed', answer_kind=full['answer_kind'], expected=full['expected'], source='alpha_rule')
        return None
    if re.search(r'threshold|minimum value|the rule\b', q):
        if threshold is None:
            return None
        return entry('designed', answer_kind='numeric', expected=threshold, source='threshold_rule')
    return None


def score_gold(raw, gold):
    """Score a local answer against a gold entry; abstain and unparsable answers are incorrect."""
    if gold['kind'] == 'item_set':
        answer = _answer_text(raw)
        if answer is None:
            return False, None, 'abstain_or_missing'
        letters = _letters(answer)
        ok = letters == set(gold['expected'])
        return ok, sorted(letters), '' if ok else 'item_set_mismatch'
    return score_answer(raw, gold['answer_kind'], gold['expected'])


def _values_list(raw):
    """Read the `values` field the delegated tier is asked to emit."""
    obj = _extract_json(raw)
    if not obj:
        return []
    values = obj.get('values')
    if isinstance(values, list):
        out = []
        for item in values:
            out.extend(_ints(str(item)))
        return out
    if values is None:
        return []
    return _ints(str(values))


def _per_item_table(raw):
    """Read the per-item table: list of (item, value, counts)."""
    obj = _extract_json(raw)
    if not obj or not isinstance(obj.get('items'), list):
        return []
    table = []
    for entry in obj['items']:
        if not isinstance(entry, dict):
            continue
        letter = re.search(r'\b([A-Da-d])\b', str(entry.get('item') or ''))
        numbers = _ints(str(entry.get('value')))
        counts = entry.get('counts')
        if isinstance(counts, str):
            counts = counts.strip().lower() in ('true', 'yes', '1')
        if letter and numbers and isinstance(counts, bool):
            table.append((letter.group(1).upper(), numbers[0], counts))
    return table


def score_gold_per_item(raw, gold):
    """Score from a per-item table: the harness aggregates the items the model marked as counting.

    The comparison against the threshold is made explicit per item (the E2b boundary defect), so this tier tests
    whether stating `value >= N` item by item repairs the comparator slip that survives the list and two-step tiers.
    """
    table = _per_item_table(raw)
    kept = [value for _letter, value, counts in table if counts]
    letters = [letter for letter, _value, counts in table if counts]
    if table and gold.get('combine') in ('sum', 'count', 'set'):
        if gold['combine'] == 'sum':
            recomputed = sum(kept)
            ok = recomputed == gold['expected']
        elif gold['combine'] == 'count':
            recomputed = len(kept)
            ok = recomputed == gold['expected']
        else:
            recomputed = sorted(letters)
            ok = set(letters) == set(gold['expected'])
        agreed = _answer_text(raw)
        consistent = None
        if agreed is not None:
            consistent = _ints(agreed) == [recomputed] if gold['combine'] != 'set' else _letters(agreed) == set(letters)
        return ok, recomputed, '' if ok else f'per_item_{gold["combine"]}_mismatch', {'values': kept, 'consistent': consistent}
    correct, parsed, reason = score_gold(raw, gold)
    return correct, parsed, reason, {'values': None, 'consistent': None}


def score_gold_delegated(raw, gold):
    """Score with harness-side aggregation: the answerer lists the values, the harness sums or counts them.

    Only sum/count answers are recomputed; item-set and single-value answers keep the model's own output, so the
    delegation cannot repair the filtering hop or the value reading, only the arithmetic that E2a identified.
    """
    if gold.get('combine') in ('sum', 'count'):
        values = _values_list(raw)
        if values:
            recomputed = sum(values) if gold['combine'] == 'sum' else len(values)
            ok = recomputed == gold['expected']
            return (ok, recomputed, '' if ok else f'delegated_{gold["combine"]}_mismatch',
                    {'values': values, 'consistent': None})
    correct, parsed, reason = score_gold(raw, gold)
    return correct, parsed, reason, {'values': None, 'consistent': None}


def _read_threshold(obj):
    """Threshold and comparator the model reported; the harness does the comparison itself."""
    text = str(obj.get('rule') or '')
    match = re.search(r'at least\s+(-?\d+)', text, re.I)
    if match:
        return int(match.group(1)), '>='
    match = re.search(r'(?:more than|greater than|above|exceeds?)\s+(-?\d+)', text, re.I)
    if match:
        return int(match.group(1)), '>'
    numbers = _ints(str(obj.get('threshold')))
    return (numbers[0], '>=') if numbers else (None, None)


def score_gold_compare(raw, gold):
    """Harness-side comparison: the model reports the rule and the item values, the harness decides and aggregates.

    This targets the comparator itself (E2b: the threshold comparison is off by one unit), so it can only be applied
    when the answer is an aggregation over per-item values.
    """
    obj = _extract_json(raw)
    if obj and gold.get('combine') in ('sum', 'count', 'set'):
        threshold, operator = _read_threshold(obj)
        table = []
        for entry in obj.get('items') or []:
            if not isinstance(entry, dict):
                continue
            letter = re.search(r'\b([A-Da-d])\b', str(entry.get('item') or ''))
            numbers = _ints(str(entry.get('value')))
            if letter and numbers:
                table.append((letter.group(1).upper(), numbers[0]))
        if threshold is not None and table:
            kept = [(letter, value) for letter, value in table
                    if (value >= threshold if operator == '>=' else value > threshold)]
            if gold['combine'] == 'sum':
                recomputed = sum(value for _letter, value in kept)
                ok = recomputed == gold['expected']
            elif gold['combine'] == 'count':
                recomputed = len(kept)
                ok = recomputed == gold['expected']
            else:
                recomputed = sorted(letter for letter, _value in kept)
                ok = set(recomputed) == set(gold['expected'])
            return (ok, recomputed, '' if ok else f'compare_{gold["combine"]}_mismatch',
                    {'values': [value for _letter, value in kept], 'consistent': None,
                     'threshold': threshold, 'operator': operator})
    correct, parsed, reason = score_gold(raw, gold)
    return correct, parsed, reason, {'values': None, 'consistent': None, 'threshold': None, 'operator': None}


def condition_material(material, condition, root):
    """Material visible under one condition; the removed side is not passed to the answerer at all."""
    text = material.get('text') or ''
    image = str(root / material['image_path']) if material.get('image_path') else None
    if condition == 'TI':
        return text, image
    if condition == 'T':
        return text, None
    if condition == 'I':
        return '', image
    raise ValueError('unknown condition')


def _plan(pool, materials, run_id, tiers, conditions):
    by_id = {m['material_id']: m for m in materials}
    plan, dropped = [], []
    for row in pool:
        material = by_id.get(row['material_id'])
        if material is None:
            dropped.append({'material_id': row['material_id'], 'question': row['question'], 'reason': 'missing_material'})
            continue
        gold = parse_gold(row['question'], material)
        if gold is None:
            dropped.append({'material_id': row['material_id'], 'question': row['question'], 'reason': 'no_gold'})
            continue
        for tier in tiers:
            for condition in conditions:
                plan.append({
                    'attempt_id': stable_id('self-verify', run_id, row['material_id'], row['question'], tier, condition),
                    'run_id': run_id, 'material_id': row['material_id'], 'question': row['question'],
                    'category': row.get('category'), 'generated_label': row.get('generated_label'),
                    'sample_index': row.get('sample_index'),
                    'tier': tier, 'condition': condition, 'gold': gold,
                })
    return plan, dropped


def run_self_verification(*, run_id, pool_path, materials_path, local_model_path, device='cuda:0',
                          tiers=TIERS, conditions=CONDITIONS, max_batch=8, max_new_tokens=256, limit=None,
                          early_stop=False, project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    from ..core.local_vision import BatchingProvider, QwenVLBackend, qwen_identity
    from ..core.providers import CompletionRequest

    pool = read_jsonl(root / pool_path)
    materials = read_jsonl(root / materials_path)
    tiers = tuple(tiers)
    conditions = tuple(conditions)
    prompts = PromptSet(root / 'prompts', names=tuple(PROMPTS[tier] for tier in tiers))
    plan, dropped = _plan(pool, materials, run_id, tiers, conditions)
    full_plan = plan
    truncated = bool(limit and limit < len(plan))
    if truncated:
        plan = plan[:limit]
    by_id = {m['material_id']: m for m in materials}
    prior_path = out / 'answers.jsonl'
    prior = {row['attempt_id'] for row in read_jsonl(prior_path)} if prior_path.exists() else set()
    backend = QwenVLBackend(local_model_path, device=device, max_new_tokens=max_new_tokens)
    identity = qwen_identity(backend, 'self-verification-v1')
    manifest_path = out / 'manifest.json'
    fingerprint = sha256_text(canonical_json({
        'config': {'tiers': tiers, 'conditions': conditions, 'max_new_tokens': max_new_tokens,
                   'early_stop': early_stop},
        'plan': full_plan, 'pool': sha256_file(root / pool_path), 'materials': sha256_file(root / materials_path),
        'prompts': prompts.snapshot(), 'identity': identity}))
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        if stored.get('fingerprint') != fingerprint:
            raise ValueError('self-verification run configuration changed')
    else:
        write_json(manifest_path, {'created_at': utc_now(), 'fingerprint': fingerprint,
                                   'config': {'tiers': tiers, 'conditions': conditions, 'device': device,
                                              'max_new_tokens': max_new_tokens, 'early_stop': early_stop},
                                   'pool_path': pool_path, 'materials_path': materials_path,
                                   'prompt_snapshot': prompts.snapshot(), 'model_identity': identity})
    write_jsonl(out / 'plan.jsonl', plan)
    write_jsonl(out / 'dropped.jsonl', dropped)
    provided = len(prior)
    provider = BatchingProvider(backend, model=backend.path.name, identity=identity, max_batch=max_batch)
    lock = threading.Lock()

    def work(slot):
        material = by_id[slot['material_id']]
        text, image = condition_material(material, slot['condition'], root)
        rendered = prompts.render(PROMPTS[slot['tier']], T=text, Q=slot['question'])
        request = CompletionRequest(task=f'self-verify-{slot["tier"]}-{slot["condition"]}', prompt=rendered,
                                    question=slot['question'], text=text, image_path=image,
                                    condition=slot['condition'],
                                    metadata={'material_id': slot['material_id'], 'tier': slot['tier']})
        completion = provider.complete(request)
        if slot['tier'] == 'delegated':
            correct, parsed, reason, detail = score_gold_delegated(completion.text, slot['gold'])
        elif slot['tier'] == 'per-item':
            correct, parsed, reason, detail = score_gold_per_item(completion.text, slot['gold'])
        elif slot['tier'] == 'delegated-compare':
            correct, parsed, reason, detail = score_gold_compare(completion.text, slot['gold'])
        else:
            correct, parsed, reason = score_gold(completion.text, slot['gold'])
            detail = {'values': None, 'consistent': None}
        record = {**slot, 'raw_output': completion.text, 'parsed_answer': parsed,
                  'correct': correct, 'score_reason': reason,
                  'delegated_values': detail.get('values'), 'consistent': detail.get('consistent'),
                  'reported_threshold': detail.get('threshold'), 'reported_operator': detail.get('operator'),
                  'finish_reason': completion.finish_reason, 'usage': completion.usage,
                  'completed_at': utc_now()}
        with lock:
            append_jsonl(prior_path, record)

    def _run(slots):
        slots = [slot for slot in slots if slot['attempt_id'] not in prior]
        if not slots:
            return 0
        with ThreadPoolExecutor(max_workers=max(2, max_batch * 2)) as pool_executor:
            futures = [pool_executor.submit(work, slot) for slot in slots]
            for future in as_completed(futures):
                future.result()
        prior.update(slot['attempt_id'] for slot in slots)
        return len(slots)

    new = 0
    if early_stop:
        new += _run([slot for slot in plan if slot['condition'] == 'TI'])
        answered = {row['attempt_id']: row for row in read_jsonl(prior_path)}
        later, skipped = [], []
        for slot in plan:
            if slot['condition'] == 'TI' or slot['attempt_id'] in answered:
                continue
            ti_id = stable_id('self-verify', run_id, slot['material_id'], slot['question'], slot['tier'], 'TI')
            ti_row = answered.get(ti_id)
            if ti_row is None:
                continue
            (later if ti_row.get('correct') else skipped).append(slot)
        new += _run(later)
        for slot in skipped:
            if slot['attempt_id'] in prior:
                continue
            append_jsonl(prior_path, {**slot, 'raw_output': None, 'parsed_answer': None, 'correct': None,
                                      'score_reason': 'skipped: full-material answer wrong',
                                      'skipped': 'ti_wrong', 'completed_at': utc_now()})
            prior.add(slot['attempt_id'])
    else:
        new = _run(plan)
    return {'run_id': run_id, 'planned': len(plan), 'already_done': provided, 'new': new,
            'dropped_no_gold': len(dropped), 'truncated': truncated, 'early_stop': early_stop,
            'skipped_single_sides': sum(1 for row in read_jsonl(prior_path) if row.get('skipped'))}


def _selected(rows):
    """Apply the selection rule to one (material, question) group of condition rows."""
    by_condition = {row['condition']: row for row in rows}
    if set(CONDITIONS) - set(by_condition):
        return None
    if not by_condition['TI']['correct'] and any(by_condition[c].get('skipped') for c in ('T', 'I')):
        return False  # early stop: the full-material answer already failed, so the rule cannot select
    if any(by_condition[c].get('correct') is None for c in CONDITIONS):
        return None  # a condition was never answered and was not skipped as unnecessary
    return bool(by_condition['TI']['correct'] and not by_condition['T']['correct']
                and not by_condition['I']['correct'])


def _sampling_curve(selections):
    """Coverage/precision of the local signal as more generated samples per material become available."""
    if not selections or any(item.get('sample_index') is None for item in selections):
        return None
    top = max(item['sample_index'] for item in selections)
    curve = []
    for k in range(1, top + 2):
        window = [item for item in selections if item['sample_index'] < k]
        materials = {item['material_id'] for item in window}
        selected = [item for item in window if item['selected']]
        true_positive = [item for item in selected if item['dual']]
        curve.append({
            'k': k, 'materials': len(materials), 'questions': len(window),
            'materials_with_dual': len({item['material_id'] for item in window if item['dual']}),
            'materials_with_selected_dual': len({item['material_id'] for item in true_positive}),
            'selected': len(selected), 'selected_dual': len(true_positive),
            'precision': round(len(true_positive) / len(selected), 3) if selected else None,
            'dual_rate': round(sum(1 for item in window if item['dual']) / len(window), 3) if window else None,
        })
    return curve


def _rate(k, n):
    return wilson(k, n) if n else {'k': 0, 'n': 0, 'p': None, 'lo': None, 'hi': None}


def _selection_metrics(selections):
    """Precision/recall of a chosen subset over the labelled question universe."""
    selected = [s for s in selections if s['selected']]
    duals = [s for s in selections if s['dual']]
    non_duals = [s for s in selections if not s['dual']]
    return {
        'questions': len(selections), 'duals': len(duals), 'selected': len(selected),
        'recall': _rate(sum(1 for s in duals if s['selected']), len(duals)),
        'precision': _rate(sum(1 for s in selected if s['dual']), len(selected)),
        'false_positive_rate': _rate(sum(1 for s in non_duals if s['selected']), len(non_duals)),
        'by_label': {label: {'k': sum(1 for s in selections if s['label'] == label and s['selected']),
                             'n': sum(1 for s in selections if s['label'] == label)}
                     for label in sorted({s['label'] for s in selections})},
        'recall_by_label': {label: _rate(sum(1 for s in duals if s['label'] == label and s['selected']),
                                         sum(1 for s in duals if s['label'] == label))
                            for label in sorted({s['label'] for s in duals})},
    }


def _external_selections(path, universe):
    """Map an API selection file onto the local question universe; unmatched questions are reported."""
    chosen = {}
    for row in read_jsonl(path):
        chosen[(row['material_id'], row['question'])] = row.get('sol_final_label') == 'dual_candidate'
    selections, unmatched = [], 0
    for item in universe:
        key = (item['material_id'], item['question'])
        if key not in chosen:
            unmatched += 1
            continue
        selections.append({**{k: item[k] for k in ('material_id', 'question', 'label', 'dual')},
                           'selected': chosen[key]})
    return selections, unmatched


def _ti_error_taxonomy(rows, materials_by_id):
    """Attribute wrong full-material answers to the item whose inclusion/exclusion explains the numeric delta.

    The grid places the threshold strictly between two values, so a tie at the threshold cannot occur; the interesting
    mistakes are an included below-threshold value (the comparator applied too loosely) or a missed qualifying value.
    """
    counts = defaultdict(int)
    for row in rows:
        if row['correct']:
            continue
        material = materials_by_id.get(row['material_id']) or {}
        values = list(material.get('values') or [])
        threshold = material.get('threshold')
        expected, answer = row['gold'].get('expected'), row['parsed_answer']
        if not isinstance(answer, int) or not isinstance(expected, int) or threshold is None or not values:
            counts['unparsed_or_item_set'] += 1
            continue
        below = [v for v in values if v < threshold]
        above = [v for v in values if v >= threshold]
        delta = answer - expected
        explained = None
        for size in range(1, len(values) + 1):
            for subset in combinations(below if delta > 0 else above, size):
                if sum(subset) == abs(delta):
                    explained = subset
                    break
            if explained:
                break
        if explained is None:
            counts['other'] += 1
        elif delta > 0 and list(explained) == [threshold - 1]:
            counts['included_threshold_minus_one'] += 1
        elif delta > 0:
            counts['included_other_below'] += 1
        else:
            counts['dropped_qualifying'] += 1
    return dict(sorted(counts.items()))


def analyze_self_verification(*, run_id, bootstrap=10000, project_root='.', output_root='outputs',
                              pvs_selection=None):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    rows = read_jsonl(out / 'answers.jsonl')
    dropped = read_jsonl(out / 'dropped.jsonl') if (out / 'dropped.jsonl').exists() else []
    manifest = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {}
    materials_by_id = {}
    if manifest.get('materials_path'):
        materials_by_id = {row['material_id']: row for row in read_jsonl(root / manifest['materials_path'])}
    tiers = sorted({row['tier'] for row in rows}) or list(TIERS)
    by_tier = defaultdict(list)
    for row in rows:
        by_tier[row['tier']].append(row)
    summary = {'run_id': run_id, 'created_at': utc_now(), 'rule': SELECT_RULE, 'tiers': {},
               'dropped_no_gold': len(dropped)}
    for tier in tiers:
        groups = defaultdict(list)
        for row in by_tier[tier]:
            groups[(row['material_id'], row['question'])].append(row)
        conditions = {condition: _rate(sum(r['correct'] for r in by_tier[tier]
                                            if r['condition'] == condition and not r.get('skipped')),
                                       sum(1 for r in by_tier[tier]
                                           if r['condition'] == condition and not r.get('skipped')))
                      for condition in CONDITIONS}
        selections, reasons = [], defaultdict(int)
        for (material_id, question), group in groups.items():
            label = group[0]['generated_label']
            dual = label in DUAL_LABELS
            chosen = _selected(group)
            if chosen is None:
                outcome = 'incomplete'
            else:
                by_condition = {row['condition']: row for row in group}
                if not by_condition['TI']['correct']:
                    outcome = 'ti_wrong'
                elif by_condition['T']['correct'] or by_condition['I']['correct']:
                    outcome = 'single_side_answers'
                else:
                    outcome = 'selected'
            reasons[outcome] += 1
            selections.append({'material_id': material_id, 'question': question, 'label': label, 'dual': dual,
                               'selected': bool(chosen), 'outcome': outcome,
                               'sample_index': group[0].get('sample_index')})
        metrics = {**_selection_metrics(selections), 'reasons': dict(sorted(reasons.items())),
                   'outcomes': {group: {outcome: sum(1 for s in selections if s['dual'] is is_dual
                                                     and s['outcome'] == outcome)
                                        for outcome in ('selected', 'ti_wrong', 'single_side_answers', 'incomplete')}
                                for group, is_dual in (('duals', True), ('non_duals', False))}}
        if pvs_selection:
            external, unmatched = _external_selections(Path(root / pvs_selection), selections)
            metrics['api_pvs'] = {**_selection_metrics(external), 'unmatched': unmatched}
        material_buckets = defaultdict(list)
        for item in selections:
            material_buckets[item['material_id']].append(item)
        coverage = {'materials': len(material_buckets),
                    'with_selected': sum(1 for items in material_buckets.values()
                                         if any(i['selected'] for i in items)),
                    'with_dual_selected': sum(1 for items in material_buckets.values()
                                              if any(i['selected'] and i['dual'] for i in items)),
                    'mean_selected': round(sum(sum(1 for i in items if i['selected'])
                                               for items in material_buckets.values())
                                           / max(1, len(material_buckets)), 3),
                    'candidates_per_material': round(len(selections) / max(1, len(material_buckets)), 2)}
        per_material = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for item in selections:
            bucket = per_material[item['material_id']][(item['label'] in DUAL_LABELS, item['selected'])]
            bucket[0] += 1
            bucket[1] += 1
        summary['tiers'][tier] = {'conditions': conditions, **metrics, 'coverage': coverage,
                                  'sampling_curve': _sampling_curve(selections),
                                  'skipped': sum(1 for r in by_tier[tier] if r.get('skipped')),
                                  'ti_error_taxonomy': _ti_error_taxonomy(
                                      [r for r in by_tier[tier] if r['condition'] == 'TI'], materials_by_id),
                                  'per_material_recall': {mid: _rate(v[(True, True)][0], v[(True, True)][1])
                                                          for mid, v in per_material.items() if v[(True, True)][1]}}
    if len(tiers) > 1:
        comparisons = {}
        per_tier_selections = {}
        for tier in tiers:
            groups = defaultdict(list)
            for row in by_tier[tier]:
                groups[(row['material_id'], row['question'])].append(row)
            selections = {}
            for key, group in groups.items():
                chosen = _selected(group)
                if chosen is not None and group[0]['generated_label'] in DUAL_LABELS:
                    selections[key] = chosen
            per_tier_selections[tier] = selections
        for lower, higher in zip(tiers, tiers[1:]):
            keys = sorted(set(per_tier_selections[lower]) & set(per_tier_selections[higher]))
            values = [(per_tier_selections[higher][k] - per_tier_selections[lower][k]) for k in keys]
            material_ids = [k[0] for k in keys]
            if values:
                result = paired_bootstrap(lambda idx: sum(values[i] for i in idx) / len(idx), material_ids,
                                          n=bootstrap, seed=0)
                comparisons[f'{higher}_minus_{lower}_recall'] = result
        summary['comparisons'] = comparisons
    write_json(out / 'self_verify_summary.json', summary)
    (out / 'self_verify_report.md').write_text(_render(run_id, summary), encoding='utf-8')
    return summary


def _fmt(rate):
    if isinstance(rate, (int, float)):
        return f'{rate:.3f}'
    if not rate or rate.get('p') is None:
        return '—'
    return f'{rate["p"]:.3f} ({rate["k"]}/{rate["n"]})'


def _render(run_id, summary):
    lines = [f'# Local self-verification {run_id}', '',
             f'Selection rule: **{summary["rule"]}**; local answerer only, no API screens. '
             f'Questions without a computable gold were dropped ({summary["dropped_no_gold"]}).', '']
    for tier, entry in summary['tiers'].items():
        lines += [f'## {tier}', '',
                  '| condition | accuracy |', '|---|---|']
        for condition in CONDITIONS:
            lines.append(f'| {condition} | {_fmt(entry["conditions"][condition])} |')
        lines += ['', f'Questions `{entry["questions"]}`, genuine duals `{entry["duals"]}`, '
                      f'selected `{entry["selected"]}`.', '',
                  f'- recall {_fmt(entry["recall"])}, precision {_fmt(entry["precision"])}, '
                  f'false-positive rate on non-duals {_fmt(entry["false_positive_rate"])}',
                  '- wiring: ' + ', '.join(f'{name} {count}' for name, count in entry['reasons'].items()),
                  f'- coverage: {entry["coverage"]["with_dual_selected"]}/{entry["coverage"]["materials"]} '
                  f'materials get at least one selected dual (mean selected '
                  f'{entry["coverage"]["mean_selected"]} of {entry["coverage"]["candidates_per_material"]} '
                  'candidates)',
                  '- duals lost to `ti_wrong` (composition gap) ' + str(entry['outcomes']['duals']['ti_wrong'])
                  + ', to `single_side_answers` ' + str(entry['outcomes']['duals']['single_side_answers']), '',
                  '- full-material error taxonomy: ' + ', '.join(
                      f'{name} {count}' for name, count in entry.get('ti_error_taxonomy', {}).items()), '',
                  (f'- early stop: {entry["skipped"]} single-side answers were skipped after a wrong full-material '
                   'answer, so T/I accuracy below is measured on the full-material-correct subset'
                   if entry.get('skipped') else ''),
                  '| label | selected / n | recall |', '|---|---:|---|']
        for label, bucket in entry['by_label'].items():
            recall = entry['recall_by_label'].get(label)
            lines.append(f'| {label} | {bucket["k"]}/{bucket["n"]} | {_fmt(recall)} |')
        lines.append('')
        curve = entry.get('sampling_curve')
        if curve:
            lines += ['| samples per material | questions | materials with a dual | materials with a selected dual '
                      '| selected | precision | raw dual rate |', '|---:|---:|---:|---:|---:|---|---|']
            for point in curve:
                lines.append(f'| {point["k"]} | {point["questions"]} | {point["materials_with_dual"]} '
                             f'| {point["materials_with_selected_dual"]} | {point["selected"]} '
                             f'| {_fmt(point["precision"]) if point["precision"] is not None else "—"} '
                             f'| {_fmt(point["dual_rate"])} |')
            lines.append('')
    if any('api_pvs' in entry for entry in summary['tiers'].values()):
        lines += ['## API PVS on the same questions', '',
                  '| condition | recall | precision | false-positive rate | selected |', '|---|---|---|---|---:|']
        for tier, entry in summary['tiers'].items():
            pvs = entry.get('api_pvs')
            if pvs:
                lines.append(f'| {tier} | {_fmt(pvs["recall"])} | {_fmt(pvs["precision"])} | '
                             f'{_fmt(pvs["false_positive_rate"])} | {pvs["selected"]} |')
        lines.append('')
    if summary.get('comparisons'):
        lines += ['## Tier comparison', '', '| contrast | diff | CI |', '|---|---:|---|']
        for name, result in summary['comparisons'].items():
            if result:
                lines.append(f'| {name} | {result["point"]:+.3f} | [{result["lo"]:+.3f}, {result["hi"]:+.3f}] |')
        lines.append('')
    return '\n'.join(lines)


# --- Real-data tier: local verifier scored against a strong-model reference -------------------------------

SOLVE_PROMPT = 'solve-v1'


def prepare_real_pool(*, candidates_path, materials_path, sample_per_material=2, project_root='.'):
    """Seeded-per-material subsample of an already labelled candidate file, for the local verifier on real data."""
    root = Path(project_root).resolve()
    candidates = read_jsonl(root / candidates_path)
    materials = {row['material_id']: row for row in read_jsonl(root / materials_path)}
    by_material = defaultdict(list)
    for row in candidates:
        by_material[row['material_id']].append(row)
    pool, seen = [], set()
    for material_id, rows in by_material.items():
        if material_id not in materials:
            continue
        if sample_per_material and sample_per_material > 0:
            count = min(sample_per_material, len(rows))
            picks = sorted({round(i * (len(rows) - 1) / (count - 1)) for i in range(count)}) if count > 1 else [0]
        else:
            picks = list(range(len(rows)))
        for index in picks:
            row = rows[index]
            if (material_id, row['question']) in seen:
                continue          # duplicate generations of the same question would double-count
            seen.add((material_id, row['question']))
            verdict = (row.get('label_consensus') or {}).get('verdict')
            pool.append({'material_id': material_id, 'question': row['question'], 'label': verdict,
                         'dual': verdict == 'dual_necessary', 'position': index})
    return pool, materials


def _solve_prompt(prompts, material, question, condition):
    visible = f"<T>{material.get('text') or ''}</T>" if condition in ('TI', 'T') else ''
    return prompts.render(SOLVE_PROMPT, VISIBLE_MATERIALS=visible, Q=question)


def run_real_verification(*, run_id, candidates_path, materials_path, local_model_path, reference_model='gpt-6-luna',
                          device='cuda:0', sample_per_material=2, max_batch=8, max_new_tokens=256, limit=None,
                          project_root='.', output_root='outputs'):
    """Reference answers from a strong API model, then three local conditions over the same questions."""
    from ..core.llm import json_object
    from ..core.providers import CompletionRequest, JudgeProvider
    from ..core.local_vision import BatchingProvider, QwenVLBackend, qwen_identity

    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    pool, materials = prepare_real_pool(candidates_path=candidates_path, materials_path=materials_path,
                                        sample_per_material=sample_per_material, project_root=project_root)
    write_jsonl(out / 'pool.jsonl', pool)
    prompts = PromptSet(root / 'prompts', names=(SOLVE_PROMPT,))

    reference_path = out / 'references.jsonl'
    done = {(row['material_id'], row['question']) for row in read_jsonl(reference_path)} \
        if reference_path.exists() else set()
    provider = JudgeProvider.from_env(root=root, run_dir=out, max_requests=5000, max_tokens=512, visual=True,
                                      model=reference_model)
    provider.max_attempts = 1
    new_references = 0
    for row in pool:
        key = (row['material_id'], row['question'])
        if key in done:
            continue
        material = materials[row['material_id']]
        image = str(root / material['image_path']) if material.get('image_path') else None
        try:
            completion = provider.complete(CompletionRequest(
                task='assess_quality', prompt=_solve_prompt(prompts, material, row['question'], 'TI'),
                question=row['question'], text=material.get('text'), image_path=image, condition='TI'))
        except Exception as exc:  # a blocked or rejected image must stay visible, not abort the run
            append_jsonl(reference_path, {**row, 'reference_answer': None, 'reference_abstain': None,
                                          'reference_error': f'{type(exc).__name__}: {exc}', 'model': reference_model,
                                          'created_at': utc_now()})
            continue
        parsed = json_object(completion.text) or {}
        answer = None if parsed.get('abstain') is True else (parsed.get('answer') or None)
        append_jsonl(reference_path, {**row, 'reference_answer': answer,
                                      'reference_abstain': parsed.get('abstain'), 'model': reference_model,
                                      'raw_output': completion.text, 'created_at': utc_now()})
        new_references += 1

    plan = []
    for row in pool:
        for condition in CONDITIONS:
            plan.append({**row, 'condition': condition,
                         'attempt_id': stable_id('self-verify-real', run_id, row['material_id'], row['question'],
                                                 condition)})
    if limit:
        plan = plan[:limit]
    answers_path = out / 'answers.jsonl'
    prior = {row['attempt_id'] for row in read_jsonl(answers_path)} if answers_path.exists() else set()
    pending = [slot for slot in plan if slot['attempt_id'] not in prior]
    new_local = 0
    if pending:
        backend = QwenVLBackend(local_model_path, device=device, max_new_tokens=max_new_tokens)
        identity = qwen_identity(backend, 'self-verification-real-v1')
        batch_provider = BatchingProvider(backend, model=backend.path.name, identity=identity, max_batch=max_batch)
        lock = threading.Lock()

        def work(slot):
            material = materials[slot['material_id']]
            image = str(root / material['image_path']) if material.get('image_path') else None
            if slot['condition'] not in ('TI', 'I'):
                image = None
            completion = batch_provider.complete(CompletionRequest(
                task=f'self-verify-real-{slot["condition"]}',
                prompt=_solve_prompt(prompts, material, slot['question'], slot['condition']),
                question=slot['question'], text=material.get('text') if slot['condition'] in ('TI', 'T') else '',
                image_path=image, condition=slot['condition']))
            with lock:
                append_jsonl(answers_path, {**slot, 'raw_output': completion.text,
                                            'finish_reason': completion.finish_reason, 'usage': completion.usage,
                                            'completed_at': utc_now()})

        with ThreadPoolExecutor(max_workers=max(2, max_batch * 2)) as executor:
            futures = [executor.submit(work, slot) for slot in pending]
            for future in as_completed(futures):
                future.result()
        new_local = len(pending)
    return {'run_id': run_id, 'pool': len(pool), 'new_references': new_references,
            'planned_local': len(plan), 'new_local': new_local, 'already_done': len(prior)}


def report_real_verification(*, run_id, equivalence_model='gpt-5.6-terra', project_root='.', output_root='outputs',
                             pvs_selection=None, limit=None):
    """Score the three local conditions against the reference with an equivalence judge, then apply the rule."""
    from ..core.llm import json_object
    from ..core.providers import CompletionRequest, JudgeProvider

    root = Path(project_root).resolve()
    out = root / output_root / run_id
    references = {(row['material_id'], row['question']): row
                  for row in read_jsonl(out / 'references.jsonl') if row.get('reference_answer')}
    rows = read_jsonl(out / 'answers.jsonl')
    if limit:
        seen, kept = set(), []
        for row in rows:
            if row['attempt_id'] in seen:
                continue
            seen.add(row['attempt_id'])
            kept.append(row)
        rows = kept[:limit]
    prompts = PromptSet(root / 'prompts', names=('answer-equivalence-v1',))
    verdict_path = out / 'equivalence.jsonl'
    judged = {row['attempt_id']: row for row in read_jsonl(verdict_path)} if verdict_path.exists() else {}
    provider = JudgeProvider.from_env(root=root, run_dir=out, max_requests=5000, max_tokens=512, visual=False,
                                      model=equivalence_model)
    provider.max_attempts = 1
    new_judgements = 0
    for row in rows:
        if row['attempt_id'] in judged:
            continue
        key = (row['material_id'], row['question'])
        reference = references.get(key)
        if reference is None:
            verdict = {'acceptable': None, 'relation': 'no_reference', 'why': 'reference missing or abstained'}
        else:
            parsed = json_object(row.get('raw_output') or '') or {}
            candidate = None if parsed.get('abstain') is True else (parsed.get('answer') or None)
            if candidate is None:
                verdict = {'acceptable': False, 'relation': 'refusal', 'why': 'local answer abstained'}
            else:
                prompt = prompts.render('answer-equivalence-v1', Q=row['question'],
                                        REF=reference['reference_answer'], CAND=candidate)
                completion = provider.complete(CompletionRequest(task='score_response', prompt=prompt,
                                                                 question=row['question']))
                judged_parsed = json_object(completion.text) or {}
                verdict = {'acceptable': judged_parsed.get('acceptable') if isinstance(judged_parsed.get('acceptable'), bool) else None,
                           'relation': judged_parsed.get('relation'), 'why': judged_parsed.get('why'),
                           'candidate': candidate}
                new_judgements += 1
        record = {**row, 'attempt_id': row['attempt_id'], **verdict}
        append_jsonl(verdict_path, record)
        judged[row['attempt_id']] = record

    groups = defaultdict(dict)
    for attempt_id, row in judged.items():
        groups[(row['material_id'], row['question'])][row['condition']] = row
    conditions = {condition: _rate(sum(1 for group in groups.values()
                                       if group.get(condition, {}).get('acceptable') is True),
                                   sum(1 for group in groups.values()
                                       if group.get(condition, {}).get('acceptable') is not None))
                  for condition in CONDITIONS}
    selections, labels = [], {}
    for row in read_jsonl(out / 'pool.jsonl'):
        labels[(row['material_id'], row['question'])] = row
    for key, group in groups.items():
        meta = labels.get(key)
        if meta is None:
            continue
        decided = all(group.get(condition, {}).get('acceptable') is not None for condition in CONDITIONS)
        chosen = None
        if decided:
            chosen = bool(group['TI']['acceptable'] and not group['T']['acceptable']
                          and not group['I']['acceptable'])
        selections.append({'material_id': key[0], 'question': key[1], 'label': meta['label'],
                           'dual': meta['dual'], 'selected': bool(chosen), 'decided': decided})
    metrics = {**_selection_metrics([s for s in selections if s['decided']]),
               'undetermined': sum(1 for s in selections if not s['decided'])}
    reference_models = sorted({row.get('model') for row in references.values() if row.get('model')})
    summary = {'run_id': run_id, 'created_at': utc_now(), 'rule': SELECT_RULE, 'model': equivalence_model,
               'reference_models': reference_models,
               'conditions': conditions, **metrics, 'new_judgements': new_judgements}
    write_json(out / 'real_verify_summary.json', summary)
    lines = [f'# Local self-verification on real data {run_id}', '',
             f'Reference answers from `{", ".join(reference_models) or "missing"}`, equivalence judged by '
             f'`{equivalence_model}`; rule **{SELECT_RULE}**.', '',
             '| condition | acceptable |', '|---|---|']
    for condition in CONDITIONS:
        lines.append(f'| {condition} | {_fmt(conditions[condition])} |')
    lines += ['', f'- recall {_fmt(metrics["recall"])}, precision {_fmt(metrics["precision"])}, '
                  f'false-positive rate on non-duals {_fmt(metrics["false_positive_rate"])}',
              f'- undetermined (missing reference or unjudged) {metrics["undetermined"]}', '',
              '| label | selected / n | recall |', '|---|---:|---|']
    for label, bucket in metrics['by_label'].items():
        lines.append(f'| {label} | {bucket["k"]}/{bucket["n"]} | {_fmt(metrics["recall_by_label"].get(label))} |')
    lines.append('')
    (out / 'real_verify_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def _answered(raw):
    """True when the local answer parses to a non-abstaining answer."""
    from .synthetic_composition import _answer_text
    return _answer_text(raw) is not None


def _normalized_answer(raw):
    """Canonical form of the answer for cross-sample comparison; None when the model abstained."""
    text = _answer_text(raw)
    if text is None:
        return None
    text = normalize_text(text).strip(' .,;:!?"\'')
    return text or None


_AGREE_JACCARD = 0.8
_STOPWORDS = {'a', 'an', 'the', 'of', 'is', 'are', 'was', 'were', 'it', 'its', 'and', 'or', 'to', 'in', 'on',
              'that', 'this', 'with', 'as', 'by', 'for', 'from', 'be', 'been', 'has', 'have'}


def _content_tokens(text):
    return {token for token in re.findall(r'[a-z0-9]+', text) if token not in _STOPWORDS}


def _answers_match(first, second):
    """Whether two free-form answers state the same thing.

    Surface forms differ between samples ("Melzer" vs "she was formerly Melzer"), so exact string equality is too
    strict. Two answers match when their content tokens have Jaccard similarity at least `_AGREE_JACCARD`, or when
    the shorter answer's content is contained in the longer one. Containment is what makes a terse answer agree with
    a restated one; it is deliberately asymmetric in spirit but symmetric in the test. Numbers are content tokens
    and are also compared as a set, so "35" never matches "53" and a different value always breaks agreement.
    """
    if not first or not second:
        return False
    if first == second:
        return True
    numbers_first = set(re.findall(r'\d+(?:\.\d+)?', first))
    numbers_second = set(re.findall(r'\d+(?:\.\d+)?', second))
    if numbers_first and numbers_second and numbers_first != numbers_second:
        return False
    left, right = _content_tokens(first), _content_tokens(second)
    if not left or not right:
        return False
    if len(left & right) / len(left | right) >= _AGREE_JACCARD:
        return True
    return (left <= right or right <= left) and len(left & right) >= 1


def _answer_evidence(raws):
    """Three levels of evidence that one side answers the question.

    `answered` is the weak signal (any sample produced an answer). `stable` is whether all samples form a single
    answer cluster, which separates a determinate answer from sampling noise. `modal` is compared against the
    full-material answer by the caller to test whether the single-side answer states the same fact, which is a
    correctness proxy available without references.
    """
    answers = [_normalized_answer(raw) for raw in raws]
    given = [answer for answer in answers if answer]
    if not given:
        return {'answered': False, 'stable': None, 'modal': None, 'n': len(answers), 'answered_n': 0, 'share': None}
    clusters = []
    for answer in given:
        for cluster in clusters:
            if _answers_match(cluster[0], answer):
                cluster.append(answer)
                break
        else:
            clusters.append([answer])
    modal_cluster = max(clusters, key=len)
    modal = max(modal_cluster, key=len)
    share = len(modal_cluster) / len(answers)
    return {'answered': True, 'stable': share == 1.0 and len(answers) > 1, 'modal': modal, 'n': len(answers),
            'answered_n': len(given), 'share': round(share, 4), 'clusters': len(clusters)}


def run_answerability_audit(*, run_id, candidates_path, materials_path, local_model_path, device='cuda:0',
                            conditions=('TI', 'T', 'I', '0'), max_batch=8, max_new_tokens=256, limit=None,
                            samples=1, temperature=0.0, project_root='.', output_root='outputs'):
    """Answer every labelled candidate under each condition locally; no reference answers, no API calls.

    This is the zero-API selection audit: whether the question can be answered with the text alone is a local
    signal, and the consensus labels are the ground truth for whether that signal selects dual questions.
    `samples > 1` draws several answers per condition so the report can separate a determinate answer (all samples
    agree) from a lucky guess, and can compare the single-side answer with the full-material answer.
    """
    from ..core.local_vision import BatchingProvider, QwenVLBackend, qwen_identity
    from ..core.providers import CompletionRequest

    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    pool, materials = prepare_real_pool(candidates_path=candidates_path, materials_path=materials_path,
                                        sample_per_material=0, project_root=project_root)
    write_jsonl(out / 'pool.jsonl', pool)
    prompts = PromptSet(root / 'prompts', names=(SOLVE_PROMPT,))
    plan = []
    for row in pool:
        for condition in conditions:
            for sample_index in range(max(1, samples)):
                if samples <= 1 and temperature <= 0:
                    attempt = stable_id('answerability-audit', run_id, row['material_id'], row['question'], condition)
                else:
                    attempt = stable_id('answerability-audit', run_id, row['material_id'], row['question'], condition,
                                        sample_index, f't{temperature}')
                plan.append({**row, 'condition': condition, 'sample_index': sample_index, 'attempt_id': attempt})
    if limit:
        plan = plan[:limit]
    answers_path = out / 'answers.jsonl'
    prior = {row['attempt_id'] for row in read_jsonl(answers_path)} if answers_path.exists() else set()
    pending = [slot for slot in plan if slot['attempt_id'] not in prior]
    if pending:
        backend = QwenVLBackend(local_model_path, device=device, max_new_tokens=max_new_tokens,
                                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                top_p=0.9, top_k=20)
        identity = qwen_identity(backend, 'answerability-audit-v1')
        batch_provider = BatchingProvider(backend, model=backend.path.name, identity=identity, max_batch=max_batch)
        lock = threading.Lock()

        def work(slot):
            material = materials[slot['material_id']]
            image = str(root / material['image_path']) if material.get('image_path') else None
            if slot['condition'] not in ('TI', 'I'):
                image = None
            completion = batch_provider.complete(CompletionRequest(
                task=f'answerability-audit-{slot["condition"]}',
                prompt=_solve_prompt(prompts, material, slot['question'], slot['condition']),
                question=slot['question'], text=material.get('text') if slot['condition'] in ('TI', 'T') else '',
                image_path=image, condition=slot['condition']))
            with lock:
                append_jsonl(answers_path, {**slot, 'raw_output': completion.text,
                                            'finish_reason': completion.finish_reason,
                                            'completed_at': utc_now()})

        with ThreadPoolExecutor(max_workers=max(2, max_batch * 2)) as executor:
            futures = [executor.submit(work, slot) for slot in pending]
            for future in as_completed(futures):
                future.result()
    return {'run_id': run_id, 'pool': len(pool), 'planned': len(plan), 'already_done': len(plan) - len(pending),
            'new': len(pending)}


def report_answerability_audit(*, run_id, bootstrap=10000, pvs_selection=None, project_root='.', output_root='outputs'):
    """Score local text-answerability screens as dual-question selectors against the consensus labels."""
    root = Path(project_root).resolve()
    directory = root / output_root / run_id
    pool = read_jsonl(directory / 'pool.jsonl')
    answers = read_jsonl(directory / 'answers.jsonl')
    collected = defaultdict(lambda: defaultdict(list))
    for row in answers:
        collected[(row['material_id'], row['question'])][row['condition']].append(row.get('raw_output') or '')
    evidence = {key: {condition: _answer_evidence(raws) for condition, raws in per.items()}
                for key, per in collected.items()}

    def sufficient(ev, side, level):
        """Whether `side` supplies enough evidence that it alone answers the question."""
        if side == 'either':
            return sufficient(ev, 'T', level) or sufficient(ev, 'I', level)
        side_ev = ev.get(side) or {}
        if not side_ev.get('answered'):
            return False
        if level == 'exists':
            return True
        full = (ev.get('TI') or {}).get('modal')
        agrees = _answers_match(side_ev.get('modal'), full)
        if level == 'stable':
            return side_ev.get('stable') is True
        if level == 'agrees':
            return agrees
        if level == 'stable_and_agrees':
            return side_ev.get('stable') is True and agrees
        raise ValueError(level)

    max_samples = max([len(per[c]) for per in collected.values() for c in per], default=1)
    levels = [('exists', 'answers'), ('agrees', 'agrees_full')]
    if max_samples > 1:
        levels = [('exists', 'answers'), ('stable', 'stable'), ('agrees', 'agrees_full'),
                  ('stable_and_agrees', 'stable_and_agrees')]
    sides = (('T', 'text'), ('I', 'image'), ('either', 'either_side'))
    rules = {}
    for side_key, side_name in sides:
        for level_key, level_name in levels:
            rules[f'reject_if_{side_name}_{level_name}'] = (side_key, level_key)
    selections = {name: [] for name in rules}
    # The blind condition ('0') answers with no material at all, so it measures whether the question is
    # answerable from prior knowledge rather than from either side.
    prior_known = any('0' in per for per in collected.values())
    if prior_known:
        for name in ('reject_if_prior_answers', 'reject_if_either_side_or_prior_answers'):
            selections[name] = []
    for row in pool:
        key = (row['material_id'], row['question'])
        ev = evidence.get(key, {})
        answered = {condition: (ev.get(condition) or {}).get('answered') for condition in CONDITIONS}
        for name, (side, level) in rules.items():
            selections[name].append({**row, 'answered': answered, 'evidence': ev,
                                     'selected': not sufficient(ev, side, level)})
        for name, (side, level) in (('reject_if_text_answers_and_ti', ('T', 'exists')),
                                    ('reject_if_either_side_answers_and_ti', ('either', 'exists'))):
            ti_known = bool((ev.get('TI') or {}).get('answered'))
            selections.setdefault(name, []).append({**row, 'answered': answered, 'evidence': ev,
                                                    'selected': ti_known and not sufficient(ev, side, level)})
        if prior_known:
            prior_answers = bool((ev.get('0') or {}).get('answered'))
            selections['reject_if_prior_answers'].append(
                {**row, 'answered': answered, 'evidence': ev, 'selected': not prior_answers})
            selections['reject_if_either_side_or_prior_answers'].append(
                {**row, 'answered': answered, 'evidence': ev,
                 'selected': not (sufficient(ev, 'either', 'exists') or prior_answers)})
    metrics = {}
    for name, rows in selections.items():
        entry = _selection_metrics(rows)
        entry.update(_material_bootstrap_answerability(rows, bootstrap))
        entry['text_answered_rate'] = _rate(sum(1 for r in rows if r['answered'].get('T')),
                                            sum(1 for r in rows if r['answered'].get('T') is not None))
        entry['image_answered_rate'] = _rate(sum(1 for r in rows if r['answered'].get('I')),
                                             sum(1 for r in rows if r['answered'].get('I') is not None))
        if prior_known:
            entry['prior_answered_rate'] = _rate(sum(1 for r in rows if (r['evidence'].get('0') or {}).get('answered')),
                                                 len(rows))
        entry['coverage'] = _coverage_answerability(rows)
        metrics[name] = entry
    ladder = {}
    base_rows = selections['reject_if_text_answers']
    for side_key in (('T', 'I', '0') if prior_known else ('T', 'I')):
        levels_out = {}
        for scope, subset in (('all', base_rows), ('duals', [r for r in base_rows if r['dual']])):
            side_evs = [(r['evidence'].get(side_key) or {}) for r in subset]
            total = sum(1 for item in side_evs if item)
            levels_out[scope] = {
                'n': total,
                'answered': _rate(sum(1 for item in side_evs if item.get('answered')), total),
                'stable': _rate(sum(1 for item in side_evs if item.get('stable')), total),
                'agrees_full': _rate(sum(1 for r, item in zip(subset, side_evs)
                                         if _answers_match(item.get('modal'),
                                                           (r['evidence'].get('TI') or {}).get('modal'))), total),
            }
        ladder[side_key] = levels_out['all']
        ladder[side_key]['on_duals'] = levels_out['duals']
    if all(ladder[side].get('on_duals', {}).get('n') for side in ('T', 'I')):
        left, right = ladder['T']['on_duals']['answered']['p'], ladder['I']['on_duals']['answered']['p']
        ladder['predicted_dual_rejection'] = round(1 - (1 - left) * (1 - right), 4)
    external = {}
    if pvs_selection:
        chosen = {}
        for row in read_jsonl(Path(pvs_selection)):
            chosen[(row['material_id'], row['question'])] = row.get('sol_final_label') == 'dual_candidate'
        labelled = []
        for row in pool:
            picked = chosen.get((row['material_id'], row['question']))
            labelled.append({**row, 'selected': bool(picked), 'pvs_known': picked is not None})
        known = [row for row in labelled if row['pvs_known']]
        external['api_pvs'] = {**_selection_metrics(known), 'matched': len(known), 'unmatched': len(pool) - len(known)}
        for name in ('reject_if_text_answers', 'reject_if_either_side_answers',
                     'reject_if_text_agrees_full', 'reject_if_either_side_agrees_full',
                     'reject_if_prior_answers', 'reject_if_either_side_or_prior_answers'):
            if name not in selections:
                continue
            local = selections[name]
            both = [{**row, 'selected': row['selected'] and ext['selected']} for row, ext in zip(local, labelled)]
            either = [{**row, 'selected': row['selected'] or ext['selected']} for row, ext in zip(local, labelled)]
            external[f'{name}_and_api'] = _selection_metrics(both)
            external[f'{name}_or_api'] = _selection_metrics(either)
            external[f'{name}_vs_api'] = _selector_comparison(local, labelled, bootstrap)
    summary = {'run_id': run_id, 'created_at': utc_now(), 'questions': len(pool),
               'samples': max_samples, 'prior_known': prior_known,
               'labels': {label: sum(1 for r in pool if r['label'] == label)
                          for label in sorted({r['label'] for r in pool})},
               'dual_rate': _rate(sum(1 for r in pool if r['dual']), len(pool)),
               'evidence_ladder': ladder,
               'rules': metrics}
    if external:
        summary['external'] = external
    write_json(directory / 'audit_summary.json', summary)
    (directory / 'audit_report.md').write_text(_render_answerability(summary), encoding='utf-8')
    return summary


def _selector_comparison(local_rows, api_rows, bootstrap):
    """Material-clustered paired comparison of a local rule against the API PVS selections."""
    from ..core.stats import paired_bootstrap
    groups = [row['material_id'] for row in local_rows]

    def precision(flags, index):
        chosen = [flags[i] for i in index if flags[i][0]]
        return float('nan') if not chosen else sum(1 for _s, dual in chosen if dual) / len(chosen)

    def recall(flags, index):
        duals = [flags[i] for i in index if flags[i][1]]
        return float('nan') if not duals else sum(1 for selected, _d in duals if selected) / len(duals)

    local = [(bool(row['selected']), bool(row['dual'])) for row in local_rows]
    api = [(bool(row['selected']), bool(row['dual'])) for row in api_rows]
    return {
        'precision_diff': paired_bootstrap(lambda i: precision(local, i) - precision(api, i), groups, n=bootstrap, seed=0),
        'recall_diff': paired_bootstrap(lambda i: recall(local, i) - recall(api, i), groups, n=bootstrap, seed=0),
    }


def _coverage_answerability(rows):
    by_material = defaultdict(list)
    for row in rows:
        by_material[row['material_id']].append(row)
    with_selected = sum(1 for group in by_material.values() if any(r['selected'] for r in group))
    return {'materials': len(by_material), 'materials_with_selected': with_selected,
            'share': _rate(with_selected, len(by_material)),
            'mean_selected': round(sum(1 for r in rows if r['selected']) / max(len(by_material), 1), 3)}


def _material_bootstrap_answerability(rows, bootstrap):
    """Material-clustered bootstrap of selection precision and of the lift over the base dual rate."""
    from ..core.stats import paired_bootstrap
    groups = [row['material_id'] for row in rows]

    def precision(index):
        chosen = [rows[i] for i in index if rows[i]['selected']]
        return float('nan') if not chosen else sum(1 for r in chosen if r['dual']) / len(chosen)

    def lift(index):
        chosen = [rows[i] for i in index if rows[i]['selected']]
        if not chosen:
            return float('nan')
        base = sum(1 for i in index if rows[i]['dual']) / len(index)
        return sum(1 for r in chosen if r['dual']) / len(chosen) - base

    return {'precision_interval': paired_bootstrap(precision, groups, n=bootstrap, seed=0),
            'lift_interval': paired_bootstrap(lift, groups, n=bootstrap, seed=0)}


def _interval(result):
    if not result or result.get('lo') is None or result['lo'] != result['lo']:
        return '[—]'
    return f'[{result["lo"]:+.3f}, {result["hi"]:+.3f}]'


def _render_answerability(summary):
    samples = summary.get('samples') or 1
    lines = [f'# Local answerability audit {summary["run_id"]}', '',
             f'{summary["questions"]} labelled questions, consensus dual rate {_fmt(summary["dual_rate"])}, '
             f'{samples} local answer(s) per condition. '
             'A question is selected when the local model **cannot** answer it from the side that would make '
             'the question redundant: `reject_if_text_answers` rejects questions the passage alone answers '
             '(so the question is image-redundant / anchored on text), `reject_if_image_answers` rejects '
             'questions the image alone answers (text-redundant / anchored on the image), and '
             '`reject_if_either_side_answers` requires both single sides to fail. '
             '`reject_if_prior_answers` rejects questions the model answers with **no material at all**, that is, '
             'from prior knowledge: no side-based test can see them, because the question is answerable without '
             'either side being sufficient.',
             '',
             'The rule suffix is the strength of the sufficiency evidence: `answers` (any sample produced an '
             'answer), `stable` (all samples agree), `agrees_full` (the answer equals the full-material answer), '
             '`stable_and_agrees` (both). Stronger evidence rejects fewer questions, trading precision for recall.',
             '', '| rule | selected | recall | precision | FPR | text-answered | image-answered | prior-answered '
             '| materials with a selection |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, entry in summary['rules'].items():
        lines.append(f'| {name} | {entry["selected"]} | {_fmt(entry["recall"])} | {_fmt(entry["precision"])} '
                     f'| {_fmt(entry["false_positive_rate"])} | {_fmt(entry["text_answered_rate"])} '
                     f'| {_fmt(entry.get("image_answered_rate"))} '
                     f'| {_fmt(entry.get("prior_answered_rate"))} '
                     f'| {entry["coverage"]["materials_with_selected"]}/{entry["coverage"]["materials"]} |')
    ladder = summary.get('evidence_ladder') or {}
    if ladder:
        lines += ['', 'Sufficiency evidence per side (share of questions at each level). '
                      'The `0` side is the blind condition, answered with no material at all:', '',
                  '| side | scope | n | answers | stable | agrees with full material |', '|---|---|---:|---:|---:|---:|']
        for side in ('T', 'I', '0'):
            entry = ladder.get(side) or {}
            if not entry:
                continue
            for scope, values in (('all', entry), ('dual only', entry.get('on_duals') or {})):
                if not values:
                    continue
                lines.append(f'| {side} | {scope} | {values["n"]} | {_fmt(values["answered"])} '
                             f'| {_fmt(values["stable"])} | {_fmt(values["agrees_full"])} |')
        prior = ((ladder.get('0') or {}).get('on_duals') or {})
        if prior.get('answered'):
            lines += ['', f'Blind answerability on genuine duals is {_fmt(prior["answered"])}: the model produces '
                          'an answer with no material at all, so those questions are answerable from prior '
                          'knowledge rather than from the pair. A selector that estimates only the two negations '
                          'cannot separate them from real duals, which is why `reject_if_prior_answers` belongs to '
                          'the method rather than being a refinement.']
        predicted = ladder.get('predicted_dual_rejection')
        if predicted is not None:
            rule = summary['rules'].get('reject_if_either_side_answers') or {}
            observed = None if not rule.get('recall') else round(1 - rule['recall']['p'], 4)
            lines += ['', f'The two-sided rule rejects a genuine dual when either side happens to answer. '
                          f'Assuming the two sides are independent, that rate is '
                          f'1 - (1 - P(T answers | dual)) (1 - P(I answers | dual)) = {predicted:+.3f}; '
                          f'the observed rejection rate among duals is {_fmt(observed)} '
                          f'(= 1 - recall). A large gap means the sides are not independent or the label is noisy.']
    if summary.get('external'):
        lines += ['', '| selector | selected | recall | precision | FPR |', '|---|---:|---:|---:|---:|']
        for name, entry in summary['external'].items():
            if 'selected' not in entry:
                continue
            lines.append(f'| {name} | {entry["selected"]} | {_fmt(entry["recall"])} | {_fmt(entry["precision"])} '
                         f'| {_fmt(entry["false_positive_rate"])} |')
        compared = {name: entry for name, entry in summary['external'].items() if 'precision_diff' in entry}
        if compared:
            lines += ['', 'Material-clustered bootstrap of the local minus API difference (paired on the same questions):', '',
                      '| local rule | precision diff | recall diff |', '|---|---:|---:|']
            for name, entry in compared.items():
                lines.append(f'| {name} | {_interval(entry["precision_diff"])} | {_interval(entry["recall_diff"])} |')
    lines += ['', 'Precision interval is a material-clustered bootstrap:', '']
    for name, entry in summary['rules'].items():
        lines.append(f'- **{name}**: precision {_fmt(entry["precision"])} '
                     f'{_interval(entry.get("precision_interval"))}, lift {_interval(entry.get("lift_interval"))}')
    return '\n'.join(lines) + '\n'
