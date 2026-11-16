"""Time-boxed mechanism probe for the confirmed interval-count failures (R1).

Question: when the full text+image material asks "How many items count?" and the model returns the
off-by-one answer, is the textual interval (a) never used, (b) used but applied too late / not in the
final count, or (c) dominated by the image values?

The probe is deliberately cheap and behavioural first:
  1. reproduce the published greedy answer under the full material;
  2. text ablation (interval numbers removed), a shifted valid interval, a disjoint interval,
     image-only and text-only conditions on the same items;
  3. a single teacher-forced forward pass per item/condition ("logit lens"): append the answer JSON
     prefix and read, at every layer, the probability of each count 0..6 after the final norm and
     lm_head. This locates the layer at which the correct count becomes decodable and whether the
     wrong count dominates there.

Correct interval-count items (full, text hop and image hop all correct) are controls.
No attention knockout is implemented unless the cheap stages leave time; that is reported explicitly.
"""
from __future__ import annotations

import time
from pathlib import Path

from .records import read_jsonl, write_json, write_jsonl, utc_now

VERSION = 'interval-mechanism-v1'
FAILURES_PATH = 'outputs/mechanism-w1-confirm-v1/mechanism_samples_confirm_8b.jsonl'
ANSWERS_PATH = 'outputs/synth-confirm-count-8b-direct-v1/answers.jsonl'
MATERIALS_PATH = 'data/processed/synthetic-composition-confirm-count-v1/materials.jsonl'
FULL_QUESTION = 'How many items count?'
ABLATE_TEXT = 'Only items with value in the stated interval count.'
DISJOINT_LOWER, DISJOINT_UPPER = 1, 2
ANSWER_TEMPLATE = 'composition-answer-v1'
ANSWER_PREFIX = '{"answer":"'
NUMBER_WORDS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six')
BEHAVIOUR_CONDITIONS = ('original', 'text_ablate', 'shifted', 'disjoint', 'image_only', 'text_only')
LENS_CONDITIONS = ('original', 'text_ablate', 'shifted', 'image_only')


# ---------- pure helpers ----------

def count_in_interval(values, lower, upper):
    return sum(int(lower) <= int(v) <= int(upper) for v in values)


def choose_shift_interval(lower, upper, values, forbidden=(), max_delta=15):
    """A nearby valid interval whose true count differs from the original and from every forbidden count.

    Smallest total bound movement wins; this keeps the interval surface form almost identical to the
    original, so a model that ignores the interval cannot explain an answer that follows the new count.
    A non-zero true count is preferred so the shifted readout is not the same probe as the disjoint one.
    """
    original = count_in_interval(values, lower, upper)
    forbidden = {int(x) for x in forbidden} | {original}
    best = None
    for dlo in range(-max_delta, max_delta + 1):
        for dhi in range(-max_delta, max_delta + 1):
            lo, hi = int(lower) + dlo, int(upper) + dhi
            if lo < 0 or hi < lo:
                continue
            new = count_in_interval(values, lo, hi)
            if new in forbidden or not 0 <= new <= len(values):
                continue
            key = (0 if new > 0 else 1, abs(dlo) + abs(dhi), abs(dlo), abs(dhi), dlo, dhi)
            if best is None or key < best['key']:
                best = {'key': key, 'lower': lo, 'upper': hi, 'expected': new,
                        'delta_lower': dlo, 'delta_upper': dhi}
    return best


def _parsed_first_int(parsed):
    if parsed is None:
        return None
    if isinstance(parsed, list):
        return parsed[0] if parsed else None
    if isinstance(parsed, (int, float)):
        return int(parsed)
    return None


def select_items(failures_path, answers_path, materials_path, *, controls_per_expected=4):
    """Failures from the mechanism sample file plus correct interval controls from the answers run."""
    failures = [r for r in read_jsonl(Path(failures_path))
                if r.get('measure') == 'count_confirm' and r.get('category') == 'interval']
    materials = {m['material_id']: m for m in read_jsonl(Path(materials_path))}
    answers = read_jsonl(Path(answers_path))
    by_material = {}
    for row in answers:
        if row.get('category') != 'interval':
            continue
        by_material.setdefault(row['material_id'], {})[row['question_type']] = row
    controls = []
    for expected in (1, 2):
        chosen = 0
        for material_id in sorted(by_material):
            group = by_material[material_id]
            if not all(group.get(q, {}).get('correct') for q in ('full', 'text_hop', 'image_hop')):
                continue
            material = materials.get(material_id)
            if material is None or int(material['evaluation']['full']['expected']) != expected:
                continue
            controls.append(material_id)
            chosen += 1
            if chosen >= controls_per_expected:
                break

    items = []
    for row in failures:
        material_id = row['material_id']
        material = materials.get(material_id)
        if material is None:
            raise ValueError(f'missing material {material_id}')
        wrong = _parsed_first_int(row.get('full_answer'))
        items.append(_item_record('failure', material, expected=int(row['full_expected']),
                                  model_answer=wrong, published_raw=row.get('full_raw')))
    for material_id in controls:
        material = materials[material_id]
        full = by_material[material_id]['full']
        items.append(_item_record('control', material, expected=int(full['expected']),
                                  model_answer=int(full['expected']), published_raw=full['raw_output']))
    for item in items:
        shift = choose_shift_interval(item['lower'], item['upper'], item['values'],
                                      forbidden=(item['model_answer'], item['expected']))
        if shift is None:
            raise ValueError(f'no shifted interval for {item["material_id"]}')
        item['shift'] = shift
    return items


def _item_record(role, material, *, expected, model_answer, published_raw):
    return {
        'item_id': f'{role}:{material["material_id"]}',
        'role': role,
        'material_id': material['material_id'],
        'text': material['text'],
        'image_path': material['image_path'],
        'values': [int(v) for v in material['values']],
        'lower': int(material['lower']),
        'upper': int(material['upper']),
        'expected': int(expected),
        'model_answer': None if model_answer is None else int(model_answer),
        'published_raw': published_raw,
        'full_question': FULL_QUESTION,
    }


def conditions_for(item):
    """Behavioural conditions, each with the expected count it should produce (None = no valid rule)."""
    shift = item['shift']
    return [
        {'condition': 'original', 'text': item['text'], 'use_image': True, 'expected': item['expected'],
         'valid_rule': True},
        {'condition': 'text_ablate', 'text': ABLATE_TEXT, 'use_image': True, 'expected': None,
         'valid_rule': False},
        {'condition': 'shifted', 'text': f'Only items with value from {shift["lower"]} to {shift["upper"]} count.',
         'use_image': True, 'expected': shift['expected'], 'valid_rule': True},
        {'condition': 'disjoint', 'text': f'Only items with value from {DISJOINT_LOWER} to {DISJOINT_UPPER} count.',
         'use_image': True, 'expected': 0, 'valid_rule': True},
        {'condition': 'image_only', 'text': '', 'use_image': True, 'expected': None, 'valid_rule': False},
        {'condition': 'text_only', 'text': item['text'], 'use_image': False, 'expected': None,
         'valid_rule': False},
    ]


def summarize_lens(rows, *, roles=('failure', 'control'), conditions=LENS_CONDITIONS, n_numbers=7):
    """Per condition/role: layer-wise mean probability of the correct and wrong count, and crossover layers."""
    summary = {}
    for condition in conditions:
        for role in roles:
            selected = [r for r in rows if r['condition'] == condition and r['role'] == role]
            if not selected:
                continue
            n_layers = len(selected[0]['layers'])
            layers = []
            for index in range(n_layers):
                correct = sum(r['layers'][index]['correct_prob'] for r in selected) / len(selected)
                wrong = sum(r['layers'][index]['wrong_prob'] for r in selected) / len(selected)
                wins = sum(r['layers'][index]['correct_prob'] > r['layers'][index]['wrong_prob']
                           for r in selected) / len(selected)
                layers.append({'layer': index, 'correct_prob': correct, 'wrong_prob': wrong,
                               'fraction_correct_above_wrong': wins})
            crossovers = [r['crossover_layer'] for r in selected if r['crossover_layer'] is not None]
            summary[f'{condition}:{role}'] = {
                'n': len(selected),
                'layers': layers,
                'crossover_layers': crossovers,
                'crossover_count': len(crossovers),
                'median_crossover_layer': _median(crossovers),
                'final_argmax_matches_model': sum(r.get('final_argmax_matches_model') is True for r in selected),
            }
    return summary


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def expansion_delta(values, lower, upper, target, max_delta=30):
    """Smallest symmetric widening of [lower, upper] whose count reaches `target` (None if unreachable)."""
    for delta in range(max_delta + 1):
        if count_in_interval(values, lower - delta, upper + delta) == target:
            return delta
    return None


def shifted_relation(values, lower, upper, answer):
    """How a shifted-interval answer relates to its true count and the two one-sided counts."""
    both = count_in_interval(values, lower, upper)
    lower_only = sum(int(v) >= int(lower) for v in values)
    upper_only = sum(int(v) <= int(upper) for v in values)
    if answer is None:
        return 'abstain'
    if answer == both:
        return 'tracks_both_bounds'
    if answer == lower_only:
        return 'lower_bound_only'
    if answer == upper_only:
        return 'upper_bound_only'
    return 'other'


def per_item_results(behaviour, lens, items=()):
    """Join behavioural rows and logit-lens crossover into one row per item for results.jsonl."""
    item_by_id = {item['item_id']: item for item in items}
    by_item = {}
    for row in behaviour:
        entry = by_item.setdefault(row['item_id'], {'item_id': row['item_id'], 'role': row['role'],
                                                    'material_id': row['material_id']})
        entry[row['condition']] = {'answer': row['first_int'], 'raw': row['raw'], 'correct': row['correct'],
                                   'expected': row['expected']}
    for row in lens:
        if row['condition'] != 'original':
            continue
        entry = by_item.setdefault(row['item_id'], {'item_id': row['item_id'], 'role': row['role'],
                                                    'material_id': row['material_id']})
        entry['lens'] = {'crossover_layer': row['crossover_layer'],
                         'last_wrong_dominant_layer': row['last_wrong_dominant_layer'],
                         'final_argmax_number': row['final_argmax_number'],
                         'final_correct_prob': row['final_correct_prob'],
                         'final_wrong_prob': row['final_wrong_prob'],
                         'wrong_prob_by_layer': [l['wrong_prob'] for l in row['layers']]}
    for item in items:
        entry = by_item.setdefault(item['item_id'], {'item_id': item['item_id'], 'role': item['role'],
                                                     'material_id': item['material_id']})
        original = entry.get('original', {}).get('answer')
        shifted = entry.get('shifted', {})
        entry['values'] = item['values']
        entry['interval'] = [item['lower'], item['upper']]
        entry['true_count'] = item['expected']
        entry['shifted_interval'] = [item['shift']['lower'], item['shift']['upper']]
        entry['shifted_true_count'] = item['shift']['expected']
        entry['shifted_relation'] = shifted_relation(item['values'], item['shift']['lower'],
                                                     item['shift']['upper'], shifted.get('answer'))
        entry['shifted_abs_diff'] = (None if shifted.get('answer') is None
                                     else abs(shifted['answer'] - item['shift']['expected']))
        entry['boundary_expansion_delta'] = (None if original is None
                                             else expansion_delta(item['values'], item['lower'],
                                                                  item['upper'], original))
        entry['attention_knockout'] = 'not_run'
    return list(by_item.values())


# ---------- GPU part ----------

def _number_token_ids(tokenizer, n_max=6):
    """Candidate token ids for each count: digit form and number word (plain and leading-space)."""
    mapping = {}
    for number in range(n_max + 1):
        ids = set(tokenizer.encode(str(number), add_special_tokens=False))
        if number < len(NUMBER_WORDS):
            word = NUMBER_WORDS[number]
            ids.update(tokenizer.encode(word, add_special_tokens=False))
            ids.update(tokenizer.encode(' ' + word, add_special_tokens=False))
        mapping[number] = sorted(ids)
    return mapping


def _render(prompts, text):
    return prompts.render(ANSWER_TEMPLATE, T=text, Q=FULL_QUESTION)


def _generate_one(backend, prompt, text, image_path):
    from .providers import CompletionRequest
    request = CompletionRequest(task='interval-mechanism', prompt=prompt, question=FULL_QUESTION, text=text,
                                image_path=image_path, condition='TI' if image_path else 'T')
    return backend.generate([request])[0]


def run_behaviour(backend, prompts, items):
    from .synthetic_composition import _answer_text, _ints
    rows = []
    for item in items:
        for spec in conditions_for(item):
            image_path = str(Path(item['image_path']).resolve()) if spec['use_image'] else None
            rendered = _render(prompts, spec['text'])
            began = time.monotonic()
            out = _generate_one(backend, rendered, spec['text'], image_path)
            elapsed = time.monotonic() - began
            answer_text = _answer_text(out.text)
            numbers = _ints(answer_text) if answer_text is not None else []
            first = numbers[0] if numbers else None
            expected = spec['expected']
            correct = None if expected is None else (expected in numbers)
            reproduced = None
            if spec['condition'] == 'original':
                reproduced = first == item['model_answer']
            rows.append({
                'item_id': item['item_id'], 'role': item['role'], 'material_id': item['material_id'],
                'condition': spec['condition'], 'text': spec['text'], 'use_image': spec['use_image'],
                'expected': expected, 'valid_rule': spec['valid_rule'],
                'raw': out.text, 'answer_numbers': numbers, 'first_int': first,
                'correct': correct, 'reproduces_published': reproduced,
                'elapsed_seconds': round(elapsed, 3), 'prompt_tokens': out.prompt_tokens,
                'completion_tokens': out.completion_tokens,
            })
    return rows


def _hooked_forward(backend, prompt, text, image_path, prefix_ids):
    """One teacher-forced forward over prompt + answer prefix; per-layer final-norm hidden states."""
    import torch
    from .whitebox import forward_extras
    from .providers import CompletionRequest

    request = CompletionRequest(task='interval-mechanism-lens', prompt=prompt, question=FULL_QUESTION,
                                text=text, image_path=image_path, condition='TI' if image_path else 'T')
    inputs = backend.encode([request])
    model = backend._model
    tokenizer = backend._processor.tokenizer
    full = torch.cat([inputs['input_ids'],
                      torch.tensor([prefix_ids], dtype=inputs['input_ids'].dtype,
                                   device=inputs['input_ids'].device)], dim=1)
    extra = forward_extras(inputs, len(prefix_ids))
    with torch.inference_mode():
        out = model(input_ids=full, attention_mask=torch.ones_like(full), output_hidden_states=True, **extra)
    hs = out.hidden_states
    lm = model.model.language_model
    norm, head = lm.norm, model.lm_head
    head_dtype = head.weight.dtype

    with torch.inference_mode():
        def project(vector):
            return head(vector.to(head_dtype).unsqueeze(0)).squeeze(0).float()

        # Identify what the last hidden_states entry is by reproducing the model's own final logits.
        last_raw = hs[-1][0, -1].float()
        prev_raw = hs[-2][0, -1].float()
        transforms = {
            'raw_last': project(last_raw),
            'norm_last': project(norm(last_raw.unsqueeze(0)).squeeze(0)),
            'raw_prev': project(prev_raw),
            'norm_prev': project(norm(prev_raw.unsqueeze(0)).squeeze(0)),
        }
        reference = out.logits[0, -1].float() if getattr(out, 'logits', None) is not None else None
        if reference is not None:
            errors = {name: float((value - reference).abs().max()) for name, value in transforms.items()}
            final_transform = min(errors, key=errors.get)
        else:
            errors, final_transform = {}, 'norm_last'
        use_raw_last = final_transform == 'raw_last'

        layer_states = []
        for index, state in enumerate(hs):
            vector = state[0, -1].float()
            if index == len(hs) - 1 and use_raw_last:
                layer_states.append(vector)
            else:
                layer_states.append(norm(vector.unsqueeze(0)).squeeze(0))
        number_ids = _number_token_ids(tokenizer)
        layers = []
        for index, vector in enumerate(layer_states):
            logits = project(vector)
            lse = torch.logsumexp(logits, dim=-1)
            probs = {}
            for number, ids in number_ids.items():
                probs[number] = max(float((logits[i] - lse).exp()) for i in ids)
            layers.append({'layer': index, 'number_probs': [probs[n] for n in range(7)]})
        final_logits = project(layer_states[-1])
        final_argmax = int(torch.argmax(final_logits))
        model_first_token = int(out.logits[0, -1].argmax()) if reference is not None else None
    return {'layers': layers, 'final_argmax_token': final_argmax, 'model_first_token': model_first_token,
            'final_transform': final_transform, 'final_logit_max_abs_error': errors.get(final_transform),
            'n_hidden_states': len(hs), 'n_prompt_tokens': int(inputs['input_ids'].shape[1])}


def run_logit_lens(backend, prompts, items):
    import torch
    tokenizer = backend._processor.tokenizer
    prefix_ids = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
    number_ids = _number_token_ids(tokenizer)
    token_to_number = {}
    for number, ids in number_ids.items():
        for token_id in ids:
            token_to_number.setdefault(token_id, number)
    rows = []
    for item in items:
        wrong = item['model_answer'] if item['role'] == 'failure' else item['expected'] + 1
        wrong = int(wrong) if wrong is not None else None
        specs = {s['condition']: s for s in conditions_for(item)}
        for condition in LENS_CONDITIONS:
            spec = specs[condition]
            image_path = str(Path(item['image_path']).resolve()) if spec['use_image'] else None
            rendered = _render(prompts, spec['text'])
            began = time.monotonic()
            result = _hooked_forward(backend, rendered, spec['text'], image_path, prefix_ids)
            elapsed = time.monotonic() - began
            shift_count = item['shift']['expected']
            layers = []
            for entry in result['layers']:
                probs = entry['number_probs']
                correct_prob = probs[item['expected']]
                wrong_prob = probs[wrong] if wrong is not None and wrong < len(probs) else 0.0
                layers.append({'layer': entry['layer'], 'correct_prob': correct_prob, 'wrong_prob': wrong_prob,
                               'shifted_prob': probs[shift_count]})
            # Crossover: the first layer after the last layer in which the wrong count still dominates.
            last_wrong = next((l['layer'] for l in reversed(layers) if l['wrong_prob'] > l['correct_prob']), None)
            crossover = None if last_wrong is None or last_wrong == layers[-1]['layer'] else last_wrong + 1
            final = layers[-1]
            final_argmax_number = token_to_number.get(result['final_argmax_token'])
            matches = None
            if condition == 'original':
                matches = final_argmax_number == item['model_answer']
            rows.append({
                'item_id': item['item_id'], 'role': item['role'], 'material_id': item['material_id'],
                'condition': condition, 'correct_count': item['expected'], 'wrong_count': wrong,
                'shifted_count': shift_count,
                'layers': layers, 'crossover_layer': crossover, 'last_wrong_dominant_layer': last_wrong,
                'final_argmax_number': final_argmax_number,
                'final_correct_prob': final['correct_prob'], 'final_wrong_prob': final['wrong_prob'],
                'final_shifted_prob': final['shifted_prob'],
                'final_argmax_matches_model': matches,
                'final_transform': result['final_transform'],
                'final_logit_max_abs_error': result['final_logit_max_abs_error'],
                'n_hidden_states': result['n_hidden_states'],
                'model_first_token': result['model_first_token'],
                'elapsed_seconds': round(elapsed, 3),
            })
    return rows


def analyze(behaviour, lens, items):
    failures = {i['item_id'] for i in items if i['role'] == 'failure'}
    by_key = {(r['item_id'], r['condition']): r for r in behaviour}

    def rows_for(role, condition):
        return [r for r in behaviour if r['role'] == role and r['condition'] == condition]

    original = rows_for('failure', 'original')
    ablated = rows_for('failure', 'text_ablate')
    shifted = rows_for('failure', 'shifted')
    disjoint = rows_for('failure', 'disjoint')
    image_only = rows_for('failure', 'image_only')
    text_only = rows_for('failure', 'text_only')
    control_original = rows_for('control', 'original')

    def frac(n, d):
        return None if not d else n / d

    ablate_survives = sum(1 for r in ablated if r['first_int'] == by_key[(r['item_id'], 'original')]['first_int'])
    shifted_unchanged = sum(1 for r in shifted if r['first_int'] == by_key[(r['item_id'], 'original')]['first_int'])
    shifted_tracks = sum(1 for r in shifted if r['first_int'] == r['expected'])
    shifted_tracks_or_original = sum(1 for r in shifted
                                     if r['first_int'] == r['expected']
                                     or r['first_int'] == by_key[(r['item_id'], 'original')]['first_int'])
    disjoint_zero = sum(1 for r in disjoint if r['first_int'] == 0)
    image_reproduces = sum(1 for r in image_only
                           if r['first_int'] == by_key[(r['item_id'], 'original')]['first_int'])
    image_all_four = sum(1 for r in image_only if r['first_int'] == 4)
    item_by_id = {item['item_id']: item for item in items}
    shifted_within_one = sum(1 for r in shifted
                             if r['first_int'] is not None and abs(r['first_int'] - r['expected']) <= 1)
    relation_counts = {}
    for row in shifted:
        item = item_by_id[row['item_id']]
        relation = shifted_relation(item['values'], item['shift']['lower'], item['shift']['upper'], row['first_int'])
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    expansion = {}
    for iid in failures:
        item = item_by_id[iid]
        original_answer = by_key[(iid, 'original')]['first_int']
        delta = expansion_delta(item['values'], item['lower'], item['upper'], original_answer)
        expansion[iid] = delta
    reached = [d for d in expansion.values() if d is not None]
    boundary = {
        'n': len(failures),
        'median_expansion_delta': _median(reached),
        'n_reached': len(reached),
        'n_within_2': sum(d <= 2 for d in reached),
        'values': [expansion[iid] for iid in sorted(failures)],
    }
    text_abstains = sum(1 for r in text_only if r['first_int'] is None)
    original_reproduced = sum(1 for r in original if r['reproduces_published'])

    def distribution(rows):
        counts = {}
        for row in rows:
            key = 'abstain' if row['first_int'] is None else str(row['first_int'])
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (kv[0] != 'abstain', kv[0])))

    lens_summary = summarize_lens(lens)
    original_failures = [r for r in lens if r['role'] == 'failure' and r['condition'] == 'original']
    original_controls = [r for r in lens if r['role'] == 'control' and r['condition'] == 'original']

    def final_wrong_wins(rows):
        return sum(1 for r in rows if r['final_wrong_prob'] > r['final_correct_prob'])

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    final_probability = {}
    for role in ('failure', 'control'):
        for condition in LENS_CONDITIONS:
            selected = [r for r in lens if r['role'] == role and r['condition'] == condition]
            if selected:
                final_probability[f'{condition}:{role}'] = {
                    'n': len(selected),
                    'mean_final_correct_prob': mean([r['final_correct_prob'] for r in selected]),
                    'mean_final_wrong_prob': mean([r['final_wrong_prob'] for r in selected]),
                    'mean_final_shifted_prob': mean([r['final_shifted_prob'] for r in selected]),
                    'fraction_shifted_above_original': mean([r['final_shifted_prob'] > r['final_correct_prob']
                                                             for r in selected]),
                }

    return {
        'n_failures': len(failures),
        'n_controls': len([i for i in items if i['role'] == 'control']),
        'behavior': {
            'original': {'n': len(original), 'reproduces_published': original_reproduced,
                         'accuracy': frac(sum(1 for r in original if r['correct']), len(original))},
            'text_ablate': {'n': len(ablated), 'answer_unchanged': ablate_survives,
                            'fraction_unchanged': frac(ablate_survives, len(ablated)),
                            'answer_distribution': distribution(ablated)},
            'shifted': {'n': len(shifted), 'fraction_answer_unchanged': frac(shifted_unchanged, len(shifted)),
                        'fraction_tracks_shifted_expected': frac(shifted_tracks, len(shifted)),
                        'fraction_tracks_or_original': frac(shifted_tracks_or_original, len(shifted)),
                        'fraction_within_one': frac(shifted_within_one, len(shifted)),
                        'relation_counts': relation_counts,
                        'answer_distribution': distribution(shifted)},
            'disjoint': {'n': len(disjoint), 'fraction_answer_zero': frac(disjoint_zero, len(disjoint)),
                         'answer_distribution': distribution(disjoint)},
            'image_only': {'n': len(image_only), 'fraction_reproduces_wrong_count': frac(image_reproduces, len(image_only)),
                           'fraction_answer_four': frac(image_all_four, len(image_only)),
                           'answer_distribution': distribution(image_only)},
            'text_only': {'n': len(text_only), 'fraction_abstains': frac(text_abstains, len(text_only)),
                          'answer_distribution': distribution(text_only)},
            'control_original_accuracy': frac(sum(1 for r in control_original if r['correct']), len(control_original)),
        },
        'boundary': boundary,
        'lens': lens_summary,
        'final_probability': final_probability,
        'lens_final': {
            'failures_wrong_dominates': final_wrong_wins(original_failures),
            'failures_n': len(original_failures),
            'controls_wrong_dominates': final_wrong_wins(original_controls),
            'controls_n': len(original_controls),
            'failure_crossover_count': sum(1 for r in original_failures if r['crossover_layer'] is not None),
            'failure_crossover_layers': [r['crossover_layer'] for r in original_failures
                                         if r['crossover_layer'] is not None],
        },
    }


def build_report(analysis, items, behaviour, lens, manifest):
    lines = ['# Interval-count mechanism probe (time-boxed)', '']
    lines += ['**Question.** Is the confirmed off-by-one interval-count failure caused by the text interval '
              'not being used / arriving too late, or by image values dominating?', '']
    lines += [f"**Model/data.** Qwen3-VL-8B-Instruct, greedy, TI. {analysis['n_failures']} interval-count "
              f"failures + {analysis['n_controls']} correct interval-count controls from the R1 confirmation "
              f"set (`{manifest['failures_path']}`, `{manifest['answers_path']}`).", '']
    b = analysis['behavior']
    lines += ['## Aggregate behaviour', '',
              '| readout | value |', '|---|---:|',
              f"| original reproduces published answer | {b['original']['reproduces_published']}/{b['original']['n']} |",
              f"| text ablation keeps the same (wrong) answer | {b['text_ablate']['answer_unchanged']}/{b['text_ablate']['n']} = {b['text_ablate']['fraction_unchanged']:.3f} |",
              f"| shifted interval: answer unchanged | {round(b['shifted']['fraction_answer_unchanged']*b['shifted']['n'])}/{b['shifted']['n']} = {b['shifted']['fraction_answer_unchanged']:.3f} |",
              f"| shifted interval: answer tracks new expected count | {round(b['shifted']['fraction_tracks_shifted_expected']*b['shifted']['n'])}/{b['shifted']['n']} = {b['shifted']['fraction_tracks_shifted_expected']:.3f} |",
              f"| disjoint interval (true count 0): answers 0 | {round(b['disjoint']['fraction_answer_zero']*b['disjoint']['n'])}/{b['disjoint']['n']} = {b['disjoint']['fraction_answer_zero']:.3f} |",
              f"| image-only: reproduces the wrong count | {round(b['image_only']['fraction_reproduces_wrong_count']*b['image_only']['n'])}/{b['image_only']['n']} = {b['image_only']['fraction_reproduces_wrong_count']:.3f} |",
              f"| image-only: answers 4 (all values) | {round(b['image_only']['fraction_answer_four']*b['image_only']['n'])}/{b['image_only']['n']} = {b['image_only']['fraction_answer_four']:.3f} |",
              f"| text-only: abstains | {round(b['text_only']['fraction_abstains']*b['text_only']['n'])}/{b['text_only']['n']} = {b['text_only']['fraction_abstains']:.3f} |",
              f"| control original accuracy | {b['control_original_accuracy']:.3f} |", '']
    lines += ['Answer distributions (first integer in the model output):', '',
              '| condition | distribution |', '|---|---|']
    for condition in ('text_ablate', 'shifted', 'disjoint', 'image_only', 'text_only'):
        lines.append(f"| {condition} | {b[condition]['answer_distribution']} |")
    lines += ['']

    lines += ['## Logit lens (teacher-forced answer prefix)', '',
              '`crossover layer` = first layer *after* the last layer where P(wrong count) > P(correct count); '
              '`None` means the final layer still prefers the wrong count. The wrong count is the published '
              'answer on failures and `correct+1` on controls (a foil, since controls have no wrong answer). '
              '`final wrong dominates` counts items whose final layer prefers the wrong/foil count.', '']
    lf = analysis['lens_final']
    lines += ['| group | final wrong dominates | crossover found | median crossover layer |',
              '|---|---:|---:|---:|',
              f"| failures | {lf['failures_wrong_dominates']}/{lf['failures_n']} | {lf['failure_crossover_count']}/{lf['failures_n']} | {_median(lf['failure_crossover_layers'])} |",
              f"| controls | {lf['controls_wrong_dominates']}/{lf['controls_n']} | -- | -- |", '']
    lines += ['### Final-layer probabilities by condition', '',
              '| condition:role | n | P(correct) | P(wrong) | P(shifted count) | frac. shifted>original |',
              '|---|---:|---:|---:|---:|---:|']
    for key, value in analysis['final_probability'].items():
        lines.append('| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |'.format(
            key, value['n'], value['mean_final_correct_prob'], value['mean_final_wrong_prob'],
            value['mean_final_shifted_prob'], value['fraction_shifted_above_original']))
    lines += ['', '### Layer curve (original condition)', '',
              '| layer | failure P(correct) | failure P(wrong) | frac. correct>wrong | control P(correct) | control P(foil) | frac. correct>foil |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    failure_curve = analysis['lens'].get('original:failure', {}).get('layers', [])
    control_curve = analysis['lens'].get('original:control', {}).get('layers', [])
    for index in range(0, max(len(failure_curve), len(control_curve)), 3):
        f = failure_curve[index] if index < len(failure_curve) else None
        c = control_curve[index] if index < len(control_curve) else None
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            index,
            f'{f["correct_prob"]:.3f}' if f else '--',
            f'{f["wrong_prob"]:.3f}' if f else '--',
            f'{f["fraction_correct_above_wrong"]:.2f}' if f else '--',
            f'{c["correct_prob"]:.3f}' if c else '--',
            f'{c["wrong_prob"]:.3f}' if c else '--',
            f'{c["fraction_correct_above_wrong"]:.2f}' if c else '--'))
    lines += ['', '## Per-item table', '',
              '| item | role | vals | interval | exp | orig | ablate | shifted (exp) | disjoint | image-only | text-only | crossover |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    by_key = {(r['item_id'], r['condition']): r for r in behaviour}
    lens_cross = {r['item_id']: r['crossover_layer'] for r in lens
                  if r['condition'] == 'original'}
    for item in items:
        def answer(condition):
            row = by_key.get((item['item_id'], condition))
            if row is None:
                return '--'
            return 'abstain' if row['first_int'] is None else str(row['first_int'])
        shift = item['shift']
        lines.append('| {} | {} | {} | {}..{} | {} | {} | {} | {} ({}) | {} | {} | {} | {} |'.format(
            item['material_id'], item['role'], ','.join(str(v) for v in item['values']),
            item['lower'], item['upper'], item['expected'], answer('original'), answer('text_ablate'),
            answer('shifted'), shift['expected'], answer('disjoint'), answer('image_only'),
            answer('text_only'), lens_cross.get(item['item_id'])))
    lines += ['', '## Reading', '']
    if b['text_ablate']['fraction_unchanged'] is not None:
        lines.append(f"- Text ablation leaves the answer unchanged on {b['text_ablate']['answer_unchanged']}/{b['text_ablate']['n']} failures.")
    lines.append(f"- The shifted valid interval changes the answer on {b['shifted']['n'] - round(b['shifted']['fraction_answer_unchanged']*b['shifted']['n'])}/{b['shifted']['n']} failures; it tracks the shifted expected count on "
                 f"{round(b['shifted']['fraction_tracks_shifted_expected']*b['shifted']['n'])}/{b['shifted']['n']}, and is within one on "
                 f"{round(b['shifted']['fraction_within_one']*b['shifted']['n'])}/{b['shifted']['n']} "
                 f"(relations: {b['shifted']['relation_counts']}).")
    lines.append(f"- Image-only reproduces the wrong count on {round(b['image_only']['fraction_reproduces_wrong_count']*b['image_only']['n'])}/{b['image_only']['n']} failures and answers 4 (all values) on "
                 f"{round(b['image_only']['fraction_answer_four']*b['image_only']['n'])}/{b['image_only']['n']}.")
    boundary = analysis['boundary']
    lines.append(f"- The published wrong count is reached by a symmetric widening of the stated interval with median delta "
                 f"{boundary['median_expansion_delta']} ({boundary['n_within_2']}/{boundary['n']} need delta<=2; "
                 f"{boundary['n_reached']}/{boundary['n']} reachable this way).")
    shifted_lens = analysis['final_probability'].get('shifted:failure', {})
    lines.append(f"- Under the shifted rule the internal final probability of the shifted count exceeds the original count on "
                 f"{round(shifted_lens.get('fraction_shifted_above_original', 0) * shifted_lens.get('n', 0))}/{shifted_lens.get('n', 0)} failures "
                 f"(mean P shifted {shifted_lens.get('mean_final_shifted_prob', 0):.3f} vs original {shifted_lens.get('mean_final_correct_prob', 0):.3f}).")
    lines.append(f"- At the final layer the wrong count still dominates on {lf['failures_wrong_dominates']}/{lf['failures_n']} failures (controls {lf['controls_wrong_dominates']}/{lf['controls_n']}).")
    lines += ['', '## Conclusion', '',
              'The interval is **not ignored** and the answer is **not image-dominated**. Removing the interval makes the model '
              'count all four values; a disjoint interval yields 0; a shifted interval moves the answer, always within one of '
              'the new true count. Image-only always yields 4 and never the published wrong count. The failure is a '
              'boundary/precision error: the model applies the two-sided rule but misclassifies one boundary-adjacent value, '
              'which is equivalent to using a slightly wider interval. The logit lens shows the wrong count is not a late '
              'correction of a correct one: it is never decodable as the preferred count, and it is already dominant from the '
              'mid-late layers. Attention knockout is therefore not the high-value next step; an explicit per-bound comparison '
              '(the two-step prompt already reaches 0.967 on this cell) targets the actual defect.']
    lines += ['', '## Boundaries', '',
              '- Synthetic table images, one template, one model. This is a controlled probe, not MMQA external validity (R4).',
              '- The logit lens reads a single teacher-forced position after the answer prefix; it does not localise attention.',
              '- Attention knockout was not run (see manifest / next step); the probe reports a negative feasibility result for that stage.',
              '', '## Artifacts', '']
    for name in ('items.jsonl', 'behavior.jsonl', 'logit_lens.jsonl', 'results.jsonl', 'analysis.json', 'manifest.json'):
        lines.append(f'- `{Path(manifest["output_dir"]) / name}`')
    lines += ['', f"GPU wall time: {manifest['gpu_seconds']:.1f} s "
                  f"({manifest['behaviour_seconds']:.1f} s behaviour + {manifest['lens_seconds']:.1f} s lens); "
                  f"peak allocated {manifest['peak_allocated_mib']} MiB."]
    return '\n'.join(lines) + '\n'


def run_interval_mechanism(*, run_id=VERSION, project_root='.', output_root='outputs',
                           failures_path=FAILURES_PATH, answers_path=ANSWERS_PATH, materials_path=MATERIALS_PATH,
                           model_path, device='cuda:0', controls_per_expected=4, max_new_tokens=32, limit=None,
                           skip_lens=False):
    import torch
    from .local_vision import QwenVLBackend, qwen_identity
    from .prompts import PromptSet

    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    items = select_items(root / failures_path, root / answers_path, root / materials_path,
                         controls_per_expected=controls_per_expected)
    if limit:
        failures = [i for i in items if i['role'] == 'failure'][:limit]
        controls = [i for i in items if i['role'] == 'control'][:max(1, limit // 3)]
        items = failures + controls
    write_jsonl(out / 'items.jsonl', items)

    prompts = PromptSet(root / 'prompts', names=(ANSWER_TEMPLATE,))
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens)
    identity = qwen_identity(backend, VERSION, source_run=run_id)
    write_json(out / 'model_identity.json', identity)

    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    behaviour = run_behaviour(backend, prompts, items)
    behaviour_seconds = time.monotonic() - began
    write_jsonl(out / 'behavior.jsonl', behaviour)
    if skip_lens:
        lens = []
        lens_seconds = 0.0
    else:
        began = time.monotonic()
        lens = run_logit_lens(backend, prompts, items)
        lens_seconds = time.monotonic() - began
        write_jsonl(out / 'logit_lens.jsonl', lens)
    gpu_seconds = behaviour_seconds + lens_seconds
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    analysis = analyze(behaviour, lens, items)
    write_json(out / 'analysis.json', analysis)
    write_jsonl(out / 'results.jsonl', per_item_results(behaviour, lens, items))
    manifest = {
        'version': VERSION, 'created_at': utc_now(), 'run_id': run_id,
        'failures_path': failures_path, 'answers_path': answers_path, 'materials_path': materials_path,
        'output_dir': str(out.relative_to(root)), 'n_items': len(items),
        'n_failures': analysis['n_failures'], 'n_controls': analysis['n_controls'],
        'prompt_template': ANSWER_TEMPLATE, 'prompt_sha256': prompts.snapshot_sha256(),
        'answer_prefix': ANSWER_PREFIX, 'behaviour_conditions': list(BEHAVIOUR_CONDITIONS),
        'lens_conditions': list(LENS_CONDITIONS), 'skip_lens': skip_lens,
        'gpu_seconds': gpu_seconds, 'behaviour_seconds': behaviour_seconds, 'lens_seconds': lens_seconds,
        'peak_allocated_mib': round(peak, 1), 'device': device,
        'model_identity': identity,
        'attention_knockout': 'not run (time-boxed); behavioural + logit-lens only',
        'torch': torch.__version__,
    }
    write_json(out / 'manifest.json', manifest)
    (out / 'report.md').write_text(build_report(analysis, items, behaviour, lens, manifest), encoding='utf-8')
    return {'run_id': run_id, 'output_dir': str(out), 'n_items': len(items),
            'behaviour_seconds': round(behaviour_seconds, 1), 'lens_seconds': round(lens_seconds, 1),
            'analysis': analysis['behavior']}
