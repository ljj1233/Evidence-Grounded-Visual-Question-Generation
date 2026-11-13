"""Ensemble necessity evaluation (answer identifiability, see docs/necessity_formal_argument.md section 9).

Two model groups ('a', 'b') screen single-modality sufficiency and recompute the witnesses proposed by the source
evaluation; an arbiter ('arb') only works on items the two did not settle; code, not a model, makes every decision.
Answer equivalence is equality of normalised values over answer slots derived from the question alone.
Undecided stays undecided: it is never counted as a pass and never converted into the opposite label.
"""
from __future__ import annotations

import collections
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .pipeline import _validate_run_id
from .prompts import PromptSet
from .providers import CompletionRequest, JudgeProvider
from .records import canonical_json, file_lock, read_jsonl, sha256_file, sha256_text, utc_now, write_json

TEMPLATES = ('ensemble-sufficiency-v1', 'ensemble-witness-v1', 'answer-slots-schema-v1', 'answer-slots-extract-v1')
GROUPS = ('a', 'b')
ARBITER = 'arb'
NO_IMAGE_TEXT = '材料仅为附图，没有任何文本材料。'


# ----------------------------------------------------------------------------- pure decision logic

def parse_json(text):
    match = re.search(r'\{.*\}', text or '', re.S)
    return json.loads(match.group(0)) if match else None


def norm_value(value):
    """Canonical form of one slot value: lowercase, punctuation-free, singular, fractions as percentages."""
    if value is None:
        return None
    text = re.sub(r'[\s\-_]+', ' ', re.sub(r'[^\w\s/.%-]', '', str(value).strip().lower())).replace('%', '').strip()
    fraction = re.fullmatch(r'(\d+)\s*/\s*(\d+)', text)
    if fraction and float(fraction.group(2)):
        text = str(float(fraction.group(1)) / float(fraction.group(2)) * 100)
    try:
        return f'{float(text):g}'
    except ValueError:
        return ' '.join(w[:-1] if len(w) > 3 and w.endswith('s') else w for w in text.split())


def vote(values):
    """Two groups must agree; with the arbiter present, two of three. Undecided (None) never counts as support."""
    need = len(values) if len(values) == 2 else 2
    yes, no = sum(v is True for v in values), sum(v is False for v in values)
    return True if yes >= need else False if no >= need else None


def key(role, kind, attempt, mod=None):
    return '|'.join(x for x in (role, kind, mod, attempt) if x)


class Evaluation:
    """Every decision as a function of recorded model outputs. `records` maps 'role|kind|mod|attempt' to the stored
    record ({'parsed': ..., 'strings': [...]}); slot extractions carry the exact candidate list they were asked about."""

    def __init__(self, items, records, reference='baseline'):
        self.reference_mode = reference      # 'baseline': the evaluated model's own answer; 'independent': both groups answer the question themselves
        self.items = [i for i in items if i['validity'] == 1 and i['question'] and (reference == 'independent' or i['reference'])]
        self.records = records
        self.slots, self.failures = {}, collections.Counter()
        for i in self.items:
            a = i['attempt_id']
            for role in GROUPS + (ARBITER,):
                record = self.records.get(key(role, 'SLOT', a))
                if record is None:
                    if role != ARBITER and self.p(role, 'S', a, 'T') is not None:
                        self.failures[role] += 1
                    continue
                data, strings, table = record.get('parsed'), record.get('strings') or [], {}
                if isinstance(data, dict) and isinstance(data.get('items'), list):
                    for entry in data['items']:
                        try:
                            table[strings[int(entry['i'])]] = {k: norm_value(v) for k, v in (entry.get('values') or {}).items()}
                        except (KeyError, ValueError, IndexError, TypeError, AttributeError):
                            pass
                else:
                    self.failures[role] += 1
                self.slots[(a, role)] = table
        self.refs = {i['attempt_id']: self._independent_reference(i) for i in self.items} if reference == 'independent' else {}
        self.arbitrated = {i['attempt_id'] for i in self.items if self.p(ARBITER, 'S', i['attempt_id'], 'T') is not None}
        self.results = {i['attempt_id']: self.decide(i) for i in self.items}

    def p(self, role, kind, attempt, mod=None):
        return (self.records.get(key(role, kind, attempt, mod)) or {}).get('parsed')

    def own_reference_answers(self, i):
        out = []
        for role in GROUPS:
            r = self.p(role, 'REF', i['attempt_id']) or {}
            out.append(r['answer'] if r.get('determinable') and r.get('answer') else None)
        return out

    def _independent_reference(self, i):
        ra, rb = self.own_reference_answers(i)
        return ra if ra and rb and self.same(i['attempt_id'], ra, rb) is True else None

    def reference(self, i):
        return self.refs.get(i['attempt_id']) if self.reference_mode == 'independent' else i['reference']

    def candidate_strings(self, i, roles):
        """Reference answer(s) plus every complete answer the given roles produced for this item (stable order)."""
        a = i['attempt_id']
        found = {i['reference']} if self.reference_mode == 'baseline' else {x for x in self.own_reference_answers(i) if x}
        for role in roles:
            for mod in ('T', 'I'):
                s = self.p(role, 'S', a, mod) or {}
                if s.get('determinable') and s.get('answer'):
                    found.add(s['answer'])
                for x in (self.p(role, 'W', a, mod) or {}).get('s') or []:
                    if isinstance(x, dict) and x.get('answer'):
                        found.add(x['answer'])
        return sorted(found)

    def same(self, a, x, y):
        """True/False/None. Each extractor that saw both strings votes; a slot missing in either answer is no vote."""
        votes = []
        for role in GROUPS + (ARBITER,):
            table = self.slots.get((a, role))
            if not table or x not in table or y not in table:
                continue
            names = set(table[x]) | set(table[y])
            if not names or any(table[x].get(n) is None or table[y].get(n) is None for n in names):
                continue
            votes.append(all(table[x][n] == table[y][n] for n in names))
        if not votes:
            return None
        if len(votes) == 1:
            return votes[0]
        yes = sum(votes)
        return True if yes > len(votes) - yes else False if yes < len(votes) - yes else None

    def sufficient(self, role, mod, i):
        """Does the retained modality `mod` alone give the full answer? True / False (not shown) / None (undecided)."""
        s = self.p(role, 'S', i['attempt_id'], mod)
        if s is None:
            return None
        if not (s.get('determinable') and s.get('answer')):
            return False
        ref = self.reference(i)
        return None if ref is None else self.same(i['attempt_id'], s['answer'], ref)

    def witness(self, role, removed, i):
        """Does the recomputed witness pair show that the answer changes when `removed` is replaced? 'nowit' = none proposed."""
        a = i['attempt_id']
        if not i['witness_text_removed' if removed == 'T' else 'witness_image_removed']:
            return 'nowit'
        w = self.p(role, 'W', a, removed)
        if w is None:
            return None
        s = w.get('s') or []
        if len(s) != 2 or not all(isinstance(x, dict) for x in s):
            return None
        if not all(x.get('presup_ok') and x.get('consistent') and x.get('answer') for x in s):
            return False
        pair = self.same(a, s[0]['answer'], s[1]['answer'])     # same normalised slot values = same answer = no witness
        return None if pair is None else not pair

    def decide(self, i, groups=None):
        a = i['attempt_id']
        if self.reference(i) is None:        # no usable reference answer: nothing can be compared
            return {'D_T': None, 'D_I': None, 'D_T_why': 'noref', 'D_I_why': 'noref', 'B': None, 'arbitrated': False}
        roles = groups or GROUPS + ((ARBITER,) if a in self.arbitrated else ())
        out = {}
        for name, single, removed in (('D_T', 'I', 'T'), ('D_I', 'T', 'I')):   # D_T: is the text needed
            suff = vote([self.sufficient(r, single, i) for r in roles])
            wits = [self.witness(r, removed, i) for r in roles]
            wit = False if all(w == 'nowit' for w in wits) else vote([None if w == 'nowit' else w for w in wits])
            out[name] = 0 if suff is True else 1 if (wit is True and suff is False) else None
            out[name + '_why'] = ('suff' if suff is True else 'wit' if out[name] == 1 else
                                  'nowit' if all(w == 'nowit' for w in wits) and suff is False else 'split')
        values = (out['D_T'], out['D_I'])
        out['B'] = 0 if 0 in values else 1 if values == (1, 1) else None
        out['arbitrated'] = a in self.arbitrated
        return out

    def undecided(self):
        """Items the two groups alone leave open: the arbiter's workload."""
        return [i for i in self.items if self.decide(i, GROUPS)['B'] is None]


def has_two_way_witness(i):
    """Only an item with a proposed witness in both directions can be confirmed dual."""
    return bool(i['witness_text_removed'] and i['witness_image_removed'])


# ----------------------------------------------------------------------------- prompts

def sufficiency_prompt(prompts, i, mod, knowledge):
    material = '<T>' + i['text'] + '</T>' if mod == 'T' else NO_IMAGE_TEXT
    return prompts.render('ensemble-sufficiency-v1', Q=i['question'], MATERIAL=material, KNOWLEDGE=knowledge)


def reference_prompt(prompts, i, knowledge):
    material = '<T>' + i['text'] + '</T>\n材料同时包含附图（见图）。'
    return prompts.render('ensemble-sufficiency-v1', Q=i['question'], MATERIAL=material, KNOWLEDGE=knowledge)


def witness_prompt(prompts, i, removed, knowledge):
    scenario = i['witness_text_removed'] if removed == 'T' else i['witness_image_removed']
    fixed = ('原图（见附图，保持不变）。文本被移除，下面两份是假想的替换文本。' if removed == 'T' else
             '原文本保持不变：<T>' + i['text'] + '</T>\n图像被移除，下面两份是对假想图像的文字描述（视作该图的实际内容）。')
    return prompts.render('ensemble-witness-v1', Q=i['question'], FIXED=fixed, KNOWLEDGE=knowledge,
                          SCENARIO_A=scenario['scenario_a'], SCENARIO_B=scenario['scenario_b'])


def extract_prompt(prompts, i, schema, strings):
    return prompts.render('answer-slots-extract-v1', Q=i['question'], SCHEMA=json.dumps(schema, ensure_ascii=False),
                          CANDIDATES='\n'.join(f'[{k}] {s}' for k, s in enumerate(strings)))


# ----------------------------------------------------------------------------- init / run / report

def init_ensemble(*, run_id, source_run, model_a, model_b, arbiter, project_root='.', output_root='outputs',
                  concurrency=32, max_http=16000, candidates_only=True, reference='independent'):
    """Freeze the item list, prompts and settings from a finished simple-baseline evaluation run."""
    root = Path(project_root).resolve()
    _validate_run_id(run_id)
    out, src = root / output_root / run_id, root / output_root / source_run
    knowledge = json.loads((src / 'manifest.json').read_text(encoding='utf-8'))['settings']['assessment']['knowledge']
    records = {r['attempt_id']: r for r in read_jsonl(src / 'records.jsonl')}
    items = []
    for s in read_jsonl(src / 'source_questions.jsonl'):
        r, m = records[s['attempt_id']], s['material']
        steps = r.get('outputs') or {}

        def witness(step):
            x = (steps.get(step) or {}).get('output')
            return x if isinstance(x, dict) and x.get('kind') == 'ambiguous' and x.get('scenario_a') and x.get('scenario_b') else None
        items.append({'attempt_id': s['attempt_id'], 'item_id': s['item_id'], 'material_id': r['material_id'], 'text': m['text'],
                      'image_path': m['image_path'], 'image_sha256': m['image_sha256'], 'question': r['question']['normalized_text'],
                      'reference': ((s.get('responses') or {}).get('TI') or {}).get('answer'), 'validity': r['question_answerability'],
                      'source_both_required': r['both_required'], 'witness_text_removed': witness('analysis_T'),
                      'witness_image_removed': witness('analysis_I')})
    for i in items:
        if i['validity'] == 1 and sha256_file(root / i['image_path']) != i['image_sha256']:
            raise ValueError('evaluation image changed: ' + i['item_id'])
    prompts = PromptSet(root / 'prompts', names=TEMPLATES)
    settings = {'models': {'a': model_a, 'b': model_b, ARBITER: arbiter}, 'concurrency': concurrency, 'max_http': max_http,
                'source_run': source_run, 'knowledge': knowledge, 'candidates_only': candidates_only, 'reference': reference}
    fingerprint = sha256_text(canonical_json({'items': items, 'prompts': prompts.snapshot(), 'settings': settings}))
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out / '.init.lock'):
        if (out / 'manifest.json').exists():
            if json.loads((out / 'manifest.json').read_text(encoding='utf-8'))['fingerprint'] != fingerprint:
                raise ValueError('frozen ensemble evaluation changed; use a new run_id')
            return {'n_slots': len(items), 'resumed': True}
        write_json(out / 'items.json', items)
        (out / 'prompts').mkdir(exist_ok=True)
        for name, text in prompts.snapshot().items():
            (out / 'prompts' / f'{name}.txt').write_text(text, encoding='utf-8')
        write_json(out / 'manifest.json', {
            'fingerprint': fingerprint, 'created_at': utc_now(), 'settings': settings, 'prompt_sha256': prompts.snapshot_sha256(),
            'n_slots': len(items), 'n_evaluated': sum(i['validity'] == 1 and bool(i['question'] and (reference == 'independent' or i['reference'])) and (not candidates_only or has_two_way_witness(i)) for i in items)})
    return {'n_slots': len(items), 'resumed': False}


class _Store:
    """Append-only results. Failed calls are logged but never cached, so a resume retries them."""

    def __init__(self, path):
        self.path, self.ok, self.http, self.lock = path, {}, 0, threading.Lock()
        for r in (read_jsonl(path) if path.exists() else []):
            self.http += r.get('attempts', 1)
            if not r.get('error'):
                self.ok[r['key']] = r

    def add(self, rec):
        with self.lock:
            if not rec.get('error'):
                self.ok[rec['key']] = rec
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _load(out):
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    prompts = PromptSet(out / 'prompts', names=TEMPLATES)
    if prompts.snapshot_sha256() != manifest['prompt_sha256']:
        raise ValueError('prompt changed since init')
    return manifest, prompts, json.loads((out / 'items.json').read_text(encoding='utf-8'))


def _default_factory(root, out):
    def make(model):
        provider = JudgeProvider.from_env(root=root, run_dir=out / 'requests', max_requests=10 ** 6, max_tokens=4000, visual=True, model=model)
        provider.max_attempts, provider.timeout = 1, 180
        return provider
    return make


def run_ensemble(*, run_id, phase='all', limit=None, project_root='.', output_root='outputs', provider_factory=None):
    """phase 'screen': sufficiency, witness recompute, slot schema and extraction by groups a and b.
    phase 'arbitrate': the arbiter redoes the same work for the items a and b left open. 'all' runs both."""
    root = Path(project_root).resolve()
    _validate_run_id(run_id)
    out = root / output_root / run_id
    manifest, prompts, items = _load(out)
    settings, knowledge = manifest['settings'], manifest['settings']['knowledge']
    store = _Store(out / 'results.jsonl')
    (out / 'requests').mkdir(exist_ok=True)
    factory = provider_factory or _default_factory(root, out)
    only = settings.get('candidates_only', False)
    ref_mode = settings.get('reference', 'baseline')
    evaluated = [i for i in items if i['validity'] == 1 and i['question'] and (ref_mode == 'independent' or i['reference']) and (not only or has_two_way_witness(i))]
    if limit:
        evaluated = evaluated[:limit]
    guard = threading.Lock()

    def ask(role, kind, attempt, prompt, mod=None, image=None, extra=None):
        k = key(role, kind, attempt, mod)
        if k in store.ok:
            return
        rec = {'key': k, **(extra or {})}
        for attempt_no in (1, 2):
            with guard:
                if store.http >= settings['max_http']:
                    raise RuntimeError('HTTP cap reached; use a new run_id with a larger max_http')
                store.http += 1
            try:
                completion = factory(settings['models'][role]).complete(CompletionRequest(
                    task='assess_quality', prompt=prompt, image_path=str(root / image) if image else None, condition=mod,
                    metadata={'role': role, 'kind': kind, 'key': k, 'try': str(attempt_no)}))
                rec.update(raw=completion.text, parsed=parse_json(completion.text), tokens=completion.usage.get('total_tokens'),
                           attempts=attempt_no)
                rec.pop('error', None)
                break
            except Exception as exc:    # provider, HTTP or JSON failure: recorded, retried once, never turned into a label
                rec.update(error=str(exc)[:200], parsed=None, attempts=attempt_no)
                time.sleep(0.5)
        store.add(rec)

    def run(tasks):
        with ThreadPoolExecutor(settings['concurrency']) as pool:
            list(pool.map(lambda t: ask(*t), tasks))

    def work(roles, subset, schema_role=None):
        tasks = []
        for i in subset:
            a = i['attempt_id']
            if schema_role:
                tasks.append((schema_role, 'SCHEMA', a, prompts.render('answer-slots-schema-v1', Q=i['question'])))
            for role in roles:
                if ref_mode == 'independent' and role in GROUPS:     # the evaluator answers the question itself, with the full material
                    tasks.append((role, 'REF', a, reference_prompt(prompts, i, knowledge), None, i['image_path']))
                tasks.append((role, 'S', a, sufficiency_prompt(prompts, i, 'T', knowledge), 'T', None))
                tasks.append((role, 'S', a, sufficiency_prompt(prompts, i, 'I', knowledge), 'I', i['image_path']))
                if i['witness_text_removed']:
                    tasks.append((role, 'W', a, witness_prompt(prompts, i, 'T', knowledge), 'T', i['image_path']))
                if i['witness_image_removed']:
                    tasks.append((role, 'W', a, witness_prompt(prompts, i, 'I', knowledge), 'I', None))
        run(tasks)
        snapshot = Evaluation(items, store.ok, ref_mode)
        tasks = []
        for i in subset:
            a = i['attempt_id']
            schema = (store.ok.get(key('a', 'SCHEMA', a)) or {}).get('parsed')
            if schema:
                strings = snapshot.candidate_strings(i, roles if ARBITER not in roles else GROUPS + (ARBITER,))
                prompt = extract_prompt(prompts, i, schema, strings)
                tasks += [(role, 'SLOT', a, prompt, None, None, {'strings': strings}) for role in roles]
        run(tasks)

    if phase in ('screen', 'all'):
        work(GROUPS, evaluated, schema_role='a')
    if phase in ('arbitrate', 'all'):
        wanted = {i['attempt_id'] for i in evaluated}
        ev = Evaluation(items, store.ok, ref_mode)
        work((ARBITER,), [i for i in ev.undecided() if i['attempt_id'] in wanted and ev.reference(i) is not None])
    return {'n_evaluated': len(evaluated), 'http_requests_total': store.http, 'results': len(store.ok)}


def report_ensemble(*, run_id, project_root='.', output_root='outputs'):
    """Fixed denominator over every slot. Two outcomes: confirmed dual / not confirmed; the split below is diagnostic."""
    root = Path(project_root).resolve()
    _validate_run_id(run_id)
    out = root / output_root / run_id
    manifest, _, items = _load(out)
    only = manifest['settings'].get('candidates_only', False)
    store = _Store(out / 'results.jsonl')
    ref_mode = manifest['settings'].get('reference', 'baseline')
    evaluation = Evaluation([i for i in items if not only or has_two_way_witness(i)], store.ok, ref_mode)
    screened = {i['attempt_id'] for i in evaluation.items if any(evaluation.p(r, 'S', i['attempt_id'], m) is not None for r in GROUPS for m in 'TI')}
    by_b = collections.Counter(r['B'] if i['attempt_id'] in screened or r['B'] is not None else 'pending'
                               for i in evaluation.items for r in [evaluation.results[i['attempt_id']]])
    invalid = sum(i['validity'] == 0 for i in items)
    validity_pending = sum(i['validity'] is None for i in items)
    usable = lambda i: bool(i['question'] and (ref_mode == 'independent' or i['reference']))
    no_input = sum(i['validity'] == 1 and not usable(i) for i in items)
    no_two_way = [i for i in items if only and i['validity'] == 1 and usable(i) and not has_two_way_witness(i)]
    cut_by_source = sum(i['source_both_required'] == 0 for i in no_two_way)
    tokens = collections.Counter()
    for r in store.ok.values():
        tokens[r['key'].split('|')[0]] += r.get('tokens') or 0
    why = collections.Counter()
    for i in evaluation.items:
        r = evaluation.results[i['attempt_id']]
        if r['B'] is None and i['attempt_id'] in screened:
            why['reference undecided (the two groups disagree or failed)' if 'noref' in (r['D_T_why'], r['D_I_why']) else
                'no witness in a direction' if 'nowit' in (r['D_T_why'], r['D_I_why']) else 'groups split or slots undecided'] += 1
    undecided = by_b[None]
    result = {'n_slots': len(items), 'confirmed_dual': by_b[1],
              'not_confirmed': invalid + by_b[0] + validity_pending + no_input + len(no_two_way) + undecided + by_b['pending'],
              'detail': {'invalid_question': invalid, 'single_modality_sufficient': by_b[0], 'validity_pending': validity_pending,
                         'no_question_or_reference': no_input, 'no_two_way_witness_not_run': len(no_two_way), 'of_which_source_already_found_single_modality_sufficient': cut_by_source, 'necessity_undecided': undecided, 'necessity_undecided_why': dict(why),
                         'not_yet_evaluated': by_b['pending']},
              'arbitrated_items': len(evaluation.arbitrated), 'slot_extraction_failures': dict(evaluation.failures),
              'tokens': dict(tokens), 'http_requests': store.http,
              'provenance': 'automatic and model-based; every confirmed item needs review before it is counted'}
    assert result['confirmed_dual'] + result['not_confirmed'] == len(items)
    write_json(out / 'results.json', result)
    write_json(out / 'final_items.json', {i['item_id']: evaluation.results[i['attempt_id']] for i in evaluation.items})
    return result
