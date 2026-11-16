"""P1 candidate dataset: stratified material selection, seeded sampled candidates with captured hidden states, and the bridge to
the existing baseline evaluation (candidates become pseudo-materials of a simple-baseline source run).

Generation of the questions happens once, in `run_candidates`. `candidate-source-*` then replays those exact questions through the
baseline machinery (so records, answering and the cascade evaluator are unchanged) and lets the local model answer them.
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .records import read_jsonl, sha256_file, sha256_text, utc_now, write_json, write_jsonl
from .prompts import PromptSet, prompt_path

VERSION = 'candidates-v1'
TEMPLATE = 'plain-v2'
SEED = 20261113


# ---------- pure helpers ----------

def candidate_seed(run_seed, material_id, index):
    """Deterministic 31-bit seed per (run seed, material, candidate index); independent of batch composition and order."""
    digest = hashlib.sha256(f'{run_seed}|{material_id}|{index}'.encode()).digest()
    return int.from_bytes(digest[:4], 'big') % 2 ** 31


def cjk_share(text):
    """Share of CJK characters among letters/ideographs; used to record the question language."""
    chars = [c for c in text if c.isalpha()]
    return sum('一' <= c <= '鿿' for c in chars) / len(chars) if chars else 0.0


def select_materials(skill_of, *, seed, n_skills, per_skill, min_per_skill=6):
    """Stratified, nested selection: eligible skills are shuffled once, materials inside a skill are shuffled once, and
    a smaller (n_skills, per_skill) request is always a subset of a larger one. Returns [(skill, material_id)]."""
    by = defaultdict(list)
    for mid, skill in skill_of.items():
        by[skill].append(mid)
    eligible = sorted(s for s, ms in by.items() if len(ms) >= min_per_skill)
    order = random.Random(seed).sample(eligible, len(eligible))
    picked = []
    for skill in order[:n_skills]:
        ms = sorted(by[skill])
        random.Random(f'{seed}:{skill}').shuffle(ms)
        picked += [(skill, m) for m in ms[:per_skill]]
    return picked


def unique_candidates(rows):
    """One pseudo-material per (material, question text). Returns (uniques, multiplicity); empty questions are kept as failures."""
    seen, uniques = {}, []
    for r in rows:
        key = (r['material_id'], r['question'].strip())
        if key not in seen:
            seen[key] = {'pseudo_id': f"{r['material_id']}~{sha256_text(key[1])[:10]}", 'material_id': r['material_id'],
                         'question': key[1], 'candidate_ids': []}
            uniques.append(seen[key])
        seen[key]['candidate_ids'].append(r['candidate_id'])
    return uniques


class ReplayProvider:
    """Answers `generate_question` with a stored question and delegates every other task to the local provider."""

    def __init__(self, question, local):
        self.question, self.local = question, local
        self.model, self.identity = local.model, local.identity

    def complete(self, request):
        if request.task == 'generate_question':
            from .providers import Completion
            return Completion(self.question, self.model, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
                                                          'network_requests': 0, 'local_requests': 0, 'cache_hit': False,
                                                          'cost': None, 'replayed': True}, 'stop')
        return self.local.complete(request)


# ---------- capture of sampled candidates (GPU) ----------

def skills_of(root, materials):
    revision = materials[0]['dataset_revision']
    problems = json.loads(Path(glob.glob(str(root / 'data/raw/scienceqa' / revision / 'problems.json'))[0]).read_text())
    return {m['material_id']: problems[m['source_id']]['skill'] for m in materials}


def image_stats(path):
    from PIL import Image, ImageStat
    with Image.open(path) as im:
        gray = im.convert('L')
        return {'width': im.width, 'height': im.height, 'mean': round(ImageStat.Stat(gray).mean[0], 2), 'std': round(ImageStat.Stat(gray).stddev[0], 2)}


def load_materials_any(root, output_root, reference_run=None, materials_file=None):
    """Materials from a baseline run (ScienceQA, default) or from any JSONL following the material contract."""
    if materials_file:
        return {m['material_id']: m for m in read_jsonl(root / materials_file)}
    return {m['material_id']: m for m in read_jsonl(Path(output_root).resolve() / reference_run / 'materials.jsonl')}


def choose_materials(materials, *, materials_file, group_field, select, max_materials, n_skills, per_skill, run_seed, root):
    """[(group, material_id)]. ScienceQA keeps the stratified nested selection; a materials file is taken as is (optionally subsampled)."""
    if not materials_file:
        return select_materials(skills_of(root, list(materials.values())), seed=run_seed, n_skills=n_skills, per_skill=per_skill)
    ids = sorted(materials)
    if select == 'sample':
        ids = random.Random(run_seed).sample(ids, min(max_materials or len(ids), len(ids)))
    elif max_materials:
        ids = ids[:max_materials]
    return [(str(materials[m].get(group_field) or 'all'), m) for m in ids]


def run_candidates(*, run_id, project_root='.', output_root='outputs', reference_run=None, model_path, device='cuda:0', n_skills=0, per_skill=0,
                   n_candidates=8, temperature=0.7, top_p=0.8, top_k=20, k=4, max_new_tokens=1024, exclude_run=None, run_seed=SEED, template=TEMPLATE,
                   materials_file=None, group_field='group', select='all', max_materials=None, intervention=None):
    from safetensors.torch import save_file
    from .local_vision import QwenVLBackend, identity_matches, qwen_identity
    from .prompts import PromptSet
    from .providers import CompletionRequest
    from .whitebox import Capturer, gpu_memory_mib
    root, out = Path(project_root).resolve(), Path(output_root).resolve() / run_id
    materials = load_materials_any(root, output_root, reference_run, materials_file)
    selection = choose_materials(materials, materials_file=materials_file, group_field=group_field, select=select, max_materials=max_materials,
                                 n_skills=n_skills, per_skill=per_skill, run_seed=run_seed, root=root)
    if exclude_run:
        skip = {r['material_id'] for r in read_jsonl(Path(output_root).resolve() / exclude_run / 'captures.jsonl')}
        selection = [(s, m) for s, m in selection if m not in skip]
    (out / 'states').mkdir(parents=True, exist_ok=True)
    prompts = PromptSet(root / 'prompts', names=(template,))
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens)
    sampling = {'temperature': temperature, 'top_p': top_p, 'top_k': top_k}
    identity = qwen_identity(backend, 'local-qwen3-vl-whitebox-sampled-v1', capture_version=VERSION, batch_size=1, capture_k=k,
                             prompt_template=template, prompt_sha256=sha256_file(prompt_path(root / 'prompts', template)),
                             state_dtype='bfloat16', sampling=sampling, n_candidates=n_candidates, run_seed=run_seed,
                             selection={'n_skills': n_skills, 'per_skill': per_skill, 'exclude_run': exclude_run, 'materials_file': materials_file,
                                        'group_field': group_field, 'select': select, 'max_materials': max_materials},
                             intervention=(intervention or {}).get('description'))
    identity['do_sample'] = True
    stored_path = out / 'model_identity.json'
    if stored_path.exists():
        stored = json.loads(stored_path.read_text())
        if not identity_matches(stored, identity):
            raise ValueError('settings changed; use a new run_id')
        if stored != identity:
            relocation = {'at': utc_now(), 'old_path': (stored.get('model_fingerprint') or {}).get('path'),
                          'new_path': (identity.get('model_fingerprint') or {}).get('path'),
                          'verified': 'all identity fields except model_fingerprint.path are unchanged'}
            with (out / 'model_identity_relocations.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(relocation, ensure_ascii=False, sort_keys=True) + '\n')
    write_json(out / 'model_identity.json', identity)
    write_json(out / 'selection.json', [{'skill': s, 'material_id': m} for s, m in selection])
    done = {r['candidate_id'] for r in read_jsonl(out / 'captures.jsonl')}
    capturer = Capturer(backend, k=k)
    backend._load()
    for sk, mid in selection:
        m = materials[mid]
        request = CompletionRequest(task='generate_question', prompt=prompts.render(template, T=m['text']), image_path=str(root / m['image_path']))
        prepared = None
        for j in range(n_candidates):
            cid = f'{mid}~c{j}'
            if cid in done:
                continue
            if prepared is None:
                prepared = capturer.prepare(request, m['text'])
            seed = candidate_seed(run_seed, mid, j)
            states, meta = capturer.capture(request, m['text'], prepared=prepared, sample={**sampling, 'seed': seed},
                                            intervention=(intervention or {}).get('factory'))
            save_file(states, str(out / 'states' / f'{cid}.safetensors'))
            row = {'candidate_id': cid, 'material_id': mid, 'index': j, 'seed': seed, 'skill': sk, 'created_at': utc_now(),
                   'states_file': f'states/{cid}.safetensors', 'text_chars': len(m['text']), 'n_choices': m.get('n_text_choices'), 'parent_material_id': m.get('parent_material_id'),
                   'template': template, 'image': image_stats(root / m['image_path']), 'zh_share': round(cjk_share(meta['question']), 3), **meta}
            with (out / 'captures.jsonl').open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    rows = read_jsonl(out / 'captures.jsonl')
    summary = {'run_id': run_id, 'materials': len(selection), 'candidates': len(rows), 'unique_questions': len(unique_candidates(rows)),
               'complete': sum(r['plan']['complete'] for r in rows), 'empty_questions': sum(not r['question'].strip() for r in rows),
               'gpu_used_mib_end': gpu_memory_mib(), 'peak_allocated_mib_max': max((r['peak_allocated_mib'] for r in rows), default=None)}
    write_json(out / 'summary.json', summary)
    return summary


# ---------- bridge to the baseline machinery ----------

def init_candidate_source(*, run_id, capture_run, project_root='.', output_root='outputs', reference_run=None, materials_file=None):
    """Freeze unique candidates as a simple-baseline source run (pseudo-material = material x unique question)."""
    from .simple_baseline import init_simple_baseline
    root = Path(project_root).resolve()
    cap = Path(output_root).resolve() / capture_run
    rows = read_jsonl(cap / 'captures.jsonl')
    materials = load_materials_any(root, output_root, reference_run, materials_file)
    uniques = unique_candidates(rows)
    pseudo = [{**materials[u['material_id']], 'material_id': u['pseudo_id'], 'parent_material_id': u['material_id']} for u in uniques]
    path = cap / f'source_materials_{run_id}.jsonl'
    write_jsonl(path, pseudo)
    write_json(cap / f'source_map_{run_id}.json', {u['pseudo_id']: {'material_id': u['material_id'], 'question': u['question'],
                                                                     'candidate_ids': u['candidate_ids']} for u in uniques})
    identity_path = cap / 'model_identity.json'
    result = init_simple_baseline(run_id=run_id, materials_path=str(path.relative_to(root)), project_root=str(root),
                                  output_root=output_root, model='Qwen3-VL-8B-Instruct-local', concurrency=16)
    if identity_path.exists():       # the replayed questions were generated with this prompt/model; the source run's own manifest only names the answering prompts
        write_json(root / output_root / run_id / 'generation_identity.json', json.loads(identity_path.read_text()))
    return {**result, 'candidates': len(rows), 'unique': len(uniques)}


def run_candidate_source(*, run_id, capture_run, project_root='.', output_root='outputs', model_path, device='cuda:0', max_batch=8, max_new_tokens=1024):
    from .local_vision import local_provider
    from .records import stable_id
    from .simple_baseline import run_simple_baseline
    cap = Path(output_root).resolve() / capture_run
    table = json.loads((cap / f'source_map_{run_id}.json').read_text())
    question_of = {stable_id('attempt', run_id, pid): v['question'] for pid, v in table.items()}
    local = local_provider(model_path, device=device, max_batch=max_batch, max_new_tokens=max_new_tokens)
    factory = lambda item, tokens: ReplayProvider(question_of[Path(item).name], local)
    return run_simple_baseline(run_id=run_id, project_root=project_root, output_root=output_root, provider_factory=factory)


def api_ledger(*, patterns, project_root='.', output_root='outputs'):
    """Calls and tokens spent by the evaluation runs whose names contain any of `patterns` (the probing route's own budget)."""
    base = Path(project_root).resolve() / output_root
    rows = []
    for status in sorted(base.glob('*/status.json')):
        run = status.parent.name
        if not any(p in run for p in patterns) or not run.endswith(('-eval-v1', '-eval-v2', '-eval')):
            continue
        data = json.loads(status.read_text())
        if 'new_http_requests' not in data:
            continue
        rows.append({'run': run, 'calls': data['new_http_requests'], 'tokens_known': (data.get('usage') or {}).get('known_total_tokens') or 0})
    return {'runs': rows, 'calls_total': sum(r['calls'] for r in rows), 'tokens_known_total': sum(r['tokens_known'] for r in rows),
            'note': 'token usage is incomplete for some calls, so the token total is a lower bound'}
