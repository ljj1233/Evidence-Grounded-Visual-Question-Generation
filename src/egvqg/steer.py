"""2b / H3 (exploratory): steer generation along the text↔image axis saved by diag-curve.

Two intervention families at one decoder layer:
  clamp  h' = h + alpha * (b - <h, w>) * w        (alpha = 1 puts the projection exactly at b)
  push   h' = h + beta * sigma * w                (sigma = std of the training projections)
Positions: prompt_end (last prompt position), all_generated (every decode step), both.
Controls: a same-norm random unit vector orthogonal to w, and the normalized skill-mean direction.
Every condition reuses the same seed list per material, so candidates are paired. Output rows follow the
unified candidates.jsonl schema; labels are produced afterwards by `label-repair`.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import candidate_table as CT
from .candidates import candidate_seed
from .centering import projection
from .intervene import find_layers
from .records import write_json, write_jsonl


# ---------- pure math ----------

def steer_numpy(h, w, *, alpha=0.0, b=0.0, beta=0.0, sigma=1.0):
    """Reference implementation of both interventions on the last axis. w is a unit vector."""
    h = np.asarray(h, float)
    w = np.asarray(w, float)
    proj = h @ w
    return h + (alpha * (b - proj))[..., None] * w + beta * sigma * w


def orthogonal_random(w, seed):
    v = np.random.default_rng(seed).normal(size=len(w))
    v -= float(v @ w) * np.asarray(w, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def skill_mean_direction(X, skills):
    skills = np.asarray(skills)
    means = [X[skills == s].mean(0) for s in np.unique(skills)]
    v = np.mean(means, axis=0) - X.mean(0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def direction_info(X, w, material, mixed_materials):
    proj = projection(X, w)
    b = float(np.mean(proj[np.isin(material, list(mixed_materials))])) if mixed_materials else float(proj.mean())
    return {'b': b, 'sigma': float(proj.std()), 'n': int(len(proj))}


# ---------- torch hook ----------

@contextmanager
def steer_hook(model, *, layer, vector, alpha=0.0, b=0.0, beta=0.0, sigma=1.0, steer_pos='prompt_end'):
    """Apply the projection intervention to decoder layer `layer`; see module docstring for positions."""
    import torch
    layers = find_layers(model)
    v = torch.as_tensor(np.asarray(vector), dtype=torch.float32)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        h = hidden.clone()
        rows = []
        if h.shape[1] > 1 and steer_pos in ('prompt_end', 'both'):
            rows = [h.shape[1] - 1]
        elif h.shape[1] == 1 and steer_pos in ('all_generated', 'both'):
            rows = [0]
        if rows:
            vv = v.to(device=h.device, dtype=h.dtype)
            for r in rows:
                proj = (h[:, r, :] * vv).sum(-1, keepdim=True)
                h[:, r, :] = h[:, r, :] + alpha * (b - proj) * vv + beta * sigma * vv
        return (h, *output[1:]) if isinstance(output, tuple) else h
    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _mock_hook_check(dim=8, seed=0):
    """CPU check: alpha=1 sets the projection to b and leaves the orthogonal complement unchanged."""
    import torch
    from torch import nn

    class Identity(nn.Module):
        def forward(self, x):
            return x

    class Mock(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Identity()])

    w = orthogonal_random(np.ones(dim) / np.sqrt(dim), seed)
    h = torch.tensor(np.random.default_rng(seed + 1).normal(size=(2, 3, dim)), dtype=torch.float32)
    model = Mock()
    with steer_hook(model, layer=0, vector=w, alpha=1.0, b=0.37, steer_pos='prompt_end'):
        out = model.layers[0](h)
    proj = (out[0, -1].numpy() @ w)
    perp_before = h[0, -1].numpy() - (h[0, -1].numpy() @ w) * w
    perp_after = out[0, -1].numpy() - proj * w
    return {'projection_minus_b': float(proj - 0.37), 'orthogonal_max_abs_change': float(np.abs(perp_before - perp_after).max()),
            'ok': bool(abs(proj - 0.37) < 1e-4 and np.abs(perp_before - perp_after).max() < 1e-4)}


# ---------- generation ----------

def _conditions(alphas, betas, w, X_skill, seed):
    specs = [{'name': 'control', 'vector': w, 'alpha': 0.0, 'beta': 0.0}]
    specs += [{'name': f'clamp_a{a:g}', 'vector': w, 'alpha': a, 'beta': 0.0} for a in alphas if a > 0]
    specs += [{'name': f'push_b{b:+g}', 'vector': w, 'alpha': 0.0, 'beta': b} for b in betas if b != 0]
    rand = orthogonal_random(w, seed + 7)
    skill = skill_mean_direction(*X_skill)
    specs += [{'name': 'ctrl_random_clamp', 'vector': rand, 'alpha': 1.0, 'beta': 0.0},
              {'name': 'ctrl_random_push+2', 'vector': rand, 'alpha': 0.0, 'beta': 2.0},
              {'name': 'ctrl_random_push-2', 'vector': rand, 'alpha': 0.0, 'beta': -2.0},
              {'name': 'ctrl_skill_clamp', 'vector': skill, 'alpha': 1.0, 'beta': 0.0},
              {'name': 'ctrl_skill_push+2', 'vector': skill, 'alpha': 0.0, 'beta': 2.0},
              {'name': 'ctrl_skill_push-2', 'vector': skill, 'alpha': 0.0, 'beta': -2.0}]
    return specs


def _generate_one(backend, request, sample, hook, max_new_tokens):
    import torch
    inputs = backend.encode([request])
    width = inputs['input_ids'].shape[1]
    torch.manual_seed(sample['seed'])
    with torch.inference_mode(), hook:
        out = backend._model.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=max_new_tokens,
                                      temperature=sample['temperature'], top_p=sample['top_p'], top_k=sample['top_k'])
    seq = out[0, width:].tolist()
    eos = backend._model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    stop = next((n for n, t in enumerate(seq) if t in eos), None)
    body = seq if stop is None else seq[:stop + 1]
    text = backend._processor.tokenizer.decode(body, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text, body, stop is not None


def run_steer(*, capture_run, label_file, project_root='.', output_root='outputs', run_id=None, layer=22, position='early',
              steer_pos='prompt_end', alphas=(0.25, 0.5, 1.0), betas=(-2, -1, 1, 2), n_candidates=8, materials=None,
              model_path=None, device='cuda:0', max_new_tokens=1024, temperature=0.7, top_p=0.8, top_k=20,
              w_dir=None, reference_run=None, materials_file=None, dry_run=False, max_calls=0):
    root = Path(project_root).resolve()
    base = root / output_root
    rows = CT.build_table(capture_run=capture_run, label_file=label_file, project_root=project_root, output_root=output_root)
    if materials:
        ids = {line.split('#')[0].strip() for line in Path(materials).read_text().splitlines() if line.split('#')[0].strip()}
        rows = [r for r in rows if r['parent_id'] in ids or r['material_id'] in ids]
    meta = CT.meta(rows)
    X = CT.load_feats(capture_run=capture_run, rows=rows, layers=(layer,), positions=(position,),
                      project_root=project_root, output_root=output_root)[position][:, 0]
    wpath = Path(w_dir) if w_dir else (base / run_id / 'w_dir.npy' if run_id else None)
    if wpath is None or not Path(wpath).exists():
        from .diag_curve import direction_axis
        w = direction_axis(rows, X, meta)
        print('[steer] w_dir.npy not found; fitted the text-image axis from the current direction labels (fallback)')
    else:
        w = np.load(wpath)
    w = w / (np.linalg.norm(w) or 1.0)
    mixed = {m for m in np.unique(meta['material_id'])
             if len({(r['label_new'] or {}).get('verdict') or r['label_old'].get('redundant') for r in rows
                     if r['material_id'] == m}) > 1}
    info = direction_info(X, w, meta['material_id'], mixed)
    specs = _conditions(alphas, betas, w, (X, meta['skill']), seed=20261113)
    parents = sorted({r['parent_id'] for r in rows})
    plan = {'capture_run': capture_run, 'layer': layer, 'position': position, 'steer_pos': steer_pos,
            'b': info['b'], 'sigma': info['sigma'], 'conditions': [s['name'] for s in specs],
            'n_candidates': n_candidates, 'materials': parents, 'seeds': {m: [candidate_seed(20261113, m, j) for j in range(n_candidates)] for m in parents}}
    if dry_run:
        check = _mock_hook_check()
        calls = len(parents) * len(specs) * n_candidates
        print(f'[dry-run] steer hook check: {check}')
        print(f'[dry-run] {len(parents)} materials x {len(specs)} conditions x {n_candidates} candidates = {calls} generations')
        print(f'[dry-run] GPU estimate: bf16 Qwen3-VL-8B weights ~17 GiB + activations/KV cache ~4-6 GiB -> ~22-24 GiB; '
              f'label evaluation afterwards ~{int(calls * 0.2)} distinct questions x 2.4 calls ~= {round(calls * 0.2 * 2.4)} API calls')
        print(f'[dry-run] b={info["b"]:.4f} sigma={info["sigma"]:.4f}; conditions={plan["conditions"]}')
        return {'dry_run': True, 'hook_check': check, 'generations': calls, **plan}
    if not model_path:
        raise ValueError('--local-model-path is required for a real steering run')
    from .local_vision import QwenVLBackend
    from .candidates import load_materials_any
    from .prompts import PromptSet
    from .providers import CompletionRequest
    cap_identity = json.loads((base / capture_run / 'model_identity.json').read_text())
    template = cap_identity.get('prompt_template', 'plain-v2')
    prompts = PromptSet(root / 'prompts', names=(template,))
    pool = load_materials_any(root, output_root, reference_run, materials_file)
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens)
    backend._load()
    backend.max_new_tokens = max_new_tokens
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'steer_plan.json', plan)
    generated = []
    for parent in parents:
        material = pool[rows[[r['parent_id'] for r in rows].index(parent)]['material_id']]
        request = CompletionRequest(task='generate_question', prompt=prompts.render(template, T=material['text']),
                                    image_path=str(root / material['image_path']))
        for spec in specs:
            hook = steer_hook(backend._model, layer=layer, vector=spec['vector'], alpha=spec['alpha'], b=info['b'],
                              beta=spec['beta'], sigma=info['sigma'], steer_pos=steer_pos)
            for j in range(n_candidates):
                seed = candidate_seed(20261113, parent, j)
                text, body, ended = _generate_one(backend, request, {'seed': seed, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k},
                                                  hook, max_new_tokens)
                generated.append({'cand_id': f'{parent}~{spec["name"]}~c{j}', 'parent_id': parent,
                                  'material_id': material['material_id'], 'skill': rows[[r['parent_id'] for r in rows].index(parent)]['skill'],
                                  'condition': spec['name'], 'question': text, 'label_old': {'redundant': None, 'direction': None},
                                  'label_new': None, 'valid': None, 'evaluated': False,
                                  'features': {'generated_token_ids': body}, 'n_question_tokens': len(body) - (1 if ended else 0),
                                  'seed': seed, 'finish': 'stop' if ended else 'length_or_timeout'})
    write_jsonl(out / 'candidates.jsonl', generated)
    write_json(out / 'summary.json', {'run_id': run_id, 'candidates': len(generated), 'conditions': [s['name'] for s in specs],
                                      'note': 'label with label-repair or the cascade evaluator before analysis'})
    return {'run_id': run_id, 'candidates': len(generated), 'conditions': [s['name'] for s in specs]}
