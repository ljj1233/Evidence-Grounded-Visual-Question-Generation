"""Activation steering of the question generator (exploratory branch, pre-registered separately from G1-G4 in the plan).

Direction: mean difference of `early` states (first question tokens) between non-redundant and redundant candidates, after
(1) subtracting each skill's mean and (2) projecting out the language direction (zh vs non-zh) so that the vector cannot simply be
"write Chinese". Controls are built the same way: a random in-distribution direction (difference of means of two random candidate
groups of the same sizes, same residualisation) and an isotropic Gaussian direction. All have the same norm (alpha * sigma, where
sigma is the std of the training projections on the real direction). The vector is added at one decoder layer to the last prompt
position and to every generated position. Training skills never include the held-out skills used for evaluation.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .records import read_jsonl, write_json


# ---------- directions (numpy) ----------

def demean_by_group(X, groups):
    X = np.asarray(X, float)
    out = X.copy()
    for g in set(groups):
        idx = [i for i, k in enumerate(groups) if k == g]
        out[idx] = X[idx] - X[idx].mean(0)
    return out


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def remove_component(v, u):
    u = unit(u)
    return v - float(v @ u) * u


def real_direction(X, y, skills, zh):
    """Unit direction at one layer plus diagnostics. X [n,H] early states; y 1 = non-redundant; zh 1 = Chinese question."""
    Xd = demean_by_group(X, skills)
    y, zh = np.asarray(y), np.asarray(zh)
    d = Xd[y == 1].mean(0) - Xd[y == 0].mean(0)
    lang = Xd[zh == 1].mean(0) - Xd[zh == 0].mean(0) if 0 < zh.sum() < len(zh) else np.zeros_like(d)
    cos_lang = float(unit(d) @ unit(lang)) if np.linalg.norm(lang) > 0 and np.linalg.norm(d) > 0 else 0.0
    d2 = unit(remove_component(d, lang)) if np.linalg.norm(lang) > 0 else unit(d)
    sigma = float((Xd @ d2).std())
    return d2, sigma, {'cos_with_language_before_removal': cos_lang, 'n_pos': int((y == 1).sum()), 'n_neg': int((y == 0).sum())}


def random_manifold_direction(X, y, skills, zh, seed):
    """Same recipe with the class labels shuffled: a random direction that lives where real activation differences live."""
    rng = np.random.default_rng(seed)
    d, _, _ = real_direction(X, rng.permutation(np.asarray(y)), skills, zh)
    return d


def gaussian_direction(h, seed):
    return unit(np.random.default_rng(seed).normal(size=h))


# ---------- hook (torch) ----------

def find_layers(model):
    for path in ('model.language_model.layers', 'language_model.layers', 'model.layers', 'layers'):
        obj = model
        try:
            for part in path.split('.'):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError('decoder layers not found')


@contextmanager
def steering(model, layer, vector, alpha=1.0):
    """Add alpha * vector to the output of decoder layer `layer`: last prompt position at prefill, the single position at each decode step."""
    import torch
    layers = find_layers(model)
    v = torch.as_tensor(np.asarray(vector), dtype=torch.float32)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        add = (alpha * v).to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.clone()
        if hidden.shape[1] > 1:
            hidden[:, -1, :] += add
        else:
            hidden[:, 0, :] += add
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def intervention_from_spec(path):
    """{'description': spec, 'factory': callable(model, n_prompt_tokens) -> context manager} for Capturer / candidates."""
    from safetensors.numpy import load_file
    spec = json.loads(Path(path).read_text())
    vectors = load_file(str(Path(path).parent / spec['vectors_file']))
    vec = vectors[spec['vector_key']]
    signed = float(spec['alpha']) * float(spec['sign']) * float(spec['sigma'])
    return {'description': spec, 'factory': lambda model, n: steering(model, spec['layer'], vec, signed)}


# ---------- preparation and screening ----------

def prepare_steering(*, run_id, training, layers, alphas, heldout, seed=20261113, project_root='.', output_root='outputs', reference_run=None,
                     materials_file=None):
    """training: [(capture_run, label_file)]. Writes vectors, one spec per condition, and the held-out materials file."""
    from safetensors.numpy import save_file
    from .probing import load_candidates
    base = Path(project_root).resolve() / output_root
    out = base / run_id / 'steer'
    out.mkdir(parents=True, exist_ok=True)
    rows, _, early, _ = load_candidates([r for r, _ in training], [f for _, f in training], project_root=project_root, output_root=output_root)
    decided = [i for i, r in enumerate(rows) if r['valid'] == 1 and r['redundant'] is not None]
    skills_all = sorted({rows[i]['skill'] for i in decided})
    held = set(np.random.default_rng(seed).choice(skills_all, size=min(heldout, len(skills_all)), replace=False).tolist())
    train = [i for i in decided if rows[i]['skill'] not in held]
    y = np.array([1 - rows[i]['redundant'] for i in train])
    skills = [rows[i]['skill'] for i in train]
    zh = np.array([int(rows[i]['zh_share'] > 0.5) for i in train])
    vectors, specs, diagnostics = {}, [], {}
    for layer in layers:
        X = early[train][:, layer]
        d, sigma, info = real_direction(X, y, skills, zh)
        vectors[f'dir_L{layer}'] = d.astype(np.float32)
        vectors[f'randM_L{layer}'] = random_manifold_direction(X, y, skills, zh, seed + layer).astype(np.float32)
        vectors[f'gauss_L{layer}'] = gaussian_direction(X.shape[1], seed + 1000 + layer).astype(np.float32)
        diagnostics[layer] = {**info, 'sigma': sigma, 'cos_dir_vs_randM': float(d @ vectors[f'randM_L{layer}']),
                              'cos_dir_vs_gauss': float(d @ vectors[f'gauss_L{layer}'])}
        for alpha in alphas:
            for name, key, sign in (('dirP', 'dir', 1), ('dirN', 'dir', -1), ('randM', 'randM', 1), ('gauss', 'gauss', 1)):
                cond = f'{name}_L{layer}_a{alpha:g}'
                specs.append({'condition': cond, 'layer': layer, 'vector_key': f'{key}_L{layer}', 'alpha': alpha, 'sign': sign, 'sigma': sigma,
                              'vectors_file': 'vectors.safetensors'})
    save_file(vectors, str(out / 'vectors.safetensors'))
    for s in specs:
        write_json(out / f"spec_{s['condition']}.json", s)
    held_materials = sorted({rows[i]['material_id'] for i in decided if rows[i]['skill'] in held})
    meta = {'heldout_skills': sorted(held), 'n_train_candidates': len(train), 'n_heldout_materials': len(held_materials), 'layers': list(layers),
            'alphas': list(alphas), 'diagnostics': diagnostics, 'conditions': [s['condition'] for s in specs], 'seed': seed}
    from .candidates import load_materials_any
    from .records import write_jsonl
    pool = load_materials_any(Path(project_root).resolve(), output_root, reference_run, materials_file)
    skill_of = {r['material_id']: r['skill'] for r in rows}
    write_jsonl(out / 'heldout_materials.jsonl', [{**pool[m], 'group': skill_of[m]} for m in held_materials if m in pool])
    write_json(out / 'steer_meta.json', meta)
    write_json(out / 'heldout_material_ids.json', held_materials)
    return meta


def screen_conditions(rows_by_condition, control='ctrl', min_ended=0.9, length_band=(0.5, 2.0), max_language_shift=0.25):
    """Label-free screening of steered generations against the unsteered control (same seeds): fluent (ends with the end token),
    similar length, language not flipped. Returns {condition: {metrics, survives}}."""
    def stats(rows):
        n = len(rows)
        return {'n': n, 'ended': sum(r['plan']['complete'] for r in rows) / n if n else float('nan'),
                'tokens': float(np.mean([r['n_generated_tokens'] for r in rows])) if n else float('nan'),
                'zh': float(np.mean([r['zh_share'] > 0.5 for r in rows])) if n else float('nan'),
                'distinct': len({r['question'].strip() for r in rows}) / n if n else float('nan')}
    base = stats(rows_by_condition[control])
    out = {}
    for cond, rows in rows_by_condition.items():
        s = stats(rows)
        ok = (s['ended'] >= min_ended and length_band[0] <= s['tokens'] / base['tokens'] <= length_band[1]
              and abs(s['zh'] - base['zh']) <= max_language_shift)
        out[cond] = {**s, 'survives': bool(ok) if cond != control else True}
    return out


def paired_effect(rows_by_condition, control='ctrl', B=1000):
    """Per-material change of 'valid and non-redundant' (and validity) against the control, with a bootstrap over materials."""
    from collections import defaultdict
    from .probing import bootstrap_groups, interval

    def per_material(rows):
        by = defaultdict(list)
        for r in rows:
            by[r['material_id']].append((1.0 if r['valid'] == 1 else 0.0, 1.0 if r['valid'] == 1 and r['redundant'] == 0 else 0.0))
        return {m: np.mean(v, 0) for m, v in by.items()}
    base = per_material(rows_by_condition[control])
    result = {}
    for cond, rows in rows_by_condition.items():
        if cond == control:
            continue
        pm = per_material(rows)
        mats = sorted(set(pm) & set(base))
        if len(mats) < 5:
            result[cond] = {'status': 'too_few_materials', 'n': len(mats)}
            continue
        d = np.array([pm[m] - base[m] for m in mats])
        boot = bootstrap_groups(np.array(mats), {'s': lambda idx: float(d[idx, 1].mean()), 'v': lambda idx: float(d[idx, 0].mean())}, B=B)
        result[cond] = {'n_materials': len(mats), 'd_success': float(d[:, 1].mean()), 'd_success_ci': interval(boot['s']),
                        'd_valid': float(d[:, 0].mean()), 'd_valid_ci': interval(boot['v'])}
    return result


def g3_verdict(effects, condition, random_conditions):
    """Pre-registered G3: the real-direction effect minus each random control's effect has an interval above 0 (here: the real effect's
    point estimate exceeds both controls' upper bounds and its own interval excludes 0) and validity does not drop by more than 2 points."""
    e = effects[condition]
    if 'status' in e:
        return {'passed': False, 'reason': e['status']}
    beats = all(e['d_success_ci'][0] > effects[c]['d_success_ci'][1] for c in random_conditions if 'status' not in effects[c])
    return {'passed': bool(e['d_success_ci'][0] > 0 and beats and e['d_valid'] >= -0.02), 'd_success': e['d_success'], 'd_success_ci': e['d_success_ci'],
            'd_valid': e['d_valid']}
