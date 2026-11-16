"""P5 geometry: per-layer anisotropy and effective rank of image-token vs text-token hidden states (prompt only, no generation).

anisotropy  = mean pairwise cosine similarity of the token states of one modality (1 = all tokens point the same way)
eff. rank   = exp(entropy of normalised singular values) of the mean-centred token matrix
Both are computed per layer and per material; the dataset mean vector of each modality is kept for the recentering intervention.
Descriptive only (pre-registered: no gate).
"""
from __future__ import annotations

import json
from pathlib import Path

from .records import read_jsonl, write_json


def pairwise_cosine_mean(sum_of_units_sq_norm, n):
    """Mean pairwise cosine of n unit vectors given ||sum u||^2 (self-pairs excluded)."""
    return float('nan') if n < 2 else (sum_of_units_sq_norm - n) / (n * (n - 1))


def effective_rank(singular_values):
    """Roy & Vetterli effective rank from singular values."""
    import math
    total = sum(singular_values)
    if total <= 0:
        return 0.0
    ps = [s / total for s in singular_values if s > 0]
    return math.exp(-sum(p * math.log(p) for p in ps))


def run_geometry(*, run_id, capture_run, project_root='.', output_root='outputs', reference_run, model_path, device='cuda:0'):
    import torch
    from safetensors.torch import save_file
    from .local_vision import QwenVLBackend
    from .prompts import PromptSet
    from .providers import CompletionRequest
    from .whitebox import Capturer
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    cap = root / output_root / capture_run
    materials = {m['material_id']: m for m in read_jsonl(root / output_root / reference_run / 'materials.jsonl')}
    mids = sorted({r['material_id'] for r in read_jsonl(cap / 'captures.jsonl')})
    template = json.loads((cap / 'model_identity.json').read_text()).get('prompt_template', 'plain-v2')
    prompts = PromptSet(root / 'prompts', names=(template,))
    backend = QwenVLBackend(model_path, device=device)
    backend._load()
    capturer = Capturer(backend)
    rows, means = [], {'image': [], 'text': []}
    for mid in mids:
        m = materials[mid]
        request = CompletionRequest(task='generate_question', prompt=prompts.render(template, T=m['text']), image_path=str(root / m['image_path']))
        inputs = backend.encode([request])
        spans = capturer.spans(inputs['input_ids'][0].tolist(), m['text'])
        extra = {k: v for k, v in inputs.items() if k not in ('input_ids', 'attention_mask')}
        with torch.inference_mode():
            hs = backend._model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True, **extra).hidden_states
        row = {'material_id': mid, 'n_image_tokens': spans['image'][1] - spans['image'][0], 'n_text_tokens': spans['text'][1] - spans['text'][0]}
        for name in ('image', 'text'):
            a, b = spans[name]
            aniso, erank, norm, mean_vec = [], [], [], []
            for h in hs:
                tok = h[0, a:b].float()
                unit = tok / tok.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                aniso.append(pairwise_cosine_mean(float(unit.sum(0).norm() ** 2), tok.shape[0]))
                sv = torch.linalg.svdvals(tok - tok.mean(0, keepdim=True)).tolist()
                erank.append(effective_rank(sv))
                norm.append(float(tok.norm(dim=-1).mean()))
                mean_vec.append(tok.mean(0).cpu())
            row[name] = {'anisotropy': aniso, 'effective_rank': erank, 'mean_norm': norm}
            means[name].append(torch.stack(mean_vec))
        rows.append(row)
    with (out / 'geometry.jsonl').open('w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    save_file({k: torch.stack(v).contiguous() for k, v in means.items()}, str(out / 'modality_means.safetensors'))
    write_json(out / 'geometry_index.json', {'capture_run': capture_run, 'materials': mids, 'template': template})
    return {'run_id': run_id, 'materials': len(rows)}


def geometry_summary(*, run_id, label_file, capture_run, project_root='.', output_root='outputs', B=1000):
    """Per-layer image-minus-text gap in anisotropy / effective rank with a material bootstrap, and its Spearman correlation with the
    material-level redundant fraction (descriptive)."""
    import numpy as np
    from scipy.stats import spearmanr
    from .probe_labels import material_profile
    from .probing import bootstrap_groups, interval
    base = Path(project_root).resolve() / output_root
    rows = read_jsonl(base / run_id / 'geometry.jsonl')
    prof = {p['material_id']: p for p in material_profile(read_jsonl(base / capture_run / label_file))}
    result = {'n_materials': len(rows), 'layers': {}}
    for key in ('anisotropy', 'effective_rank'):
        img = np.array([r['image'][key] for r in rows])
        txt = np.array([r['text'][key] for r in rows])
        gap = img - txt
        mats = np.array([r['material_id'] for r in rows])
        frac = np.array([prof[m]['redundant_fraction'] if m in prof and prof[m]['redundant_fraction'] is not None else np.nan for m in mats])
        ok = ~np.isnan(frac)
        per_layer = []
        for layer in range(gap.shape[1]):
            boot = bootstrap_groups(mats, {'g': lambda idx, l=layer: float(np.nanmean(gap[idx, l]))}, B=B // 5, seed=layer)
            rho = spearmanr(gap[ok, layer], frac[ok]).statistic if ok.sum() > 5 else float('nan')
            per_layer.append({'layer': layer, 'image_mean': float(np.nanmean(img[:, layer])), 'text_mean': float(np.nanmean(txt[:, layer])),
                              'gap_mean': float(np.nanmean(gap[:, layer])), 'gap_ci': interval(boot['g']), 'spearman_gap_vs_redundant_fraction': float(rho)})
        result['layers'][key] = per_layer
    write_json(base / run_id / 'geometry_summary.json', result)
    return {'n_materials': result['n_materials']}
