"""P5/P6 readout: how far do hidden states move when one modality is removed or replaced? (no attention weights are used)

For every captured candidate the *same* question tokens are teacher-forced under several prompts:
  full            image + text (reference)
  image_removed   no image tokens at all (largest distribution shift)
  image_gray      image replaced by a uniform gray image of the same size (token count preserved)
  image_swap      image of another material of the same skill
  text_removed    material text replaced by the empty string
  text_swap       text of another material of the same skill
At the last prompt position (`prompt_end`) and the last question token (`question_end`) the per-layer relative L2 distance
||h_full - h_variant|| / ||h_full|| and the cosine distance are stored. Conclusions should hold under several schemes;
removal is out of distribution, which is why swap and gray variants are included.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .records import read_jsonl
from .whitebox import forward_extras, has_question_states

VERSION = 'modality-distance-v1'
SCHEMES = {'image_removed': ('none', 'same'), 'image_gray': ('gray', 'same'), 'image_swap': ('swap', 'same'),
           'text_removed': ('same', 'empty'), 'text_swap': ('same', 'swap')}


def swap_partner(material_id, same_group_ids, parent_of=None):
    """Deterministic partner: the next material (cyclic, sorted) of the same group whose parent differs (variants of one source
    material must never swap with each other); None if there is none."""
    parent_of = parent_of or {}
    ids = sorted(set(same_group_ids))
    if material_id not in ids:
        return None
    mine = parent_of.get(material_id, material_id)
    start = ids.index(material_id)
    for step in range(1, len(ids)):
        cand = ids[(start + step) % len(ids)]
        if parent_of.get(cand, cand) != mine:
            return cand
    return None


def modality_scores(row, scheme_pairs=(('image_removed', 'text_removed'), ('image_gray', 'text_removed'), ('image_swap', 'text_swap')),
                    position='question_end', layer=-1):
    """Per scheme pair: image-minus-text shift and the smaller of the two shifts (both modalities matter)."""
    out = {}
    for img, txt in scheme_pairs:
        di, dt = row['rel_l2'][img][position][layer], row['rel_l2'][txt][position][layer]
        out[f'{img}|{txt}'] = {'asymmetry': di - dt, 'min_shift': min(di, dt)}
    return out


def run_distances(*, run_id, project_root='.', output_root='outputs', reference_run=None, model_path, device='cuda:0', limit=None, max_new_tokens=1024, materials_file=None):
    import torch
    from PIL import Image
    from .local_vision import QwenVLBackend
    from .prompts import PromptSet
    from .providers import CompletionRequest
    root = Path(project_root).resolve()
    cap = root / output_root / run_id
    from .candidates import load_materials_any
    materials = load_materials_any(root, output_root, reference_run, materials_file)
    rows = read_jsonl(cap / 'captures.jsonl')
    identity = json.loads((cap / 'model_identity.json').read_text())
    template = identity.get('prompt_template', 'plain-v2')
    pool, parent_of = {}, {}
    for r in rows:
        pool.setdefault(r['skill'], set()).add(r['material_id'])
        parent_of[r['material_id']] = r.get('parent_material_id') or r['material_id']
    everything = set().union(*pool.values()) if pool else set()
    prompts = PromptSet(root / 'prompts', names=(template,))
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens)
    backend._load()
    model = backend._model
    gray_dir = cap / 'tmp_gray'
    gray_dir.mkdir(exist_ok=True)
    done = {r['candidate_id'] for r in read_jsonl(cap / 'distances.jsonl')}

    def gray_path(src):
        with Image.open(src) as im:
            path = gray_dir / f'{im.width}x{im.height}.png'
            if not path.exists():
                Image.new('RGB', im.size, (128, 128, 128)).save(path)
        return str(path)

    def states(inputs, body, n_q):
        width = inputs['input_ids'].shape[1]
        full = torch.cat([inputs['input_ids'], torch.tensor([body[:n_q]], device=inputs['input_ids'].device)], dim=1)
        extra = forward_extras(inputs, n_q)
        with torch.inference_mode():
            hs = model(input_ids=full, attention_mask=torch.ones_like(full), output_hidden_states=True, **extra).hidden_states
        pick = lambda pos: torch.stack([h[0, pos] for h in hs]).float()
        return {'prompt_end': pick(width - 1), 'question_end': pick(width + n_q - 1)}

    count = 0
    cache, cached_mid = {}, None
    for r in rows:
        if r['candidate_id'] in done or not has_question_states(r):
            continue
        if limit and count >= limit:
            break
        began = time.monotonic()
        mid = r['material_id']
        m = materials[mid]
        partner = swap_partner(mid, pool[r['skill']], parent_of) or swap_partner(mid, everything, parent_of)
        pm = materials.get(partner) if partner else None
        body = r['generated_token_ids']
        n_q = r['plan']['n_question_tokens']

        def request(image, text):
            if image == 'gray':
                img = gray_path(root / m['image_path'])
            elif image == 'none':
                img = None
            else:
                img = str(root / (m['image_path'] if image == 'same' else pm['image_path']))
            t = {'same': m['text'], 'empty': '', 'swap': pm['text'] if pm else None}[text]
            return CompletionRequest(task='generate_question', prompt=prompts.render(template, T=t), image_path=img)

        if mid != cached_mid:          # captured candidates of one material share every prompt variant, so encode once per material
            cache.clear()
            cached_mid = mid

        def encoded(image, text):
            if (image, text) not in cache:
                cache[(image, text)] = backend.encode([request(image, text)])
            return cache[(image, text)]

        ref_states = states(encoded('same', 'same'), body, n_q)
        row = {'candidate_id': r['candidate_id'], 'material_id': mid, 'partner': partner, 'rel_l2': {}, 'cosine': {}}
        for name, (image, text) in SCHEMES.items():
            if (image == 'swap' or text == 'swap') and not pm:
                continue
            st = states(encoded(image, text), body, n_q)
            row['rel_l2'][name], row['cosine'][name] = {}, {}
            for pos in ('prompt_end', 'question_end'):
                a, b = ref_states[pos], st[pos]
                row['rel_l2'][name][pos] = [round(float(x), 5) for x in ((a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6))]
                row['cosine'][name][pos] = [round(float(x), 5) for x in (1 - torch.nn.functional.cosine_similarity(a, b, dim=-1))]
        row['seconds'] = round(time.monotonic() - began, 2)
        with (cap / 'distances.jsonl').open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, sort_keys=True) + '\n')
        count += 1
    return {'run_id': run_id, 'distances': len(read_jsonl(cap / 'distances.jsonl')), 'version': VERSION}
