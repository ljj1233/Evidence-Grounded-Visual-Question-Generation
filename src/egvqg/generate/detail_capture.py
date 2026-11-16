"""Teacher-forced detail capture from an existing candidate run (no new generation).

For each stored candidate the same prompt is encoded once per material and the stored question tokens are fed back
through the model.  From that single forward we save, at the last prompt position and at every question-token
position: the per-layer hidden states, the per-head outputs (input of each layer's `o_proj`, reshaped to
[heads, head_dim]), and the prompt-stage image/text mean-pooled states.  This adds head, token-subset and
generation-step readouts without changing the candidate set.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..core.records import append_jsonl, read_jsonl, utc_now, write_json
from .capture import forward_extras, has_question_states

VERSION = 'detail-capture-v1'


def detail_positions(n_prompt_tokens, n_body_tokens):
    """Positions used for the last-prompt state and every stored question/EOS token (teacher forcing, causal)."""
    if n_prompt_tokens < 1 or n_body_tokens < 1:
        raise ValueError('invalid prompt/body length')
    return [n_prompt_tokens - 1] + list(range(n_prompt_tokens, n_prompt_tokens + n_body_tokens))


def _o_proj_modules(model):
    found = []
    for name, module in model.named_modules():
        match = re.search(r'layers\.(\d+)\.self_attn\.o_proj$', name)
        if match:
            found.append((int(match.group(1)), module))
    return sorted(found)


def _material_filter(path):
    """Optional one-ID-per-line or JSONL `{"material_id": ...}` filter; None means all materials."""
    if path is None:
        return None
    ids = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ids.add(json.loads(line)['material_id'] if line.startswith('{') else line)
    return ids


def run_detail_capture(*, run_id, capture_run, project_root='.', output_root='outputs',
                       reference_run='qwen3vl8b-baseline-test1257-simple-v1', materials_file=None,
                       model_path, device='cuda:0', limit=None, material_ids_file=None, prompts_file=None):
    import torch
    from safetensors.torch import save_file
    from .candidates import load_materials_any
    from ..core.local_vision import QwenVLBackend
    from ..core.prompts import PromptSet
    from ..core.providers import CompletionRequest
    from .capture import Capturer

    root = Path(project_root).resolve()
    base = root / output_root
    cap = base / capture_run
    out = base / run_id
    (out / 'detail').mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(cap / 'captures.jsonl')
    allowed = _material_filter(material_ids_file)
    if allowed is not None:
        rows = [r for r in rows if r['material_id'] in allowed]
    prompt_of = {r['material_id']: r['prompt'] for r in read_jsonl(Path(prompts_file))} if prompts_file else {}
    identity = None
    if (cap / 'model_identity.json').exists():
        import json
        identity = json.loads((cap / 'model_identity.json').read_text())
    template = (identity or {}).get('prompt_template', 'plain-v2')
    materials = load_materials_any(root, output_root, reference_run, materials_file)
    prompts = PromptSet(root / 'prompts', names=(template,))
    backend = QwenVLBackend(model_path, device=device)
    backend._load()
    model = backend._model
    tokenizer = backend._processor.tokenizer
    capturer = Capturer(backend, k=4)
    heads = int(model.config.text_config.num_attention_heads)
    head_dim = int(model.config.text_config.hidden_size // heads)
    modules = _o_proj_modules(model)
    if len(modules) != int(model.config.text_config.num_hidden_layers):
        raise ValueError(f'captured {len(modules)} o_proj modules, expected {model.config.text_config.num_hidden_layers}')

    done = set()
    index_path = out / 'detail_index.jsonl'
    if index_path.exists():
        done = {r['candidate_id'] for r in read_jsonl(index_path)}
    state = {'positions': None, 'captured': {}}

    def make_hook(layer):
        def hook(module, args):
            x = args[0][0]                      # [seq, heads*head_dim]
            pos = state['positions']
            state['captured'][layer] = x[pos].reshape(len(pos), heads, head_dim).contiguous()
        return hook

    handles = [module.register_forward_pre_hook(make_hook(layer)) for layer, module in modules]
    prepared, prepared_mid = None, None
    count = 0
    try:
        for row in rows:
            if limit is not None and count >= limit:
                break
            cid = row['candidate_id']
            if cid in done or not has_question_states(row):
                continue
            mid = row['material_id']
            if mid != prepared_mid:
                material = materials[mid]
                prompt_text = prompt_of.get(mid) or prompts.render(template, T=material['text'])
                request = CompletionRequest(task='generate_question', prompt=prompt_text,
                                            image_path=str(root / material['image_path']))
                prepared = capturer.prepare(request, material['text'])
                prepared_mid = mid
            inputs, ids, spans = prepared
            body = [int(t) for t in row['generated_token_ids']]
            if not body:
                continue
            width = inputs['input_ids'].shape[1]
            positions = detail_positions(width, len(body))
            full = torch.cat([inputs['input_ids'],
                              torch.tensor([body], dtype=inputs['input_ids'].dtype, device=inputs['input_ids'].device)], dim=1)
            extra = forward_extras(inputs, len(body))
            state['positions'] = torch.tensor(positions, device=inputs['input_ids'].device)
            state['captured'] = {}
            with torch.inference_mode():
                hs = model(input_ids=full, attention_mask=torch.ones_like(full), output_hidden_states=True, **extra).hidden_states
            hidden_last = torch.stack([h[0, state['positions']] for h in hs], dim=1).to(torch.bfloat16).cpu()
            head_last = torch.stack([state['captured'][layer] for layer, _ in modules], dim=1).to(torch.bfloat16).cpu()
            img_start, img_end = spans['image'] if 'image' in spans else (0, 0)
            txt_start, txt_end = spans['text']
            pooled_image = (torch.stack([h[0, img_start:img_end].mean(0) for h in hs]).to(torch.bfloat16).cpu()
                            if img_end > img_start else torch.zeros(len(hs), hidden_last.shape[-1], dtype=torch.bfloat16))
            pooled_text = torch.stack([h[0, txt_start:txt_end].mean(0) for h in hs]).to(torch.bfloat16).cpu()
            save_file({'hidden_last': hidden_last, 'head_last': head_last,
                       'pooled_image': pooled_image, 'pooled_text': pooled_text}, str(out / 'detail' / f'{cid}.safetensors'))
            append_jsonl(index_path, {'candidate_id': cid, 'material_id': mid, 'file': f'detail/{cid}.safetensors',
                                      'n_prompt_tokens': width, 'n_body_tokens': len(body),
                                      'n_question_tokens': (row.get('plan') or {}).get('n_question_tokens'),
                                      'positions': positions, 'spans': spans, 'template': template,
                                      'teacher_forced': True, 'created_at': utc_now(), 'version': VERSION})
            count += 1
    finally:
        for handle in handles:
            handle.remove()
    summary = {'run_id': run_id, 'capture_run': capture_run, 'captured': count,
               'files': len(list((out / 'detail').glob('*.safetensors'))), 'version': VERSION,
               'template': template, 'heads': heads, 'head_dim': head_dim}
    write_json(out / 'summary.json', summary)
    return summary
