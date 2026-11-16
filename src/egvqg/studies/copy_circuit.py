"""E4: find the attention heads that copy a numeral from the context text into the generated question,
then test at decode time whether ablating them turns a copied single-modality question into a genuine
dual question.

Attribution needs real attention weights, so it loads its own model with `attn_implementation='eager'`.
Intervention does not need weights: it zeroes the output slice of selected heads at `o_proj` on decode
steps only (a head-output ablation, not a span-restricted attention mask; stated as such in the report).
"""
from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from pathlib import Path

from .inline_analysis import contains_number
from ..core.records import read_jsonl, write_json, write_jsonl, stable_id, utc_now

PROMPT = 'plain-v2'
DEFAULT_CATEGORY = 'derived'
_DIGIT = re.compile(r'\d')


def _render(root, text):
    from ..core.prompts import prompt_path

    return prompt_path(Path(root) / 'prompts', PROMPT).read_text(encoding='utf-8').replace('$T', text)


def _load_eager(path, device):
    import torch
    import transformers
    from transformers import AutoProcessor
    path = Path(path)
    architecture = json.loads((path / 'config.json').read_text())['architectures'][0]
    model = getattr(transformers, architecture).from_pretrained(
        str(path), local_files_only=True, dtype=torch.bfloat16, device_map={'': device},
        attn_implementation='eager').eval()
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    return model, processor


def _text_positions(inputs):
    """Positions of real text tokens (mm_token_type_ids 0); image/video placeholders are excluded."""
    types = inputs.get('mm_token_type_ids')
    ids = inputs['input_ids'][0].tolist()
    if types is None:
        return list(range(len(ids)))
    return [i for i, kind in enumerate(types[0].tolist()) if kind == 0]


def _digit_tokens(tokenizer, ids, allowed):
    out = []
    for position in allowed:
        text = tokenizer.decode([ids[position]])
        if _DIGIT.search(text):
            out.append(position)
    return out


def _generate_with_attention(model, processor, prompt, image_path, max_new_tokens):
    import torch
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    conversation = [[{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                  {'type': 'text', 'text': prompt}]}]]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt',
        enable_thinking=False, processor_kwargs={'padding': True}).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             return_dict_in_generate=True, output_attentions=True)
    width = inputs['input_ids'].shape[1]
    generated = out.sequences[0, width:].tolist()
    return inputs, out, generated, width


def attribute_material(model, processor, root, material, max_new_tokens=24):
    """Per-head attention from the first copied numeral in the question to the context numeral tokens."""
    tokenizer = processor.tokenizer
    prompt = _render(root, material['text'])
    inputs, out, generated, width = _generate_with_attention(
        model, processor, prompt, Path(root) / material['image_path'], max_new_tokens)
    ids = inputs['input_ids'][0].tolist()
    text_positions = _text_positions(inputs)
    context_digits = _digit_tokens(tokenizer, ids, text_positions)
    question = ''.join(tokenizer.decode([i]) for i in generated).strip()
    generated_digits = [(step, tokenizer.decode([token]))
                        for step, token in enumerate(generated) if _DIGIT.search(tokenizer.decode([token]))]
    row = {'material_id': material['material_id'], 'question': question, 'copied': bool(generated_digits),
           'context_digit_tokens': [tokenizer.decode([ids[p]]).strip() for p in context_digits],
           'question_digits': [text for _, text in generated_digits], 'heads': {}}
    if not generated_digits or not context_digits:
        return row
    step = generated_digits[0][0]
    attentions = out.attentions[step]
    scores = {}
    for layer, attn in enumerate(attentions):
        weights = attn[0, :, -1, :]                            # [heads, keys]
        mass = weights[:, context_digits].sum(-1)
        for head in range(mass.shape[0]):
            scores[f'{layer}:{head}'] = round(float(mass[head]), 4)
    row['heads'] = scores
    row['top'] = sorted(scores.items(), key=lambda kv: -kv[1])[:10]
    return row


def _aggregate(rows, top_k=20):
    totals, appearances = defaultdict(float), defaultdict(int)
    copied = [row for row in rows if row.get('top')]
    for row in copied:
        for key, mass in row['top']:
            totals[key] += mass
            appearances[key] += 1
    ranked = sorted(totals, key=lambda key: (-appearances[key], -totals[key] / appearances[key]))
    heads = [{'head': key, 'layer': int(key.split(':')[0]), 'index': int(key.split(':')[1]),
              'mean_top_mass': round(totals[key] / appearances[key], 4), 'materials_in_top': appearances[key]}
             for key in ranked[:top_k]]
    return {'materials': len(rows), 'copied_materials': len(copied), 'heads': heads}


def attribute_head_copy(*, run_id, materials_path, model_path, category=DEFAULT_CATEGORY, n_materials=20,
                        max_new_tokens=24, device='cuda:0', project_root='.', output_root='outputs'):
    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    materials = [m for m in read_jsonl(Path(materials_path)) if m['category'] == category][:n_materials]
    if not materials:
        raise ValueError(f'no materials of category {category}')
    model, processor = _load_eager(model_path, device)
    rows = [attribute_material(model, processor, root, material, max_new_tokens) for material in materials]
    write_jsonl(out / 'attribution.jsonl', rows)
    summary = {'created_at': utc_now(), 'run_id': run_id, 'category': category, 'prompt': PROMPT,
               'model_path': str(model_path), 'max_new_tokens': max_new_tokens, **_aggregate(rows)}
    write_json(out / 'heads.json', summary)
    (out / 'attribution_report.md').write_text(_render_attribution(summary), encoding='utf-8')
    return summary


def _render_attribution(summary):
    lines = [f'# Copy-head attribution {summary["run_id"]}', '',
             f'Category `{summary["category"]}`, prompt `{summary["prompt"]}`, '
             f'{summary["copied_materials"]}/{summary["materials"]} materials copy a context numeral into the question.', '',
             'Heads ranked by how often they are in a material\'s top-10 by attention to the context numeral, '
             'then by mean mass.', '',
             '| rank | head | mean top-10 mass | materials in top-10 |', '|---:|---|---:|---:|']
    for rank, head in enumerate(summary['heads'], start=1):
        lines.append(f'| {rank} | L{head["layer"]:02d}H{head["index"]:02d} | {head["mean_top_mass"]:.3f} '
                     f'| {head["materials_in_top"]} |')
    return '\n'.join(lines) + '\n'


# ---------- intervention: head-output ablation at decode time ----------

ABLATIONS = ('control', 'top8', 'top16', 'random8')


def _language_layers(model):
    """The ordered text-decoder layers, independent of the wrapper naming in this transformers version."""
    named = [(name, module) for name, module in model.named_modules() if name.endswith('.self_attn.o_proj')]
    ranked = [item for item in named if 'language_model' in item[0]] or named
    indexed = []
    for name, module in ranked:
        parts = [p for p in name.split('.') if p.isdigit()]
        if parts:
            indexed.append((int(parts[-1]), module))
    return [module for _, module in sorted(indexed)]


def _head_ablation_hook(modules, heads, head_dim):
    """Zero the output slice of the given heads on decode steps (one query position); prefill is untouched."""
    def hook(_module, args):
        hidden = args[0]
        if hidden.shape[1] != 1:            # prefill: every prompt position attends normally
            return None
        updated = hidden.clone()
        for head in heads:
            updated[..., head * head_dim:(head + 1) * head_dim] = 0
        return (updated,) + args[1:]

    handles = []
    for layer, heads_at_layer in heads.items():
        handle = modules[layer].register_forward_pre_hook(hook)
        handles.append(handle)
    return handles


def _heads_by_layer(keys):
    grouped = defaultdict(list)
    for key in keys:
        layer, head = key.split(':')
        grouped[int(layer)].append(int(head))
    return dict(grouped)


def run_copy_ablation(*, run_id, heads_file, materials_path, model_path, conditions=ABLATIONS, n_materials=20,
                      category=DEFAULT_CATEGORY, samples=4, temperature=0.7, top_p=0.8, top_k=20,
                      max_new_tokens=40, device='cuda:0', project_root='.', output_root='outputs'):
    """Generate with selected copy heads silenced at decode time and report copy/dual rates per condition."""
    import random

    from ..core.local_vision import QwenVLBackend
    from ..core.providers import CompletionRequest
    from .synthetic_composition import classify_generated_question

    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    heads = json.loads(Path(heads_file).read_text())['heads']
    ranked = [head['head'] for head in heads]
    materials = [m for m in read_jsonl(Path(materials_path)) if m['category'] == category][:n_materials]
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens, do_sample=True,
                            temperature=temperature, top_p=top_p, top_k=top_k)
    backend._load()
    modules = _language_layers(backend._model)
    text_config = backend._model.config.text_config if hasattr(backend._model.config, 'text_config') \
        else backend._model.config
    head_dim = text_config.head_dim
    all_heads = [f'{layer}:{head}' for layer in range(len(modules)) for head in range(text_config.num_attention_heads)]
    chosen = {'control': [], 'top8': ranked[:8], 'top16': ranked[:16],
              'random8': random.Random(20261116).sample(ranked, 8), 'all': all_heads}
    manifest = {'created_at': utc_now(), 'run_id': run_id, 'heads_file': str(heads_file), 'chosen': chosen,
                'category': category, 'samples': samples, 'temperature': temperature, 'top_p': top_p,
                'top_k': top_k, 'max_new_tokens': max_new_tokens, 'conditions': list(conditions),
                'model_path': str(model_path)}
    write_json(out / 'manifest.json', manifest)
    rows = []
    for condition in conditions:
        heads_here = _heads_by_layer(chosen[condition])
        for material in materials:
            for sample in range(samples):
                prompt = _render(root, material['text'])
                request = CompletionRequest(task='copy-ablation', prompt=prompt, text=material['text'],
                                            image_path=str(root / material['image_path']), condition='TI',
                                            metadata={'material_id': material['material_id']})
                handles = _head_ablation_hook(modules, heads_here, head_dim) if heads_here else []
                try:
                    completion = backend.generate([request])[0]
                finally:
                    for handle in handles:
                        handle.remove()
                question = completion.text.strip().replace('\n', ' ')
                rows.append({'condition': condition, 'material_id': material['material_id'], 'sample_index': sample,
                             'question': question, 'generated_label': classify_generated_question(question, material),
                             'has_number': contains_number(question, material['text'], min_digits=1),
                             'mentions_threshold': bool(re.search(r'at least|minimum|more than|greater than', question.lower())),
                             'finish_reason': completion.finish, 'completed_at': utc_now()})
    write_jsonl(out / 'ablations.jsonl', rows)
    summary = _summarize_ablations(run_id, rows, manifest)
    write_json(out / 'ablation_summary.json', summary)
    (out / 'ablation_report.md').write_text(_render_ablation(summary), encoding='utf-8')
    return summary


def _summarize_ablations(run_id, rows, manifest, materials_by_id=None):
    from collections import Counter
    conditions = {}
    for condition in manifest['conditions']:
        subset = [row for row in rows if row['condition'] == condition]
        duplicates = Counter((row['material_id'], row['question']) for row in subset)
        conditions[condition] = {
            'n': len(subset),
            'unique_questions': len(duplicates),
            'copy_rate': sum(1 for row in subset if row['has_number']) / max(len(subset), 1),
            'threshold_mention_rate': sum(1 for row in subset if row['mentions_threshold']) / max(len(subset), 1),
            'labels': dict(Counter(row['generated_label'] for row in subset)),
            'dual_rate': sum(1 for row in subset if row['generated_label'] == 'dual_derived') / max(len(subset), 1),
            'single_image_rate': sum(1 for row in subset if row['generated_label'] == 'single_image') / max(len(subset), 1),
            'examples': [row['question'] for row in subset[:6]],
        }
        if materials_by_id:
            faithful, numbers = 0, []
            for row in subset:
                material = materials_by_id.get(row['material_id']) or {}
                found = [int(x) for x in re.findall(r'\d+', row['question'])]
                if found and found[0] == material.get('threshold'):
                    faithful += 1
                numbers.extend(found[:1])
            conditions[condition]['faithful_threshold_rate'] = faithful / max(len(subset), 1)
            conditions[condition]['first_numbers'] = dict(Counter(numbers).most_common(8))
    return {'run_id': run_id, 'manifest': manifest, 'conditions': conditions}


def _render_ablation(summary, note=None):
    default = ('Heads silenced on decode steps only (prefill untouched). `has_number` means the question '
               'contains a numeral, which on these materials can only come from the context text.')
    lines = [f'# Copy-head ablation {summary["run_id"]}', '', note or default, '',
             '| condition | n | unique | copy rate | threshold mention | dual rate | single-image rate | faithful threshold |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, entry in summary['conditions'].items():
        faithful = entry.get('faithful_threshold_rate')
        lines.append(f'| {name} | {entry["n"]} | {entry["unique_questions"]} | {entry["copy_rate"]:.3f} '
                     f'| {entry["threshold_mention_rate"]:.3f} | {entry["dual_rate"]:.3f} '
                     f'| {entry["single_image_rate"]:.3f} |'
                     + (f' {faithful:.3f} |' if faithful is not None else ''))
    if any(entry.get('faithful_threshold_rate') is not None for entry in summary['conditions'].values()):
        lines.append('')
        lines.append('`faithful threshold` = the first numeral in the question equals the material threshold, '
                     'i.e. the question states the *correct* rule. First-numeral counts are in the summary JSON.')
    lines += ['', 'Examples:', '']
    for name, entry in summary['conditions'].items():
        lines.append(f'- **{name}**: ' + ' | '.join(entry['examples']))
    return '\n'.join(lines) + '\n'


# ---------- intervention: block attention to the context numeral span at generation ----------

MASK_CONDITIONS = ('control', 'block_text_digits', 'block_text_span')


class _KeyBlock:
    """Additive attention bias: generated query positions may not attend to the named key positions."""

    def __init__(self):
        self.spec = None          # {'keys': [...], 'query_start': int, 'strength': float}

    def bias(self, query_len, key_len):
        import torch
        spec = self.spec
        if not spec:
            return None
        keys = [k for k in spec['keys'] if k < key_len]
        if not keys or query_len < 1:
            return None
        matrix = torch.zeros((1, 1, query_len, key_len), dtype=torch.float32,
                             device=torch.device('cuda', torch.cuda.current_device()))
        # Cached decoding passes one query row per step (query_len == 1), so the absolute row offset
        # would fall outside the tensor. A single row can only be a decode step, block it.
        if query_len == 1:
            rows = slice(0, 1)
        elif spec['query_start'] < query_len:
            rows = slice(spec['query_start'], query_len)
        else:
            return None
        matrix[:, :, rows, keys] = spec['strength']
        return matrix


_PATCH = {'block': None, 'original': None}


def _install_patch(block):
    import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
    if _PATCH['original'] is None:
        _PATCH['original'] = modeling.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        import torch
        extra = block.bias(query.shape[2], key.shape[2])
        if extra is not None:
            attention_mask = extra if attention_mask is None else attention_mask + extra
        return _PATCH['original'](module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    modeling.eager_attention_forward = patched
    _PATCH['block'] = block


def _remove_patch():
    if _PATCH['original'] is not None:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
        modeling.eager_attention_forward = _PATCH['original']
    _PATCH['block'] = None


def run_copy_mask(*, run_id, materials_path, model_path, conditions=MASK_CONDITIONS, n_materials=20,
                  category=DEFAULT_CATEGORY, samples=4, temperature=0.7, top_p=0.8, top_k=20,
                  max_new_tokens=40, strength=-1e4, device='cuda:0', project_root='.', output_root='outputs'):
    """Generate with attention to the context numeral tokens blocked from the generated positions."""
    import torch
    from PIL import Image

    from .synthetic_composition import classify_generated_question

    root = Path(project_root).resolve()
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    materials = [m for m in read_jsonl(Path(materials_path)) if m['category'] == category][:n_materials]
    model, processor = _load_eager(model_path, device)
    tokenizer = processor.tokenizer
    torch.manual_seed(42)
    block = _KeyBlock()
    _install_patch(block)
    rows = []
    try:
        for condition in conditions:
            for material in materials:
                prompt = _render(root, material['text'])
                image = Image.open(root / material['image_path']).convert('RGB')
                conversation = [[{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                             {'type': 'text', 'text': prompt}]}]]
                inputs = processor.apply_chat_template(
                    conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
                    return_tensors='pt', enable_thinking=False,
                    processor_kwargs={'padding': True}).to(model.device)
                width = inputs['input_ids'].shape[1]
                digit_positions = _digit_tokens(tokenizer, inputs['input_ids'][0].tolist(),
                                                _text_positions(inputs))
                span_positions = _text_positions(inputs)
                if condition == 'control':
                    block.spec = None
                elif condition == 'block_text_span':
                    block.spec = {'keys': span_positions, 'query_start': width, 'strength': strength}
                else:
                    block.spec = {'keys': digit_positions, 'query_start': width, 'strength': strength}
                for sample in range(samples):
                    with torch.no_grad():
                        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                                   temperature=temperature, top_p=top_p, top_k=top_k,
                                                   num_beams=1)
                    tokens = generated[0, width:].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False).strip().replace('\n', ' ')
                    rows.append({'condition': condition, 'material_id': material['material_id'],
                                 'sample_index': sample, 'question': text,
                                 'generated_label': classify_generated_question(text, material),
                                 'has_number': contains_number(text, material['text'], min_digits=1),
                                 'mentions_threshold': bool(re.search(
                                     r'at least|minimum|more than|greater than|threshold', text.lower())),
                                 'blocked_tokens': [tokenizer.decode([p]).strip() for p in digit_positions],
                                 'completed_at': utc_now()})
    finally:
        _remove_patch()
    write_jsonl(out / 'masked.jsonl', rows)
    manifest = {'created_at': utc_now(), 'run_id': run_id, 'category': category, 'samples': samples,
                'temperature': temperature, 'top_p': top_p, 'top_k': top_k,
                'max_new_tokens': max_new_tokens, 'strength': strength, 'conditions': list(conditions),
                'model_path': str(model_path)}
    summary = _summarize_ablations(run_id, rows, manifest)
    write_json(out / 'mask_summary.json', summary)
    (out / 'mask_report.md').write_text(_render_ablation(summary), encoding='utf-8')
    return summary


def mask_report(*, run_id, sources, materials_path=None, project_root='.', output_root='outputs', title=None):
    """Merge several `copy-mask` runs into one three-condition report.

    Runs are merged rather than re-generated because a run only holds the conditions named on its own command
    line; the merge keeps every condition's rows exactly as they were produced.
    """
    root = Path(project_root).resolve()
    rows, conditions, manifests = [], [], {}
    for source in sources:
        directory = root / output_root / source
        subset = [row for row in read_jsonl(directory / 'masked.jsonl')]
        if not subset:
            raise ValueError(f'{directory} has no masked.jsonl rows')
        manifest = json.loads((directory / 'mask_summary.json').read_text(encoding='utf-8'))['manifest']
        manifests[source] = {'conditions': manifest['conditions'], 'samples': manifest['samples'],
                             'temperature': manifest['temperature'], 'max_new_tokens': manifest['max_new_tokens'],
                             'category': manifest['category'], 'strength': manifest['strength']}
        for condition in manifest['conditions']:
            if condition not in conditions:
                conditions.append(condition)
        rows.extend(subset)
    merged = {'created_at': utc_now(), 'run_id': run_id, 'sources': manifests, 'conditions': conditions}
    materials_by_id = None
    if materials_path:
        materials_by_id = {m['material_id']: m for m in read_jsonl(Path(materials_path))}
        merged['materials_path'] = str(materials_path)
    summary = _summarize_ablations(run_id, rows, merged, materials_by_id)
    summary['manifest']['category'] = next(iter(manifests.values()))['category']
    summary['manifest']['samples'] = next(iter(manifests.values()))['samples']
    summary['manifest']['temperature'] = next(iter(manifests.values()))['temperature']
    summary['manifest']['max_new_tokens'] = next(iter(manifests.values()))['max_new_tokens']
    summary['manifest']['strength'] = next(iter(manifests.values()))['strength']
    out = root / output_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'masked.jsonl', rows)
    write_json(out / 'mask_summary.json', summary)
    header = title or f'# Copy-source attention blocking {run_id}'
    note = ('Each condition is a generation run: `control` has no bias, `block_text_digits` forbids the '
            'generated positions from attending to the context numeral tokens, and `block_text_span` forbids '
            'them from attending to the whole text span. `has_number` can only come from the context text on '
            'these materials.')
    body = _render_ablation(summary, note).split('\n', 2)[2]
    (out / 'mask_report.md').write_text(header + '\n\n' + body, encoding='utf-8')
    return summary
