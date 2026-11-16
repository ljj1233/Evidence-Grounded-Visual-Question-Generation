"""White-box capture for the probing route (P0): hidden states of a local Qwen3-VL at three positions of question generation.

For one material (image + text) the fixed neutral prompt (`plain-v2` by default; the template is a parameter) is run with greedy decoding at batch size 1. Per layer (index 0 = embeddings;
the LAST entry depends on the transformers version, verified by hook: 4.57.6 = output of the last block BEFORE the final norm, >=5.22 = AFTER the
final norm; entries 1..n-1 are block outputs and agree across versions up to bf16 noise) the module saves
  prompt_end    last prompt position (the state that predicts the first question token)
  early         states of the first K generated question tokens (state *after* reading token i, i = 0..K-1)
  question_end  state of the last question token (the state that predicts the end-of-turn token)
plus token-span metadata (image tokens, text tokens, user turn, generation prompt). Index arithmetic and span location are
pure functions so they can be tested without a GPU; torch/transformers are imported lazily.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from ..core.records import read_jsonl, sha256_file, utc_now, write_json
from ..core.prompts import PromptSet, prompt_path

VERSION = 'whitebox-capture-v1'
TEMPLATE = 'plain-v2'


# ---------- pure helpers (no torch) ----------

def capture_plan(n_generated, ended, k):
    """Where the three positions live inside `generate(..., output_hidden_states=True)` output.

    hidden_states[step]: step 0 holds the whole prompt (read its last position); step s >= 1 holds the single token generated
    at step s-1, i.e. generated token i is visible at step i+1. The final generated token is never fed back, so when the
    question ends in the end-of-turn token (`ended`) every question token is visible, and when it was cut by the length limit
    the last token is not.
    """
    if n_generated < 0 or k < 1:
        raise ValueError('invalid capture arguments')
    n_question = n_generated - 1 if ended else n_generated
    visible = min(n_question, max(n_generated - 1, 0))      # question tokens whose state exists
    early = [(i + 1, 0) for i in range(min(k, visible))]
    return {'prompt_end': (0, -1), 'early': early,
            'question_end': (visible, 0) if visible else None,
            'n_question_tokens': n_question, 'n_visible_question_tokens': visible, 'complete': bool(ended)}


def token_span(offsets, start, end):
    """Half-open token index range whose character offsets overlap [start, end)."""
    hit = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not hit:
        raise ValueError('no token overlaps the character range')
    if hit[-1] - hit[0] + 1 != len(hit):
        raise ValueError('token range not contiguous')
    return hit[0], hit[-1] + 1


def locate_spans(ids, offsets, rendered, text, *, image_id, vision_start, vision_end):
    """Named token spans [start, end) of one rendered chat prompt. `rendered` is the decoded prompt, `offsets` its token offsets."""
    spans = {'n_prompt_tokens': len(ids)}
    pads = [i for i, t in enumerate(ids) if t == image_id]
    if pads:
        if pads[-1] - pads[0] + 1 != len(pads):
            raise ValueError('image tokens not contiguous')
        if ids[pads[0] - 1] != vision_start or ids[pads[-1] + 1] != vision_end:
            raise ValueError('image tokens not wrapped by vision markers')
        spans['image'] = (pads[0], pads[-1] + 1)
        spans['image_block'] = (pads[0] - 1, pads[-1] + 2)
    block = f'<T>{text}</T>'
    at = rendered.find(block)
    if at < 0 or rendered.find(block, at + 1) >= 0:
        raise ValueError('material text not found exactly once in the rendered prompt')
    spans['text'] = token_span(offsets, at + 3, at + 3 + len(text))
    spans['text_block'] = token_span(offsets, at, at + len(block))
    head = '<|im_start|>user\n'
    user_at = rendered.find(head)
    user_end = rendered.find('<|im_end|>', user_at)
    if user_at < 0 or user_end < 0:
        raise ValueError('user turn not found')
    spans['user'] = token_span(offsets, user_at + len(head), user_end)
    gen_at = rendered.rfind('<|im_start|>')
    spans['generation_prompt'] = (token_span(offsets, gen_at, gen_at + 1)[0], len(ids))
    return {k: list(v) if isinstance(v, tuple) else v for k, v in spans.items()}


def forward_extras(inputs, n_appended):
    """Model inputs other than ids/mask for a forward pass over prompt + `n_appended` extra text tokens. transformers>=5 also returns
    `mm_token_type_ids` (1 = image token) as long as the prompt only; appended text tokens get type 0."""
    import torch
    extra = {k: v for k, v in inputs.items() if k not in ('input_ids', 'attention_mask')}
    if 'mm_token_type_ids' in extra:
        types = extra['mm_token_type_ids']
        extra['mm_token_type_ids'] = torch.cat([types, torch.zeros(types.shape[0], n_appended, dtype=types.dtype, device=types.device)], dim=1)
    return extra


def has_question_states(row):
    """True when the capture holds early-token and question-end states (an empty question has none)."""
    return bool(row['plan']['early']) and bool(row['plan']['question_end'])


def compare_to_reference(text, reference):
    """Exact comparison of a regenerated question with the baseline's raw output."""
    if text == reference:
        return {'match': 'exact', 'first_diff_char': None}
    if text.strip() == reference.strip():
        return {'match': 'whitespace_only', 'first_diff_char': None}
    first = next((n for n, (a, b) in enumerate(zip(text, reference)) if a != b), min(len(text), len(reference)))
    return {'match': 'different', 'first_diff_char': first}


def gpu_memory_mib():
    """Used memory of the physical card this process is limited to (first CUDA_VISIBLE_DEVICES entry), or None."""
    index = (os.environ.get('CUDA_VISIBLE_DEVICES') or '').split(',')[0].strip()
    if not index.isdigit():
        return None
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', index],
                             capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        return int(out)
    except Exception:
        return None


# ---------- GPU part ----------

class Capturer:
    def __init__(self, backend, k=4):
        self.backend, self.k = backend, k
        self._ids = self._fast = None

    def _special_ids(self):
        if self._ids is None:
            cfg = self.backend._model.config
            tok = self.backend._processor.tokenizer
            self._ids = {'image_id': cfg.image_token_id, 'vision_start': cfg.vision_start_token_id,
                         'vision_end': cfg.vision_end_token_id, 'eos': {tok.convert_tokens_to_ids('<|im_end|>'),
                         tok.convert_tokens_to_ids('<|endoftext|>')}}
        return self._ids

    def spans(self, ids, text):
        tok = self.backend._processor.tokenizer
        special = self._special_ids()
        rendered = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if self._fast is None:      # the backend's processor uses the slow tokenizer, which has no offsets
            from transformers import AutoTokenizer
            self._fast = AutoTokenizer.from_pretrained(str(self.backend.path), use_fast=True, local_files_only=True)
        enc = self._fast(rendered, add_special_tokens=False, return_offsets_mapping=True)
        if list(enc['input_ids']) != list(ids):
            raise ValueError('prompt does not round-trip through the tokenizer')
        return locate_spans(ids, [tuple(o) for o in enc['offset_mapping']], rendered, text, image_id=special['image_id'],
                            vision_start=special['vision_start'], vision_end=special['vision_end'])

    def prepare(self, request, text):
        """Encode one material prompt once. Every sampled candidate of that material reuses this exact encoding: the image and
        prompt are identical, so CPU image preprocessing and tokenization are not repeated (generation still runs at batch size 1)."""
        inputs = self.backend.encode([request])
        ids = inputs['input_ids'][0].tolist()
        return inputs, ids, self.spans(ids, text)

    def capture(self, request, text, *, check=False, sample=None, intervention=None, prepared=None):
        """sample=None: greedy. sample={'temperature','top_p','top_k','seed'}: seeded sampling (batch size 1, so the seed fixes the draw).
        `prepared` is the output of `prepare` for this material; reusing it does not change any tensor handed to generate."""
        import torch
        backend, model = self.backend, self.backend._model
        inputs, ids, spans = prepared if prepared is not None else self.prepare(request, text)
        device = model.device
        on_cuda = device.type == 'cuda'
        if on_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        if sample:
            torch.manual_seed(sample['seed'])
            decoding = {'do_sample': True, 'temperature': sample['temperature'], 'top_p': sample['top_p'], 'top_k': sample['top_k']}
        else:
            decoding = {'do_sample': False}
        began = time.monotonic()
        from contextlib import nullcontext
        with torch.inference_mode(), (intervention(model, len(ids)) if intervention else nullcontext()):
            out = model.generate(**inputs, num_beams=1, max_new_tokens=backend.max_new_tokens, **decoding,
                                 max_time=backend.max_time, output_hidden_states=True, return_dict_in_generate=True)
            if on_cuda:
                torch.cuda.synchronize(device)
            seconds = time.monotonic() - began
            seq = out.sequences[0, len(ids):].tolist()
            eos = self._special_ids()['eos']
            stop = next((n for n, t in enumerate(seq) if t in eos), None)
            body = seq if stop is None else seq[:stop + 1]
            ended = stop is not None
            if len(out.hidden_states) < len(body):        # more steps than tokens is fine (Qwen3.5 continues after <|im_end|>); fewer is a bug
                raise ValueError(f'hidden-state steps {len(out.hidden_states)} < generated tokens {len(body)}')
            plan = capture_plan(len(body), ended, self.k)

            def pick(step, pos):
                return torch.stack([layer[0, pos] for layer in out.hidden_states[step]]).cpu().contiguous()
            states = {'prompt_end': pick(*plan['prompt_end'])}
            if plan['early']:
                states['early'] = torch.stack([pick(*p) for p in plan['early']])
            if plan['question_end']:
                states['question_end'] = pick(*plan['question_end'])
            check_result = self._teacher_forced_check(inputs, body, plan, states) if check else None
            peak = torch.cuda.max_memory_allocated(device) if on_cuda else 0
        question = backend._processor.tokenizer.decode(body, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        meta = {'question': question, 'generated_token_ids': body, 'finish': 'stop' if ended else 'length_or_timeout',
                'plan': {**plan, 'early': [list(p) for p in plan['early']],
                         'question_end': list(plan['question_end']) if plan['question_end'] else None,
                         'prompt_end': list(plan['prompt_end'])},
                'spans': spans, 'image_grid_thw': inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else None,
                'n_prompt_tokens': len(ids), 'n_generated_tokens': len(body), 'generation_seconds': round(seconds, 3),
                'tokens_per_second': round(len(body) / seconds, 2) if seconds else None,
                'peak_allocated_mib': round(peak / 2 ** 20, 1), 'teacher_forced_check': check_result}
        return states, meta

    def _teacher_forced_check(self, inputs, body, plan, states):
        """Re-run prompt + generated tokens in one forward pass; relative error of each captured state against it."""
        import torch
        model = self.backend._model
        width = inputs['input_ids'].shape[1]
        full = torch.cat([inputs['input_ids'], torch.tensor([body], device=inputs['input_ids'].device)], dim=1)
        extra = forward_extras(inputs, len(body))
        hs = model(input_ids=full, attention_mask=torch.ones_like(full), output_hidden_states=True, **extra).hidden_states
        ref = lambda pos: torch.stack([h[0, pos] for h in hs]).float().cpu()
        wanted = {'prompt_end': width - 1}
        if 'early' in states:
            wanted.update({f'early_{j}': width + j for j in range(states['early'].shape[0])})
        if 'question_end' in states:
            wanted['question_end'] = width + plan['n_visible_question_tokens'] - 1
        result = {}
        for name, pos in wanted.items():
            got = states['prompt_end'] if name == 'prompt_end' else states['question_end'] if name == 'question_end' else states['early'][int(name[6:])]
            rel = ((got.float() - ref(pos)).norm(dim=-1) / ref(pos).norm(dim=-1).clamp_min(1e-6))
            result[name] = {'max_rel_error': round(float(rel.max()), 5), 'mean_rel_error': round(float(rel.mean()), 5),
                            'worst_layer': int(rel.argmax())}
        return result


# ---------- run ----------

def run_capture(*, run_id, project_root='.', output_root='outputs', reference_run, model_path, device='cuda:0',
                limit=None, k=4, max_new_tokens=1024, check=True, template=TEMPLATE):
    from safetensors.torch import save_file
    from ..core.local_vision import QwenVLBackend, qwen_identity
    from ..core.prompts import PromptSet
    from ..core.providers import CompletionRequest
    root, out = Path(project_root).resolve(), Path(output_root).resolve() / run_id
    ref_dir = Path(output_root).resolve() / reference_run
    plan = read_jsonl(ref_dir / 'plan.jsonl')
    materials = {m['material_id']: m for m in read_jsonl(ref_dir / 'materials.jsonl')}
    reference = {r['item_id']: r['generation']['raw_output'] for r in read_jsonl(ref_dir / 'records.jsonl')}
    plan = plan[:limit] if limit else plan
    (out / 'states').mkdir(parents=True, exist_ok=True)
    prompts = PromptSet(root / 'prompts', names=(template,))
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens)
    identity = qwen_identity(backend, 'local-qwen3-vl-whitebox-v1', capture_version=VERSION, batch_size=1, capture_k=k,
                             prompt_template=template, prompt_sha256=sha256_file(prompt_path(root / 'prompts', template)),
                             hidden_state_layers='embeddings + one entry per block; the last entry is AFTER the final norm in transformers>=5.22 (verified: equals norm(last block output)); earlier 4.57.6 runs stored it BEFORE the norm',
                             state_dtype='bfloat16', reference_run=reference_run)
    identity_path = out / 'model_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('model/capture settings changed; use a new run_id')
    write_json(identity_path, identity)
    done = {r['material_id'] for r in read_jsonl(out / 'captures.jsonl')}
    gpu = {'before_load_mib': gpu_memory_mib()}
    capturer = Capturer(backend, k=k)
    backend._load()
    gpu['after_load_mib'] = gpu_memory_mib()
    for slot in plan:
        mid = slot['material_id']
        if mid in done:
            continue
        material = materials[mid]
        request = CompletionRequest(task='generate_question', prompt=prompts.render(template, T=material['text']),
                                    image_path=str(root / material['image_path']))
        states, meta = capturer.capture(request, material['text'], check=check)
        path = out / 'states' / f'{mid}.safetensors'
        save_file(states, str(path))
        row = {'material_id': mid, 'item_id': slot['item_id'], 'created_at': utc_now(), 'states_file': f'states/{mid}.safetensors',
               'state_shapes': {n: list(t.shape) for n, t in states.items()}, 'reference': compare_to_reference(meta['question'], reference[slot['item_id']]),
               **meta}
        with (out / 'captures.jsonl').open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    gpu['after_run_mib'] = gpu_memory_mib()
    write_json(out / 'gpu.json', {**gpu, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
    return report_capture(run_id=run_id, output_root=output_root)


def report_capture(*, run_id, output_root='outputs'):
    out = Path(output_root).resolve() / run_id
    rows = read_jsonl(out / 'captures.jsonl')
    gpu = json.loads((out / 'gpu.json').read_text()) if (out / 'gpu.json').exists() else {}
    exact = sum(r['reference']['match'] == 'exact' for r in rows)
    summary = {'run_id': run_id, 'materials': len(rows), 'exact_match': exact,
               'whitespace_only': sum(r['reference']['match'] == 'whitespace_only' for r in rows),
               'different': [r['material_id'] for r in rows if r['reference']['match'] == 'different'],
               'complete_questions': sum(r['plan']['complete'] for r in rows),
               'peak_allocated_mib_max': max((r['peak_allocated_mib'] for r in rows), default=None),
               'tokens_per_second_median': sorted(r['tokens_per_second'] for r in rows)[len(rows) // 2] if rows else None,
               'gpu': gpu}
    write_json(out / 'summary.json', summary)
    return summary


def compare_captures(*, run_a, run_b, project_root='.', output_root='outputs'):
    """Version check: same materials captured under two software versions. Question equality, prompt-length equality and the relative
    error of the captured states per layer (prompt_end always; question_end only where the question tokens are identical)."""
    import torch
    from safetensors.torch import load_file
    base = Path(project_root).resolve() / output_root
    a = {r['material_id']: r for r in read_jsonl(base / run_a / 'captures.jsonl')}
    b = {r['material_id']: r for r in read_jsonl(base / run_b / 'captures.jsonl')}
    rows, err = [], {'prompt_end': [], 'question_end': []}
    for mid in sorted(set(a) & set(b)):
        ra, rb = a[mid], b[mid]
        same_prompt = ra['n_prompt_tokens'] == rb['n_prompt_tokens'] and ra['image_grid_thw'] == rb['image_grid_thw']
        same_q = ra['generated_token_ids'] == rb['generated_token_ids']
        row = {'material_id': mid, 'same_prompt_tokens': same_prompt, 'prompt_tokens': [ra['n_prompt_tokens'], rb['n_prompt_tokens']],
               'same_question_tokens': same_q, 'question_a': ra['question'], 'question_b': rb['question']}
        ta, tb = load_file(str(base / run_a / ra['states_file'])), load_file(str(base / run_b / rb['states_file']))
        for name in ('prompt_end', 'question_end'):
            if name == 'question_end' and not (same_q and same_prompt):
                continue
            if name == 'prompt_end' and not same_prompt:
                continue
            x, y = ta[name].float(), tb[name].float()
            rel = ((x - y).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6))
            row[f'{name}_max_rel'], row[f'{name}_mean_rel'] = round(float(rel.max()), 5), round(float(rel.mean()), 5)
            err[name].append(rel)
        rows.append(row)
    summary = {'run_a': run_a, 'run_b': run_b, 'materials': len(rows), 'same_prompt_tokens': sum(r['same_prompt_tokens'] for r in rows),
               'same_question_tokens': sum(r['same_question_tokens'] for r in rows),
               'mean_rel_error_per_layer': {k: [round(float(x), 5) for x in torch.stack(v).mean(0)] for k, v in err.items() if v}, 'rows': rows}
    write_json(base / run_b / f'compare_to_{run_a}.json', summary)
    return {k: v for k, v in summary.items() if k not in ('rows', 'mean_rel_error_per_layer')}
