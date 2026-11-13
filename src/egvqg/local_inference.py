"""Single-GPU, offline Hugging Face inference through the existing call contract."""
from __future__ import annotations

import json
import time
from importlib.metadata import version
from pathlib import Path

from .providers import Completion, JudgeProvider, ProviderError, RequestBudgetExceeded
from .records import append_jsonl, canonical_json, read_jsonl, sha256_file, sha256_text, utc_now, write_json


def model_fingerprint(model_path):
    path = Path(model_path).expanduser().resolve()
    if not (path / 'config.json').is_file():
        raise ValueError('local model config.json missing')
    configs = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in ('.json', '.jinja'))
    weights = sorted(path.glob('*.safetensors'))
    if not weights:
        raise ValueError('local safetensors weights missing')
    # Hash the shared files in place; no model copy and no network download.
    return {'path': str(path), 'files': {p.name: sha256_file(p) for p in configs + weights}}


class LocalVisionProvider:
    def __init__(self, *, model_path, run_dir, device='cuda:4', max_calls=30,
                 max_new_tokens=1024, timeout=180, max_image_pixels=1048576):
        if max_calls < 1 or max_new_tokens < 1 or not 0 < timeout <= 600:
            raise ValueError('invalid local inference limits')
        self.fingerprint = model_fingerprint(model_path)
        config = json.loads((Path(self.fingerprint['path']) / 'config.json').read_text())
        if config.get('model_type') != 'qwen3_vl':
            raise ValueError('this adapter currently supports Qwen3-VL only')
        self.model = Path(self.fingerprint['path']).name
        self.run_dir, self.device = Path(run_dir), device
        self.ledger_path = self.run_dir / 'local_requests.jsonl'
        self.max_calls, self.max_new_tokens, self.timeout = max_calls, max_new_tokens, timeout
        self.max_image_pixels = max_image_pixels
        self.identity = {'adapter': 'local-qwen3-vl-transformers-v1', 'model': self.model,
            'model_fingerprint': self.fingerprint, 'device': device, 'dtype': 'bfloat16',
            'attention': 'sdpa', 'do_sample': False, 'num_beams': 1, 'seed': 42,
            'max_new_tokens': max_new_tokens, 'timeout_seconds': timeout,
            'max_image_pixels': max_image_pixels, 'processor_use_fast': False, 'local_files_only': True,
            'torch': version('torch'), 'transformers': version('transformers')}
        self._model = self._processor = None
        self.load_seconds = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        begin = time.monotonic()
        if not torch.cuda.is_available():
            raise ProviderError('local_cuda_unavailable')
        torch.manual_seed(42)
        self._processor = AutoProcessor.from_pretrained(self.fingerprint['path'], local_files_only=True,
                            use_fast=False, max_pixels=self.max_image_pixels)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.fingerprint['path'],
            local_files_only=True, dtype=torch.bfloat16, device_map={'': self.device}, attn_implementation='sdpa')
        self._model.eval()
        self.load_seconds = round(time.monotonic() - begin, 3)
        runtime = {'identity': self.identity,
            'load_seconds': self.load_seconds, 'gpu_name': torch.cuda.get_device_name(self.device),
            'cuda': torch.version.cuda, 'created_at': utc_now()}
        write_json(self.run_dir / 'local_runtime.json', runtime)
        append_jsonl(self.ledger_path, {'event': 'model_loaded', **runtime})

    def complete(self, request):
        started = [r for r in read_jsonl(self.ledger_path) if r['event'] == 'started']
        if len(started) >= self.max_calls:
            raise RequestBudgetExceeded('cumulative local model call budget exhausted')
        image_hash = sha256_file(Path(request.image_path)) if request.image_path else None
        image_hashes = [] if image_hash is None else [image_hash]
        signature = sha256_text(canonical_json({'identity': self.identity, 'prompt': request.prompt,
            'images': image_hashes, 'condition': request.condition, 'metadata': dict(request.metadata)}))
        # A local call interrupted after durable intent is not silently re-generated.
        prior = next((r for r in started if r['signature'] == signature), None)
        if prior:
            raise ProviderError('interrupted_local_call_not_resampled')
        count = len(started) + 1
        append_jsonl(self.ledger_path, {'event': 'started', 'request_no': count,
            'caller_call_id': request.metadata.get('caller_call_id'), 'signature': signature,
            'image_sha256': image_hash, 'created_at': utc_now()})
        begin = time.monotonic()
        try:
            self._load()
            import torch
            from PIL import Image
            content = []
            if request.image_path:
                with Image.open(request.image_path) as image:
                    content.append({'type': 'image', 'image': image.convert('RGB').copy()})
            content.append({'type': 'text', 'text': request.prompt})
            messages = [{'role': 'user', 'content': content}]
            inputs = self._processor.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt').to(self._model.device)
            input_tokens = inputs.input_ids.shape[1]
            with torch.inference_mode():
                output = self._model.generate(**inputs, do_sample=False, num_beams=1,
                    max_new_tokens=self.max_new_tokens, max_time=self.timeout)
            generated = output[0, input_tokens:]
            raw = self._processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            stop_ids = self._model.generation_config.eos_token_id
            stop_ids = stop_ids if isinstance(stop_ids, list) else [stop_ids]
            finish = 'stop' if len(generated) and int(generated[-1]) in stop_ids else 'length_or_timeout'
            usage = {'prompt_tokens': input_tokens, 'completion_tokens': len(generated),
                'total_tokens': input_tokens + len(generated), 'network_requests': 0,
                'local_requests': 1, 'cache_hit': False, 'cost': None,
                'elapsed_seconds': round(time.monotonic() - begin, 3),
                'input_audit': {'image_transmitted': bool(request.image_path), 'image_sha256': image_hash,
                    'history_messages': 0, 'role_model_identity': self.identity}}
        except (RuntimeError, OSError, ValueError) as exc:
            # Avoid recording arbitrary exception text that could contain user environment data.
            error = 'local_inference_failure:' + type(exc).__name__
            append_jsonl(self.ledger_path, {'event': 'failed', 'request_no': count, 'error': error,
                         'created_at': utc_now()})
            raise ProviderError(error) from None
        append_jsonl(self.ledger_path, {'event': 'completed', 'request_no': count,
            'usage': usage, 'finish_reason': finish, 'created_at': utc_now()})
        return Completion(raw, self.model, usage, finish)


class LocalExperimentProvider:
    """The local model generates and answers; the independent gateway model judges."""
    def __init__(self, local, judge):
        self.local, self.judge = local, judge
        self.model = 'local-generator-and-solver / remote-judge'
        self.identity = {'adapter': 'explicit-role-routing-v1',
            'generation': local.identity, 'condition_answering': local.identity,
            'reference_and_scoring': judge.identity}
        judge.max_attempts = 1

    def remote_requests_for_call(self, call_id):
        return sum(r.get('event') == 'started' and r.get('caller_call_id') == call_id
                   for r in read_jsonl(self.judge.ledger_path))

    def recover_completion(self, request):
        role = request.metadata['role']
        if role == 'generation' or role.startswith('answer_'):
            raise ProviderError('interrupted_local_call_not_resampled')
        return self.judge.complete(request, cache_only=True)

    def complete(self, request):
        role = request.metadata['role']
        provider = self.local if role == 'generation' or role.startswith('answer_') else self.judge
        return provider.complete(request)


def run_local_experiment(*, run_id, model_path, device='cuda:4', max_local_calls=30,
                         max_requests=80, max_new_tokens=1024, stages=None,
                         project_root='.', output_root='outputs', dry_run=False, max_steps=None):
    from .experiments import run_experiment
    root = Path(project_root).resolve(); out = root / output_root / run_id
    stages = stages or ('generation', 'reference', 'answers', 'scores')
    if dry_run:
        result = run_experiment(run_id=run_id, project_root=root, output_root=output_root,
                                stages=stages, max_requests=max_requests, dry_run=True)
        local_count = sum(n for step, n in result['pending_by_step'].items()
                          if step == 'generation' or step.startswith('answer_'))
        remote_count = result['max_logical_calls'] - local_count
        config = json.loads((out / 'manifest.json').read_text())['config']
        result.update(max_local_model_calls=local_count, max_gateway_logical_calls=remote_count,
                      max_http_with_technical_retries=min(max_requests, remote_count * config['max_technical_attempts']),
                      local_model_cumulative_cap=max_local_calls,
                      note='Local generation/answering are not HTTP calls; interrupted local inference is not resampled.')
        return result
    config = json.loads((out / 'manifest.json').read_text())['config']
    local = LocalVisionProvider(model_path=model_path, run_dir=out, device=device,
                               max_calls=max_local_calls, max_new_tokens=max_new_tokens)
    judge = JudgeProvider.from_env(root=root, run_dir=out, model='gpt-6-luna',
                 max_requests=max_requests, max_tokens=config['max_output_tokens'], visual=True)
    routed = LocalExperimentProvider(local, judge)
    return run_experiment(run_id=run_id, project_root=root, output_root=output_root,
                          stages=stages, max_requests=max_requests, provider=routed, max_steps=max_steps)
