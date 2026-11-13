"""Local vision-language inference behind the provider contract (model, identity, complete(request)).

Many threads call `complete` concurrently; one worker thread owns the GPU and groups waiting requests into a batch
(greedy decoding, left padding), so a single card is kept busy without vLLM. A batch that fails is retried
request by request, so one bad input cannot take the others down; a request that still fails raises ProviderError.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

from .providers import Completion, ProviderError
from .records import sha256_file


@dataclass(frozen=True)
class LocalOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish: str = 'stop'


class BatchingProvider:
    def __init__(self, backend, *, model, identity, max_batch=8, wait_seconds=0.05):
        if max_batch < 1 or wait_seconds < 0:
            raise ValueError('invalid batching limits')
        self.backend, self.model, self.identity = backend, model, identity
        self.max_batch, self.wait_seconds = max_batch, wait_seconds
        self.batch_sizes = []
        self._queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name='local-vision-worker').start()

    def complete(self, request):
        future = Future()
        self._queue.put((request, future))
        return future.result()

    def _loop(self):
        while True:
            batch = [self._queue.get()]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                try:
                    batch.append(self._queue.get(timeout=remaining) if remaining > 0 else self._queue.get_nowait())
                except queue.Empty:
                    break
            self._run(batch)

    def _run(self, batch):
        began = time.monotonic()
        try:
            outputs = self.backend.generate([request for request, _ in batch])
            if len(outputs) != len(batch):
                raise ProviderError('local_batch_size_mismatch')
        except Exception:
            if len(batch) > 1:                      # isolate the failing input instead of failing the whole batch
                for item in batch:
                    self._run([item])
                return
            batch[0][1].set_exception(ProviderError('local_inference_failure'))
            return
        self.batch_sizes.append(len(batch))
        elapsed = round(time.monotonic() - began, 3)
        for (request, future), out in zip(batch, outputs):
            future.set_result(Completion(out.text, self.model, {
                'prompt_tokens': out.prompt_tokens, 'completion_tokens': out.completion_tokens,
                'total_tokens': out.prompt_tokens + out.completion_tokens, 'network_requests': 0, 'local_requests': 1,
                'cache_hit': False, 'cost': None, 'elapsed_seconds': elapsed, 'batch_size': len(batch)}, out.finish))


class QwenVLBackend:
    """Hugging Face Qwen3-VL, bf16, sdpa, greedy. Weights are loaded on first use."""

    def __init__(self, model_path, *, device='cuda:0', max_new_tokens=1024, max_image_pixels=1048576, max_time=900):
        self.path = Path(model_path).expanduser().resolve()
        self.device, self.max_new_tokens, self.max_image_pixels, self.max_time = device, max_new_tokens, max_image_pixels, max_time
        self._model = self._processor = None

    def fingerprint(self):
        """Config/tokenizer files hashed, weight shards by size plus the index hash (full 16 GB hashing is not needed per run)."""
        small = {p.name: sha256_file(p) for p in sorted(self.path.iterdir()) if p.is_file() and p.suffix in ('.json', '.jinja', '.txt')}
        shards = {p.name: p.stat().st_size for p in sorted(self.path.glob('*.safetensors'))}
        if not shards or not (self.path / 'config.json').is_file():
            raise ValueError('local model files missing')
        return {'path': str(self.path), 'files': small, 'weight_shard_bytes': shards}

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        torch.manual_seed(42)
        self._processor = AutoProcessor.from_pretrained(str(self.path), local_files_only=True, use_fast=False, max_pixels=self.max_image_pixels)
        self._processor.tokenizer.padding_side = 'left'
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.path), local_files_only=True, dtype=torch.bfloat16, device_map={'': self.device}, attn_implementation='sdpa')
        self._model.eval()

    def generate(self, requests):
        import torch
        from PIL import Image
        self._load()
        conversations, opened = [], []
        try:
            for request in requests:
                content = []
                if request.image_path:
                    image = Image.open(request.image_path)
                    opened.append(image)
                    content.append({'type': 'image', 'image': image.convert('RGB')})
                content.append({'type': 'text', 'text': request.prompt})
                conversations.append([{'role': 'user', 'content': content}])
            inputs = self._processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True,
                                                         return_dict=True, return_tensors='pt', padding=True).to(self._model.device)
            width = inputs['input_ids'].shape[1]
            with torch.inference_mode():
                generated = self._model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens, max_time=self.max_time)
        finally:
            for image in opened:
                image.close()
        pad = self._processor.tokenizer.pad_token_id
        eos = self._model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        outputs = []
        for row, prompt_row in zip(generated[:, width:], inputs['attention_mask']):
            ids = [int(t) for t in row]
            stop = next((n for n, t in enumerate(ids) if t in eos), None)
            body = ids if stop is None else ids[:stop + 1]
            text = self._processor.tokenizer.decode(body, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            outputs.append(LocalOutput(text, int(prompt_row.sum()), len(body), 'stop' if stop is not None else 'length_or_timeout'))
        return outputs


def local_provider(model_path, *, device='cuda:0', max_batch=8, max_new_tokens=1024, max_image_pixels=1048576):
    """A provider for `simple-baseline-run` that runs a local Qwen3-VL model."""
    import torch
    import transformers
    backend = QwenVLBackend(model_path, device=device, max_new_tokens=max_new_tokens, max_image_pixels=max_image_pixels)
    identity = {'adapter': 'local-qwen3-vl-batched-v1', 'model': backend.path.name, 'model_fingerprint': backend.fingerprint(),
                'device_kind': 'cuda', 'dtype': 'bfloat16', 'attention': 'sdpa', 'do_sample': False, 'num_beams': 1,
                'max_new_tokens': max_new_tokens, 'max_image_pixels': max_image_pixels, 'seed': 42,
                'batch_composition_can_change_low_order_bits': True, 'torch': torch.__version__, 'transformers': transformers.__version__}
    return BatchingProvider(backend, model=backend.path.name, identity=identity, max_batch=max_batch)
