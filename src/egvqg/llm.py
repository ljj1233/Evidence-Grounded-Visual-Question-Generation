"""One cached API call path for the VQAG modules (label repair, fact splitting, filtering).

Cache key = sha256(model, prompt, image content id), stored under `outputs/cache/vqag/`. The transport is the
existing `providers.JudgeProvider` (which also keeps its own cache and ledger); this wrapper adds the coarser
model+prompt+image key, a planned/actual call counter and a hard `--max-calls` cap. Dry runs never build a
provider and never read credentials.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from .records import append_jsonl, read_jsonl, sha256_file, sha256_text, utc_now, write_json


def json_object(text):
    """First JSON object in a model reply (tolerates markdown fences and leading prose); None if unparsable."""
    if not isinstance(text, str):
        return None
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S)
    body = fenced.group(1) if fenced else None
    if body is None:
        start, end = text.find('{'), text.rfind('}')
        body = text[start:end + 1] if 0 <= start < end else None
    if not body:
        return None
    try:
        value = json.loads(body)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


class CachedCaller:
    def __init__(self, *, project_root='.', run_id=None, output_root='outputs', dry_run=False, max_calls=0, env_prefix='LLM_JUDGE',
                 max_tokens=4096, timeout=180):
        self.root = Path(project_root).resolve()
        self.cache_dir = self.root / output_root / 'cache/vqag'
        self.run_dir = self.root / output_root / run_id if run_id else None
        self.dry_run, self.max_calls, self.env_prefix = dry_run, int(max_calls or 0), env_prefix
        self.max_tokens, self.timeout = int(max_tokens), float(timeout)
        self.planned = self.actual = self.cache_hits = 0
        self.examples = []
        self._providers = {}
        self._lock = threading.Lock()
        self._key_locks = {}
        self._key_locks_guard = threading.Lock()
        if self.run_dir and not dry_run:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model, prompt, image_path):
        image_id = sha256_file(Path(image_path)) if image_path else None
        return sha256_text(json.dumps({'model': model, 'prompt': prompt, 'image_id': image_id,
                                       'max_tokens': self.max_tokens}, sort_keys=True)), image_id

    def _key_lock(self, key):
        with self._key_locks_guard:
            return self._key_locks.setdefault(key, threading.Lock())

    def call(self, *, model, prompt, image_path=None, task='assess_quality'):
        """Return {'text', 'model', 'cache_hit'} or None in dry-run. Raises when the call cap would be exceeded.
        Thread safe: identical requests are serialised by their cache key, different keys may run concurrently."""
        key, image_id = self._key(model, prompt, image_path)
        if self.dry_run:
            with self._lock:
                self.planned += 1
                if len(self.examples) < 2:
                    self.examples.append({'model': model, 'image': bool(image_path), 'prompt': prompt[:600]})
            return None
        from .providers import CompletionRequest, JudgeProvider
        with self._key_lock(key):
            path = self.cache_dir / f'{key}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get('key') != key or sha256_text(saved.get('text', '')) != saved.get('text_sha256'):
                    raise ValueError('vqag cache integrity mismatch')
                with self._lock:
                    self.cache_hits += 1
                return {'text': saved['text'], 'model': saved['model'], 'cache_hit': True}
            with self._lock:
                if self.max_calls and self.actual >= self.max_calls:
                    raise ValueError(f'call cap reached: {self.actual} >= {self.max_calls}')
                self.actual += 1                       # reserve before sending: a lost response may still be billed
                provider = self._providers.get(model)
            if provider is None:
                provider = JudgeProvider.from_env(root=self.root, run_dir=self.run_dir or self.cache_dir.parent,
                                                  max_requests=100000, max_tokens=self.max_tokens, visual=True, model=model,
                                                  env_prefix=self.env_prefix, timeout=self.timeout)
                with self._lock:
                    provider = self._providers.setdefault(model, provider)
            completion = provider.complete(CompletionRequest(task=task, prompt=prompt, image_path=image_path, text=None))
            usage = completion.usage or {}
            record = {'key': key, 'text': completion.text, 'text_sha256': sha256_text(completion.text),
                      'model': completion.model, 'image_id': image_id, 'usage': usage}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            write_json(path, record)
            if self.run_dir:
                append_jsonl(self.run_dir / 'calls_vqag.jsonl', {'event': 'completed', 'kept_cache_key': key,
                                                                 'model': model, 'image': bool(image_path),
                                                                 'usage': usage, 'created_at': utc_now()})
            return {'text': completion.text, 'model': completion.model, 'cache_hit': False}

    def summary(self):
        return {'planned_calls': self.planned, 'api_calls': self.actual, 'cache_hits': self.cache_hits,
                'max_calls': self.max_calls, 'dry_run': self.dry_run,
                'note': 'dry-run counts are upper estimates of distinct calls; cached requests cost nothing'}


def read_ids(path):
    """Material-id list file: one id per line, blank lines and # comments ignored."""
    return {line.split('#')[0].strip() for line in Path(path).read_text().splitlines() if line.split('#')[0].strip()}


def read_jsonl_safe(path):
    return read_jsonl(path) if Path(path).exists() else []
