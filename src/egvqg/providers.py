"""Replaceable model interface plus the scripted mock provider used in Stage A."""

from __future__ import annotations

import json
import base64
import hashlib
import http.client
import os
import socket
import time
from urllib.parse import urlsplit, urlunsplit
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

from .records import CallRecord, JsonlTable, append_jsonl, canonical_json, read_jsonl, sha256_text, utc_now, write_json


class ProviderError(RuntimeError):
    """Transport/API-level failure; may be retried under the configured policy."""


class RequestBudgetExceeded(RuntimeError):
    """Stop the run without recording a missing score as a completed judgement."""


class OutputParseError(ValueError):
    """The provider returned successfully but its output is not usable."""


@dataclass(frozen=True)
class CompletionRequest:
    task: str
    prompt: str
    question: str | None = None
    text: str | None = None
    image_path: str | None = None
    condition: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"


class Provider(Protocol):
    def complete(self, request: CompletionRequest) -> Completion:
        ...


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        last_fence = text.rfind("```")
        if first_newline == -1 or last_fence <= first_newline:
            raise OutputParseError("malformed fenced output")
        text = text[first_newline + 1 : last_fence].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise OutputParseError("no JSON object found in model output")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise OutputParseError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(value, dict):
        raise OutputParseError("model output JSON is not an object")
    return value


JSON_REPAIRS = ('escape_invalid_backslashes',)


def parse_model_output(raw: str, output_format: str, repair: str | None = None):
    """Parse one completion; return (parsed, repair_record_or_None).

    Strict parsing is always tried first. The only supported repair is opt-in and
    purely syntactic: a backslash that does not start a legal JSON escape (typical
    for unescaped LaTeX such as ``\\(``) is doubled so the string keeps the literal
    characters the model wrote. The raw output is never modified by the caller, and
    truncated or otherwise malformed output still fails.
    """
    if output_format != 'json':
        return raw.strip(), None
    try:
        return parse_json_object(raw), None
    except OutputParseError as strict_error:
        if repair is None:
            raise
        if repair not in JSON_REPAIRS:
            raise ValueError(f'unknown json_repair: {repair}') from None
        import re
        count = 0
        def fix(match):
            nonlocal count
            follower = match.group(1)
            if follower in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't') or (
                    follower == 'u' and re.match(r'[0-9a-fA-F]{4}', raw[match.end():match.end() + 4])):
                return match.group(0)
            count += 1
            return '\\\\' + follower
        repaired = re.sub(r'\\(.?)', fix, raw, flags=re.S)
        if not count:
            raise
        value = parse_json_object(repaired)  # may raise: the failure then stands
        return value, {'kind': repair, 'replacements': count, 'strict_error': str(strict_error),
                       'raw_sha256': hashlib.sha256(raw.encode()).hexdigest()}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class CallRecorder:
    """Calls a provider and appends one audit record per request."""

    def __init__(
        self,
        table: JsonlTable,
        *,
        run_id: str,
        protocol_version: str,
        default_model: str,
    ):
        self.table = table
        self.run_id = run_id
        self.protocol_version = protocol_version
        self.default_model = default_model
        self._next_number = len(table) + 1

    @property
    def added_count(self) -> int:
        return self.table.added_count

    def _next_id(self) -> str:
        while True:
            candidate = f"call_{self._next_number:06d}"
            self._next_number += 1
            if self.table.get(candidate) is None:
                return candidate

    def complete(
        self,
        provider: Provider,
        request: CompletionRequest,
        *,
        role: str,
        material_id: str | None = None,
        question_id: str | None = None,
        model: str | None = None,
    ) -> Completion:
        prompt_sha = sha256_text(request.prompt)
        input_manifest = {
            "text_included": request.text is not None,
            "image_included": request.image_path is not None,
            "question_included": request.question is not None,
            "text_sha256": sha256_text(request.text) if request.text is not None else None,
            "image_path": request.image_path,
            "question_sha256": sha256_text(request.question) if request.question is not None else None,
            "condition": request.condition,
            "metadata_keys": sorted(request.metadata),
        }
        try:
            completion = provider.complete(request)
        except ProviderError as exc:
            self.table.add(
                CallRecord(
                    call_id=self._next_id(),
                    run_id=self.run_id,
                    protocol_version=self.protocol_version,
                    role=role,
                    task=request.task,
                    material_id=material_id,
                    question_id=question_id,
                    condition=request.condition,
                    model=model or self.default_model,
                    prompt_sha256=prompt_sha,
                    input_manifest=input_manifest,
                    status="technical_failure",
                    error=str(exc),
                    raw_output=None,
                    usage={},
                    created_at=utc_now(),
                ).to_dict()
            )
            raise
        self.table.add(
            CallRecord(
                call_id=self._next_id(),
                run_id=self.run_id,
                protocol_version=self.protocol_version,
                role=role,
                task=request.task,
                material_id=material_id,
                question_id=question_id,
                condition=request.condition,
                model=completion.model or model or self.default_model,
                prompt_sha256=prompt_sha,
                input_manifest={**input_manifest, **completion.usage.get('input_audit',
                    completion.usage.get('original_usage', {}).get('input_audit', {}))},
                status="ok",
                error=None,
                raw_output=completion.text,
                usage=dict(completion.usage),
                created_at=utc_now(),
            ).to_dict()
        )
        return completion


class MockProvider:
    """Deterministic scripted provider. It must never be reported as a real model."""

    model = "mock-scripted-v1"

    def __init__(self, script_path: Path):
        self.script_path = script_path
        self.script = json.loads(script_path.read_text(encoding="utf-8"))
        self.questions: dict[tuple[str, str], dict[str, Any]] = {}
        self.validity: dict[str, dict[str, Any]] = {}
        self.answers: dict[str, dict[str, Any]] = {}
        self.responses: dict[str, dict[str, dict[str, Any]]] = {}
        self.evidence: dict[str, dict[str, Any]] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        materials = self.script.get("materials")
        if not isinstance(materials, dict):
            raise ValueError(f"mock script {self.script_path} has no materials object")
        for material_id, material in materials.items():
            self.evidence[material_id] = material.get("evidence", {"text": [], "image": []})
            for method, spec in material.get("methods", {}).items():
                self.questions[(material_id, method)] = spec
            for question, spec in material.get("validity", {}).items():
                self._index_unique(self.validity, question, spec, "validity")
            for question, spec in material.get("answers", {}).items():
                self._index_unique(self.answers, question, spec, "answer")
            for question, spec in material.get("responses", {}).items():
                self._index_unique(self.responses, question, spec, "response")

    @staticmethod
    def _index_unique(target: dict, key: str, value: Any, kind: str) -> None:
        if key in target:
            raise ValueError(f"duplicate mock {kind} entry for question: {key!r}")
        target[key] = value

    def _complete(self, request: CompletionRequest, value: Any, *, finish_reason: str = "stop") -> Completion:
        text = _json_text(value) if not isinstance(value, str) else value
        return Completion(
            text=text,
            model=self.model,
            usage={"input_chars": len(request.prompt), "output_chars": len(text)},
            finish_reason=finish_reason,
        )

    def complete(self, request: CompletionRequest) -> Completion:
        task = request.task
        if task == "generate_question":
            material_id = request.metadata.get("material_id")
            method = request.metadata.get("method")
            if material_id is None or method is None:
                raise ProviderError("mock generation requires material_id and method metadata")
            spec = self.questions.get((material_id, method))
            if spec is None:
                raise ProviderError(f"mock script has no question for {material_id}/{method}")
            if "technical_failure" in spec:
                raise ProviderError(f"simulated technical failure: {spec['technical_failure']}")
            if "raw" in spec:
                return self._complete(
                    request,
                    str(spec["raw"]),
                    finish_reason=str(spec.get("finish_reason", "stop")),
                )
            question = spec.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ProviderError(f"mock script question is invalid for {material_id}/{method}")
            return self._complete(request, {"question": question})

        if task in {"extract_text_evidence", "extract_image_evidence"}:
            material_id = request.metadata.get("material_id")
            if material_id is None:
                raise ProviderError("mock evidence extraction requires material_id metadata")
            evidence = self.evidence.get(material_id)
            if evidence is None:
                raise ProviderError(f"mock script has no evidence for {material_id}")
            key = "text" if task == "extract_text_evidence" else "image"
            return self._complete(request, {"items": evidence.get(key, [])})

        if task == "pair_evidence":
            material_id = request.metadata.get("material_id")
            evidence = self.evidence.get(material_id or "", {"text": [], "image": []})
            return self._complete(
                request,
                {
                    "pair": {
                        "text": evidence.get("text", []),
                        "image": evidence.get("image", []),
                    },
                    "rationale": "synthetic fixture: pair the rule with the visual reading",
                },
            )

        if task == "validate_question":
            spec = self.validity.get(request.question or "")
            if spec is None:
                return self._complete(
                    request,
                    {"status": "unresolved", "reasons": ["mock script has no validity label"]},
                )
            return self._complete(request, spec)

        if task == "annotate_answer":
            spec = self.answers.get(request.question or "")
            if spec is None:
                return self._complete(
                    request,
                    {
                        "status": "unresolved",
                        "answer": None,
                        "acceptable_answers": [],
                        "rubric": None,
                        "evidence_text": [],
                        "evidence_image": [],
                        "notes": "mock script has no answer for this question",
                    },
                )
            candidate = spec.get("candidate", {})
            return self._complete(
                request,
                {
                    "status": candidate.get("status", "candidate"),
                    "answer": candidate.get("answer"),
                    "acceptable_answers": candidate.get("acceptable_answers", []),
                    "rubric": candidate.get("rubric"),
                    "evidence_text": candidate.get("evidence_text", []),
                    "evidence_image": candidate.get("evidence_image", []),
                },
            )

        if task == "verify_answer":
            spec = self.answers.get(request.question or "")
            if spec is None or "verification" not in spec:
                return self._complete(
                    request,
                    {
                        "status": "unresolved",
                        "answer": None,
                        "acceptable_answers": [],
                        "rubric": None,
                        "notes": "mock script has no verification for this question",
                    },
                )
            return self._complete(request, spec["verification"])

        if task == "answer_question":
            if request.condition is None:
                raise ProviderError("mock answer_question requires a condition")
            question_responses = self.responses.get(request.question or "", {})
            spec = question_responses.get(request.condition)
            if spec is None:
                return self._complete(request, {"status": "refused", "answer": None})
            return self._complete(request, spec)

        raise ProviderError(f"mock provider does not implement task {task!r}")


class JudgeHTTPError(RuntimeError):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f'judge_http_{status}')


class JudgeProvider:
    """Cached Python gateway adapter, with optional visual research tasks.

    Cache writes precede downstream scoring, so a persisted completion survives
    process interruption. A lost remote response may still have been billed.
    """

    def __init__(self, *, model: str, base_url: str, api_key: str, cache_dir: Path,
                 ledger_path: Path, max_requests: int = 12, timeout: float = 60,
                 max_tokens: int = 768, max_attempts: int = 1,
                 api: str = 'auto', connect_ip: str | None = None,
                 local_address: str | None = None, visual: bool = False):
        if not model or not base_url.startswith(('https://', 'http://')) or not api_key:
            raise ValueError('judge requires model, HTTP(S) base URL and API key')
        if max_requests < 1 or not 0 < timeout <= 300 or max_tokens < 1 or not 1 <= max_attempts <= 3:
            raise ValueError('invalid judge budget/timeout/token limit/attempt count')
        self.model = model
        self.visual = visual
        parts = urlsplit(base_url.rstrip('/'))
        if not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError('judge URL must have a hostname and no credentials/query/fragment')
        base_path = parts.path
        for suffix in ('/chat/completions', '/responses'):
            if base_path.endswith(suffix):
                base_path = base_path[:-len(suffix)]
                break
        if api == 'auto':
            api = 'responses' if model.lower().startswith(('gpt-', 'luna')) else 'chat_completions'
        if api not in ('chat_completions', 'responses'):
            raise ValueError('LLM_JUDGE_API must be auto, chat_completions or responses')
        self.api = api
        self._url = urlunsplit((parts.scheme, parts.netloc, base_path + (
            '/responses' if api == 'responses' else '/chat/completions'), '', ''))
        self.connect_ip, self.local_address = connect_ip, local_address     # set LLM_JUDGE_CONNECT_IP / LLM_JUDGE_LOCAL_ADDRESS to pin a gateway
        self._key = api_key
        self.cache_dir, self.ledger_path = cache_dir, ledger_path
        self.max_requests, self.timeout = max_requests, timeout
        self.max_tokens, self.max_attempts = max_tokens, max_attempts
        self.identity = {'adapter': 'judge-http-v2', 'model': model, 'api': api,
                         'endpoint_sha256': sha256_text(self._url),
                         'temperature': 0 if api == 'chat_completions' else None,
                         'max_tokens': max_tokens, 'trust_env': False,
                         'connect_ip': self.connect_ip, 'local_address': self.local_address,
                         'tls_verify': True}
        if visual:
            self.identity['adapter'] = 'judge-vision-v1'

    @classmethod
    def from_env(cls, *, root: Path, run_dir: Path, max_requests: int = 12,
                 max_tokens: int = 768, visual: bool = False,
                 env_prefix: str = 'LLM_JUDGE', model: str | None = None) -> 'JudgeProvider':
        from dotenv import dotenv_values
        values = {**dotenv_values(root / '.env'), **os.environ}
        required = tuple(f'{env_prefix}_{suffix}' for suffix in ('MODEL_NAME', 'BASE_URL', 'API_KEY'))
        missing=[key for key in required[1:] if not values.get(key)]
        if not (model or values.get(required[0])):missing.append(required[0])
        if missing:
            raise ValueError('missing environment variables: '+', '.join(missing))
        try:
            timeout = float(values.get(f'{env_prefix}_TIMEOUT', 60))
            attempts = min(3, max(1, int(values.get(f'{env_prefix}_MAX_RETRIES', 0)) + 1))
        except (ValueError, TypeError):
            raise ValueError('invalid judge timeout/retry environment settings') from None
        return cls(model=model or values[required[0]], base_url=values[required[1]],
                   api_key=values[required[2]], cache_dir=root / 'outputs/cache/judge',
                   ledger_path=run_dir / 'requests.jsonl', max_requests=max_requests,
                   timeout=timeout, max_tokens=max_tokens, max_attempts=attempts,
                   api=values.get(f'{env_prefix}_API', 'auto'),
                   connect_ip=values.get(f'{env_prefix}_CONNECT_IP'),
                   local_address=values.get(f'{env_prefix}_LOCAL_ADDRESS'), visual=visual)

    def _post_json(self, payload: dict) -> dict:
        """Direct connection ignores proxy env; DNS override is per connection.

        HTTPSConnection still uses the original host for Host, TLS SNI and
        certificate validation. Only its TCP destination/source are overridden.
        No process-wide socket monkeypatch and no disabled certificate checks.
        """
        parts = urlsplit(self._url)
        connection_type = http.client.HTTPSConnection if parts.scheme == 'https' else http.client.HTTPConnection
        connection = connection_type(parts.hostname, parts.port, timeout=self.timeout,
                                     source_address=(self.local_address, 0) if self.local_address else None)
        if self.connect_ip:
            def connect(address, timeout, source_address):
                return socket.create_connection((self.connect_ip, address[1]), timeout, source_address)
            connection._create_connection = connect
        try:
            connection.request('POST', parts.path, body=json.dumps(payload, ensure_ascii=False).encode(),
                               headers={'Content-Type': 'application/json',
                                        'Authorization': 'Bearer ' + self._key})
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                raise JudgeHTTPError(response.status)
            body = response.read(4 * 1024 * 1024 + 1)
            if len(body) > 4 * 1024 * 1024:
                raise ValueError('judge response exceeds size limit')
            return json.loads(body)
        finally:
            connection.close()

    def complete(self, request: CompletionRequest, *, cache_only: bool = False) -> Completion:
        if not self.visual and (request.task != 'score_response' or request.text is not None or request.image_path is not None):
            raise ValueError('JudgeProvider supports text-only score_response requests')
        if self.visual and request.task not in {'score_response', 'assess_quality', 'verify_reference',
                                               'answer_question', 'probe_image', 'screen_material', 'audit_material',
                                               'refine_question', 'check_question', 'generate_question', 'audit_question'}:
            raise ValueError('visual judge supports only diagnostic evaluation tasks')
        image_data = Path(request.image_path).read_bytes() if request.image_path else None
        image_hash = hashlib.sha256(image_data).hexdigest() if image_data is not None else None
        content = request.prompt
        if image_data is not None:
            if len(image_data) > 20 * 1024 * 1024:
                raise ValueError('image exceeds 20 MiB request limit')
            if image_data.startswith(b'\x89PNG\r\n\x1a\n'):
                mime = 'image/png'
            elif image_data.startswith(b'\xff\xd8\xff'):
                mime = 'image/jpeg'
            else:
                raise ValueError('diagnostic vision supports PNG/JPEG images only')
            url = f'data:{mime};base64,' + base64.b64encode(image_data).decode('ascii')
            if self.api == 'responses':
                content = [{'type': 'input_text', 'text': request.prompt},
                           {'type': 'input_image', 'image_url': url, 'detail': 'high'}]
            else:
                content = [{'type': 'text', 'text': request.prompt},
                           {'type': 'image_url', 'image_url': {'url': url, 'detail': 'high'}}]
        # Condition and protocol affect cache identity but are hidden from the judge.
        cache_inputs = {
            'identity': self.identity, 'prompt': request.prompt, 'question': request.question,
            'condition': request.condition, 'metadata': {k: v for k, v in request.metadata.items() if k != 'caller_call_id'},
        }
        if self.visual:
            cache_inputs.update(task=request.task, image_sha256=image_hash,
                                text_sha256=sha256_text(request.text) if request.text is not None else None)
        cache_key = sha256_text(canonical_json(cache_inputs))
        path = self.cache_dir / f'{cache_key}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['key'] != cache_key or sha256_text(saved['text']) != saved['text_sha256']:
                raise ValueError('judge cache integrity mismatch')
            return Completion(saved['text'], saved['model'],
                              {'cache_hit': True, 'network_requests': 0,
                               'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
                               'cost': 0, 'original_usage': saved['usage']}, saved['finish_reason'])
        if cache_only:
            raise ProviderError('interrupted_call_without_cached_response:billing_unknown')
        if self.api == 'responses':
            payload = {'model': self.model, 'max_output_tokens': self.max_tokens,
                       'input': [{'role': 'user', 'content': content}], 'store': False}
        else:
            payload = {'model': self.model, 'temperature': 0, 'max_tokens': self.max_tokens,
                       'messages': [{'role': 'user', 'content': content}]}
        failures = []
        start = time.monotonic()
        for attempt in range(self.max_attempts):
            # A durable intent is written BEFORE sending. An interrupted request still
            # consumes the local request budget and remains visible as uncertain billing.
            previous = read_jsonl(self.ledger_path)
            count = sum(row.get('event') == 'started' for row in previous)
            if count >= self.max_requests:
                raise RequestBudgetExceeded('judge request budget exhausted; increase --max-requests to resume')
            request_no = count + 1
            append_jsonl(self.ledger_path, {'event': 'started', 'request_no': request_no,
                         'cache_key': cache_key, 'caller_call_id': request.metadata.get('caller_call_id'),
                         'created_at': utc_now()})
            try:
                data = self._post_json(payload)
                if self.api == 'responses':
                    content = ''.join(part['text'] for item in data.get('output', [])
                                      if item.get('type') == 'message'
                                      for part in item.get('content', []) if part.get('type') == 'output_text')
                    finish = 'stop' if data.get('status') == 'completed' else str(data.get('status', 'unknown'))
                    raw_usage = data.get('usage') or {}
                    raw_usage = {**raw_usage, 'prompt_tokens': raw_usage.get('input_tokens'),
                                 'completion_tokens': raw_usage.get('output_tokens')}
                else:
                    choice = data['choices'][0]
                    content = choice['message'].get('content')
                    finish = str(choice.get('finish_reason', 'unknown'))
                    raw_usage = data.get('usage') or {}
                if not isinstance(content, str):
                    raise ValueError('missing textual completion')
                usage = {key: (raw_usage.get(key) if type(raw_usage.get(key)) is int else None) for key in
                         ('prompt_tokens', 'completion_tokens', 'total_tokens')}
                cost = raw_usage.get('cost')
                usage.update(cost=cost if type(cost) in (int, float) else None, cache_hit=False,
                             network_requests=attempt + 1, failed_attempts=failures,
                             finish_reason=finish,
                             elapsed_seconds=round(time.monotonic() - start, 3))
                if self.visual:
                    usage['input_audit'] = {'image_transmitted': image_data is not None,
                                            'image_sha256': image_hash,
                                            'image_bytes': len(image_data) if image_data is not None else 0,
                                            'payload_sha256': sha256_text(canonical_json(payload))}
                completion = Completion(content.replace(self._key, '[REDACTED]'),
                                        str(data.get('model') or self.model), usage, finish)
            except JudgeHTTPError as exc:
                error = f'judge_http_{exc.status}'
                retryable = exc.status == 429 or exc.status >= 500
            except (OSError, http.client.HTTPException) as exc:
                kind = type(exc).__name__
                error, retryable = f'judge_transport_failure:{kind}:billing_unknown', False
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                error, retryable = 'judge_invalid_api_response', False
            else:
                write_json(path, {'key': cache_key, 'text': completion.text,
                                 'text_sha256': sha256_text(completion.text),
                                 'model': completion.model, 'usage': usage, 'finish_reason': finish})
                append_jsonl(self.ledger_path, {'event': 'completed', 'request_no': request_no,
                             'cache_key': cache_key, 'usage': usage, 'created_at': utc_now()})
                return completion
            append_jsonl(self.ledger_path, {'event': 'failed', 'request_no': request_no,
                         'cache_key': cache_key, 'error': error, 'created_at': utc_now()})
            failures.append(error)
            if not retryable or attempt + 1 == self.max_attempts:
                raise ProviderError(error) from None
            time.sleep(min(attempt + 1, 2))
        raise AssertionError('unreachable')


class ModelCaller:
    """Durable, stateless calls for the versioned three-phase protocol.

    The provider performs one transport attempt; this layer records each bounded
    retry. Completed raw calls survive a crash before records.jsonl is updated.
    """

    def __init__(self, *, provider: Provider, prompts, run_dir: Path,
                 protocol_version: str, max_attempts: int = 3, json_repair: str | None = None):
        if not 1 <= max_attempts <= 3:
            raise ValueError('max_attempts must be between 1 and 3')
        if json_repair is not None and json_repair not in JSON_REPAIRS:
            raise ValueError(f'unknown json_repair: {json_repair}')
        self.json_repair = json_repair
        self.provider, self.prompts, self.run_dir = provider, prompts, run_dir
        self.protocol_version, self.max_attempts = protocol_version, max_attempts
        if isinstance(provider, JudgeProvider) and provider.max_attempts != 1:
            raise ValueError('ModelCaller owns retries; configure provider.max_attempts=1')

    def call_model(self, *, operation_id: str, attempt_id: str, role: str,
                   template: str, variables: dict[str, str], task: str,
                   text: str | None = None, image_path: str | None = None,
                   question: str | None = None, condition: str | None = None,
                   output_format: str = 'json', nonce: str | None = None) -> dict:
        from .records import atomic_upsert, file_lock, sha256_file, stable_id
        prompt = self.prompts.render(template, **variables)
        metadata = {'protocol_version': self.protocol_version, 'prompt_version': template,
                    'role': role, 'output_format': output_format}
        if nonce is not None:
            metadata['reproducibility_probe'] = nonce
        request = CompletionRequest(task=task, prompt=prompt, text=text, image_path=image_path,
                                    question=question, condition=condition, metadata=metadata)
        identity = getattr(self.provider, 'identity', {'model': self.provider.model})
        inputs = {'prompt': prompt, 'text': text, 'question': question,
                  'images': [] if image_path is None else [
                      {'path': image_path, 'sha256': sha256_file(Path(image_path)), 'detail': 'high'}],
                  'history': [], 'condition': condition, 'metadata': metadata,
                  'model_config': identity}
        signature = sha256_text(canonical_json(inputs))
        path = self.run_dir / 'calls.jsonl'
        # Serialize operations in one run. Cross-run network parallelism remains bounded
        # by the process invoking the CLI, rather than a new scheduling service.
        with file_lock(self.run_dir / '.model.lock'):
            for index in range(self.max_attempts):
                call_id = stable_id('call', operation_id, index)
                prior = next((r for r in read_jsonl(path) if r['call_id'] == call_id), None)
                if prior and prior['input_sha256'] != signature:
                    raise ValueError('call input changed for completed operation; use a new run_id')
                if prior and prior['status'] not in ('started', 'budget_paused'):
                    if prior.get('retryable') and index + 1 < self.max_attempts:
                        continue
                    # A crash may occur between durable raw output and JSON parsing.
                    if prior['status'] == 'ok' and prior.get('parsed_output') is None:
                        try:
                            if prior.get('finish_reason') != 'stop':
                                raise OutputParseError('incomplete_completion')
                            prior['parsed_output'], repaired = parse_model_output(
                                prior['raw_output'], output_format, self.json_repair)
                            if repaired:
                                prior['parse_repair'] = repaired
                        except OutputParseError as exc:
                            prior.update(status='parse_failure', error=str(exc))
                        atomic_upsert(path, 'call_id', prior)
                    return prior
                started = prior or {
                    'call_id': call_id, 'operation_id': operation_id,
                    'attempt_id': attempt_id, 'run_id': self.run_dir.name,
                    'protocol_version': self.protocol_version, 'role': role,
                    'step': template, 'task': task, 'condition': condition,
                    'technical_attempt_index': index + 1, 'prompt_version': template,
                    'prompt_sha256': sha256_text(prompt), 'input_sha256': signature,
                    'actual_input': inputs, 'started_at': utc_now(), 'status': 'started',
                    'raw_output': None, 'parsed_output': None, 'usage': None,
                }
                atomic_upsert(path, 'call_id', started)
                begin = time.monotonic()
                result = dict(started)
                attempt_request = replace(request, metadata={**request.metadata, 'caller_call_id': call_id})
                def remote_count():
                    if hasattr(self.provider, 'remote_requests_for_call'):
                        return self.provider.remote_requests_for_call(call_id)
                    return sum(r.get('event') == 'started' and r.get('caller_call_id') == call_id
                               for r in read_jsonl(self.provider.ledger_path))
                try:
                    if prior and prior['status'] == 'started':
                        if hasattr(self.provider, 'recover_completion'):
                            completion = self.provider.recover_completion(attempt_request)
                        elif isinstance(self.provider, JudgeProvider):
                            completion = self.provider.complete(attempt_request, cache_only=True)
                        else:
                            raise ProviderError('interrupted_call_without_cached_response:billing_unknown')
                    else:
                        completion = self.provider.complete(attempt_request)
                except RequestBudgetExceeded:
                    atomic_upsert(path, 'call_id', {**started, 'status': 'budget_paused'})
                    raise
                except ProviderError as exc:
                    error = str(exc)
                    retryable = error == 'judge_http_429' or any(
                        error == f'judge_http_{code}' for code in range(500, 600))
                    result.update(status='technical_failure', error=error, retryable=retryable,
                                  ended_at=utc_now(), elapsed_seconds=round(time.monotonic()-begin, 6),
                                  usage=None,
                                  remote_requests=remote_count() if isinstance(self.provider, JudgeProvider) or hasattr(self.provider, 'remote_requests_for_call') else 1)
                    atomic_upsert(path, 'call_id', result)
                    if retryable and index + 1 < self.max_attempts:
                        time.sleep(min(index + 1, 2))
                        continue
                    return result
                recovered = bool(prior and prior['status'] == 'started' and (isinstance(self.provider, JudgeProvider) or hasattr(self.provider, 'remote_requests_for_call')) and remote_count())
                usage = completion.usage.get('original_usage', completion.usage) if recovered else completion.usage
                result.update(status='ok', raw_output=completion.text, model=completion.model,
                              usage=usage, finish_reason=completion.finish_reason,
                              cache_hit=bool(completion.usage.get('cache_hit')) and not recovered,
                              recovered_from_cache=recovered,
                              ended_at=utc_now(), elapsed_seconds=round(time.monotonic()-begin, 6),
                              retryable=False, error=None,
                              remote_requests=remote_count() if isinstance(self.provider, JudgeProvider) or hasattr(self.provider, 'remote_requests_for_call') else
                                  completion.usage.get('network_requests', 0 if completion.usage.get('cache_hit') else 1))
                # Persist RAW output before any parsing, including malformed JSON.
                atomic_upsert(path, 'call_id', result)
                try:
                    if completion.finish_reason != 'stop':
                        raise OutputParseError('incomplete_completion')
                    result['parsed_output'], repaired = parse_model_output(
                        completion.text, output_format, self.json_repair)
                    if repaired:
                        result['parse_repair'] = repaired
                except OutputParseError as exc:
                    result.update(status='parse_failure', error=str(exc), parsed_output=None)
                atomic_upsert(path, 'call_id', result)
                return result
        raise AssertionError('unreachable')


def call_model(caller: ModelCaller, **kwargs) -> dict:
    """Public entry point; every experiment role routes through this function."""
    return caller.call_model(**kwargs)
