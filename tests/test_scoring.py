"""Judge schema, binding, input isolation, cache invalidation and interrupted resume."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from egvqg.evaluation import score_response
from egvqg.pipeline import RunConfigMismatch, execute_run, rescore_run, summarize_run
from egvqg.prompts import PromptSet
from egvqg.providers import CallRecorder, Completion, CompletionRequest, JudgeProvider, RequestBudgetExceeded
from egvqg.records import JsonlTable, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('payload,finish,expected', [
    ({'status': 'scored', 'correct': True, 'rationale': 'equivalent units'}, 'stop', True),
    ({'status': 'scored', 'correct': False, 'rationale': 'wrong value'}, 'stop', False),
    ({'status': 'unresolved', 'correct': None, 'rationale': 'ambiguous'}, 'stop', None),
    ({'status': 'scored', 'correct': 'true', 'rationale': 'bad boolean'}, 'stop', None),
    ({'status': 'scored', 'correct': True, 'rationale': 'truncated'}, 'length', None),
    ({'status': 'scored', 'correct': True}, 'stop', None),
])
def test_model_score_schema_and_question_binding(tmp_path, payload, finish, expected):
    requests = []

    class FakeJudge:
        def complete(self, request):
            requests.append(request)
            return Completion(json.dumps(payload), 'test-judge', finish_reason=finish)

    response = dict(response_id='r', question_id='q', condition='I', status='answered', answer='one metre')
    answer = dict(answer_id='a', question_id='q', status='verified', answer='100 cm', rubric='length')
    recorder = CallRecorder(JsonlTable(tmp_path / 'calls.jsonl', 'call_id'),
                            run_id='test', protocol_version='p', default_model='test-judge')
    kwargs = dict(response=response, answer=answer, run_id='test', protocol_version='p',
                  scorer='test-judge', question='What is the length?', provider=FakeJudge(),
                  prompts=PromptSet(ROOT / 'prompts', names=('score_response',)), recorder=recorder)
    result = score_response(**kwargs)
    assert result.correct is expected
    assert result.status == ('unresolved' if expected is None else 'scored')
    assert 'one metre' in requests[0].prompt and '100 cm' in requests[0].prompt
    assert requests[0].text is None and requests[0].image_path is None
    answer['question_id'] = 'other-question'
    assert score_response(**kwargs).rationale == 'answer_question_mismatch'
    assert len(requests) == 1


def _judge(tmp_path, max_requests=100):
    return JudgeProvider(model='test-judge', base_url='https://example.invalid/v1', api_key='SECRET',
                         cache_dir=tmp_path / 'cache', ledger_path=tmp_path / 'requests.jsonl',
                         max_requests=max_requests)


def _transport(monkeypatch):
    requests = []

    def fetch(self, payload):
        requests.append(payload)
        assert len(payload['messages']) == 1
        assert isinstance(payload['messages'][0]['content'], str)
        result = {'model': 'test-judge', 'choices': [{'message': {'content': json.dumps(
            {'status': 'scored', 'correct': True, 'rationale': 'offline scripted judgement'})},
            'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        return result

    monkeypatch.setattr(JudgeProvider, '_post_json', fetch)
    return requests


def test_judge_cache_and_material_condition_protocol_invalidation(tmp_path, monkeypatch):
    requests = _transport(monkeypatch)
    judge = _judge(tmp_path)
    request = CompletionRequest(task='score_response', prompt='Q + reference + rubric + response',
                                question='Q', condition='TI', metadata={'protocol_version': 'v1'})
    first = judge.complete(request)
    # New process/provider instance restores a persisted completion without HTTP.
    assert _judge(tmp_path).complete(request).usage['cache_hit'] is True
    assert len(requests) == 1
    for changed in (replace(request, prompt=request.prompt + ' changed reference/rubric'),
                    replace(request, question='changed Q'), replace(request, condition='T'),
                    replace(request, metadata={'protocol_version': 'v2'})):
        assert judge.complete(changed).usage['cache_hit'] is False
    judge.model = 'other-model'
    judge.identity = {**judge.identity, 'model': 'other-model'}
    judge.complete(request)
    assert len(requests) == 6
    assert first.usage['total_tokens'] == 15 and first.usage['cost'] is None
    assert 'SECRET' not in (tmp_path / 'requests.jsonl').read_text()


def test_model_rescore_budget_resume_and_source_immutability(tmp_path, monkeypatch):
    requests = _transport(monkeypatch)
    source = execute_run(config_path='configs/stage_a.json', project_root=ROOT,
                         output_root=tmp_path, run_id='source', limit=1)
    original_scores = (source.run_dir / 'scores.jsonl').read_bytes()
    judge = _judge(tmp_path / 'scored', max_requests=1)
    kwargs = dict(source_run='source', run_id='scored', project_root=ROOT, output_root=tmp_path)
    with pytest.raises(RequestBudgetExceeded):
        rescore_run(**kwargs, judge=judge)
    assert len(requests) == 1
    judge = _judge(tmp_path / 'scored', max_requests=20)
    result = rescore_run(**kwargs, judge=judge)
    assert result.planned == 3
    assert len(requests) == 8
    assert (source.run_dir / 'scores.jsonl').read_bytes() == original_scores
    second = rescore_run(**kwargs, judge=judge)
    assert second.new_calls == second.new_records == 0
    assert len(requests) == 8
    assert summarize_run(run_id='scored', project_root=ROOT, output_root=tmp_path).metrics == result.metrics
    rows = read_jsonl(result.run_dir / 'scores.jsonl')
    assert len(rows) == len({r['score_id'] for r in rows}) == 12
    judge.identity = {**judge.identity, 'max_tokens': 999}
    with pytest.raises(RunConfigMismatch):
        rescore_run(**kwargs, judge=judge)


def test_transport_failure_is_unresolved_and_does_not_log_credentials(tmp_path, monkeypatch):
    import ssl
    from egvqg.providers import ProviderError

    def fail(self, payload):
        raise ssl.SSLEOFError('SECRET https://private.invalid credential')

    monkeypatch.setattr(JudgeProvider, '_post_json', fail)
    judge = _judge(tmp_path)
    with pytest.raises(ProviderError, match='SSLEOFError') as error:
        judge.complete(CompletionRequest(task='score_response', prompt='test'))
    assert 'SECRET' not in str(error.value)
    assert not list((tmp_path / 'cache').glob('*.json'))
    ledger = (tmp_path / 'requests.jsonl').read_text()
    assert 'SECRET' not in ledger and 'private.invalid' not in ledger
    assert len(read_jsonl(tmp_path / 'requests.jsonl')) == 2


@pytest.mark.parametrize('base,model,api,path,pinned', [
    ('https://gateway.example.test/v1/chat/completions/', 'gpt-test', 'auto', '/v1/responses', True),
    ('https://gateway.example.test/v1/responses', 'DeepSeek-test', 'auto', '/v1/chat/completions', True),
    ('http://127.0.0.1:8101/v1', 'gpt-test', 'chat_completions', '/v1/chat/completions', False),
])
def test_gateway_tcp_binding_sni_host_proxy_isolation_and_api(tmp_path, monkeypatch, base, model, api, path, pinned):
    import io
    import socket
    import ssl

    tcp, tls, sent = [], [], []
    before_dns = socket.getaddrinfo
    text = '{"status":"scored","correct":true,"rationale":"equivalent"}'
    if path.endswith('/responses'):
        data = {'status': 'completed', 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': text}]}],
            'usage': {'input_tokens': 7, 'output_tokens': 3, 'total_tokens': 10}}
    else:
        data = {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}}
    body = json.dumps(data).encode()

    class FakeSocket:
        def setsockopt(self, *args): pass
        def sendall(self, value): sent.append(value)
        def makefile(self, *args):
            return io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
        def close(self): pass

    def connect(address, timeout, source_address=None):
        tcp.append((address, source_address))
        return FakeSocket()

    def wrap(context, sock, *, server_hostname):
        assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
        tls.append(server_hostname)
        return sock

    monkeypatch.setenv('HTTPS_PROXY', 'http://unusable-proxy.invalid:9999')
    monkeypatch.setenv('ALL_PROXY', 'http://unusable-proxy.invalid:9999')
    monkeypatch.setattr(socket, 'create_connection', connect)
    monkeypatch.setattr(ssl.SSLContext, 'wrap_socket', wrap)
    judge = JudgeProvider(model=model, api=api, base_url=base, api_key='SECRET',
                         cache_dir=tmp_path/'cache', ledger_path=tmp_path/'requests.jsonl',
                         connect_ip='203.0.113.7' if pinned else None, local_address='198.51.100.9' if pinned else None)
    result = judge.complete(CompletionRequest(task='score_response', prompt='grade this'))
    wire = b''.join(sent)
    assert b'POST ' + path.encode() + b' HTTP/1.1' in wire
    assert result.text == text and result.finish_reason == 'stop'
    assert result.usage['prompt_tokens'] == 7 and result.usage['total_tokens'] == 10
    if pinned:
        assert tcp == [(('203.0.113.7', 443), ('198.51.100.9', 0))]
        assert tls == ['gateway.example.test']
        assert b'Host: gateway.example.test' in wire
    else:
        assert tcp == [(('127.0.0.1', 8101), None)] and tls == []
    payload = json.loads(wire.split(b'\r\n\r\n', 1)[1])
    if path.endswith('/responses'):
        assert 'input' in payload and 'max_output_tokens' in payload
        assert 'messages' not in payload and 'temperature' not in payload
    else:
        assert 'messages' in payload and 'max_tokens' in payload and 'input' not in payload
    assert socket.getaddrinfo is before_dns


@pytest.mark.parametrize('status', ['incomplete', 'failed'])
def test_responses_incomplete_output_stays_unresolved(tmp_path, monkeypatch, status):
    def post(self, payload):
        return {'status': status, 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': '{"status":"scored","correct":true,"rationale":"x"}'}]}],
            'usage': {'input_tokens': 5, 'output_tokens': 2, 'total_tokens': 7}}
    monkeypatch.setattr(JudgeProvider, '_post_json', post)
    judge = JudgeProvider(model='gpt-test', base_url='https://example.invalid/v1', api_key='SECRET',
                         cache_dir=tmp_path/'cache', ledger_path=tmp_path/'requests.jsonl')
    result = judge.complete(CompletionRequest(task='score_response', prompt='grade'))
    assert result.finish_reason != 'stop'
    # score_response's existing truncated-completion regression verifies this
    # finish reason cannot be promoted to a scored judgement.
