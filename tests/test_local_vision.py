"""Batching provider: request/response pairing, batch limits, failure isolation, usage, CLI wiring."""
import threading
import time

import pytest

from egvqg.local_vision import BatchingProvider, LocalOutput
from egvqg.providers import CompletionRequest, ProviderError


class Fake:
    def __init__(self, bad=(), delay=0.0):
        self.calls, self.bad, self.delay = [], set(bad), delay

    def generate(self, requests):
        self.calls.append([r.prompt for r in requests])
        if any(r.prompt in self.bad for r in requests):
            raise RuntimeError('boom')
        time.sleep(self.delay)
        return [LocalOutput('echo:' + r.prompt, 3, 2) for r in requests]


def provider(fake, **kw):
    return BatchingProvider(fake, model='fake', identity={'model': 'fake'}, wait_seconds=0.3, **kw)


def fire(p, prompts):
    results, errors = {}, {}

    def one(prompt):
        try:
            results[prompt] = p.complete(CompletionRequest(task='t', prompt=prompt))
        except ProviderError as exc:
            errors[prompt] = exc
    threads = [threading.Thread(target=one, args=(x,)) for x in prompts]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    return results, errors


def test_concurrent_requests_are_batched_within_the_limit_and_each_gets_its_own_answer():
    fake = Fake(delay=0.05)
    results, errors = fire(provider(fake, max_batch=4), [f'p{n}' for n in range(10)])
    assert not errors and {k: v.text for k, v in results.items()} == {f'p{n}': f'echo:p{n}' for n in range(10)}
    sizes = [len(c) for c in fake.calls]
    assert max(sizes) <= 4 and max(sizes) > 1 and sum(sizes) == 10


def test_one_bad_request_does_not_fail_the_rest_of_its_batch():
    fake = Fake(bad=['BAD'])
    results, errors = fire(provider(fake, max_batch=8), ['a', 'BAD', 'b', 'c'])
    assert set(results) == {'a', 'b', 'c'} and set(errors) == {'BAD'}
    assert all(r.text == 'echo:' + k for k, r in results.items())


def test_usage_reports_tokens_batch_size_and_no_network():
    result = provider(Fake(), max_batch=2).complete(CompletionRequest(task='t', prompt='x'))
    usage = result.usage
    assert (usage['prompt_tokens'], usage['completion_tokens'], usage['total_tokens']) == (3, 2, 5)
    assert usage['network_requests'] == 0 and usage['batch_size'] == 1 and result.finish_reason == 'stop'


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        BatchingProvider(Fake(), model='m', identity={}, max_batch=0)


def test_cli_exposes_the_local_model_options():
    from egvqg.cli import build_parser
    args = build_parser().parse_args(['simple-baseline-run', '--run-id', 'r', '--local-model-path', '/m', '--max-batch', '16'])
    assert (args.local_model_path, args.max_batch, args.device, args.max_new_tokens) == ('/m', 16, 'cuda:0', 1024)
