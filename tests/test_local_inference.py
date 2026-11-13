"""Offline role routing, local model invalidation and budget checks."""
import pytest

from egvqg.local_inference import LocalExperimentProvider, LocalVisionProvider, model_fingerprint
from egvqg.providers import Completion, CompletionRequest, ModelCaller, ProviderError, RequestBudgetExceeded
from egvqg.prompts import PromptSet
from egvqg.records import append_jsonl, atomic_upsert, read_jsonl, write_json


def test_weight_invalidation_and_cumulative_local_budget(tmp_path, monkeypatch):
    model=tmp_path/'weights';model.mkdir()
    write_json(model/'config.json',{'model_type':'qwen3_vl'})
    weights=model/'model.safetensors';weights.write_bytes(b'old')
    before=model_fingerprint(model);weights.write_bytes(b'new')
    assert before!=model_fingerprint(model)
    monkeypatch.setattr('egvqg.local_inference.version',lambda _: 'offline')
    provider=LocalVisionProvider(model_path=model,run_dir=tmp_path,max_calls=1)
    append_jsonl(provider.ledger_path,{'event':'started','signature':'prior'})
    with pytest.raises(RequestBudgetExceeded):
        provider.complete(CompletionRequest(task='generate_question',prompt='fixture',metadata={'role':'generation'}))
    assert len(read_jsonl(provider.ledger_path))==1


def test_routing_failure_accounting_and_resume(tmp_path):
    (tmp_path/'prompts').mkdir();(tmp_path/'prompts/t.txt').write_text('$Q')
    prompts=PromptSet(tmp_path/'prompts',names=('t',))
    calls=[]
    class Mock:
        def __init__(self,name):
            self.model=name;self.identity={'model':name};self.ledger_path=tmp_path/'requests.jsonl';self.max_attempts=1
        def complete(self,request,cache_only=False):
            calls.append((self.model,request))
            if request.prompt=='fail':raise ProviderError('local_inference_failure:fixture')
            if self.model=='judge' and not cache_only:
                append_jsonl(self.ledger_path,{'event':'started','caller_call_id':request.metadata['caller_call_id']})
            return Completion('{"ok":true}',self.model,{'network_requests':int(self.model=='judge')})
    routed=LocalExperimentProvider(Mock('local'),Mock('judge'))
    caller=ModelCaller(provider=routed,prompts=prompts,run_dir=tmp_path,protocol_version='fixture',max_attempts=1)
    for role in ('generation','answer_TI','answer_T','answer_I','answer_0','candidate_answer','score_TI'):
        result=caller.call_model(operation_id=role,attempt_id='fixture',role=role,template='t',variables={'Q':'fixture'},task='assess_quality')
        expected='local' if role=='generation' or role.startswith('answer_') else 'judge'
        assert result['model']==expected and result['remote_requests']==int(expected=='judge')
    count=len(calls)
    caller.call_model(operation_id='generation',attempt_id='fixture',role='generation',template='t',variables={'Q':'fixture'},task='assess_quality')
    assert len(calls)==count
    failure=caller.call_model(operation_id='failure',attempt_id='fixture',role='generation',template='t',variables={'Q':'fail'},task='assess_quality')
    assert failure['status']=='technical_failure' and failure['remote_requests']==0
    remote=next(r for r in read_jsonl(tmp_path/'calls.jsonl') if r['role']=='candidate_answer')
    remote.update(status='started',raw_output=None,parsed_output=None)
    atomic_upsert(tmp_path/'calls.jsonl','call_id',remote)
    remote_started=len(read_jsonl(tmp_path/'requests.jsonl'))
    recovered=caller.call_model(operation_id='candidate_answer',attempt_id='fixture',role='candidate_answer',template='t',variables={'Q':'fixture'},task='assess_quality')
    assert recovered['recovered_from_cache'] and recovered['remote_requests']==1
    assert len(read_jsonl(tmp_path/'requests.jsonl'))==remote_started
    local=next(r for r in read_jsonl(tmp_path/'calls.jsonl') if r['role']=='generation')
    local.update(status='started',raw_output=None,parsed_output=None)
    atomic_upsert(tmp_path/'calls.jsonl','call_id',local)
    completed_calls=len(calls)
    interrupted=caller.call_model(operation_id='generation',attempt_id='fixture',role='generation',template='t',variables={'Q':'fixture'},task='assess_quality')
    assert interrupted['status']=='technical_failure' and interrupted['remote_requests']==0
    assert len(calls)==completed_calls
