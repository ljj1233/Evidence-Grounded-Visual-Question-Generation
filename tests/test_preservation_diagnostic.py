"""Offline checks only; mock outputs are not diagnostic/model evidence."""
import itertools
import json
import shutil
from pathlib import Path

import pytest

from egvqg.evidence_verification import CHECKS
from egvqg.preservation_diagnostic import (
    TEMPLATES, arm_result, call_inputs, conjunction, counted_rejection,
    init_diagnostic, report_diagnostic, run_diagnostic,
)
from egvqg.providers import JudgeProvider, RequestBudgetExceeded
from egvqg.records import canonical_json, read_jsonl, sha256_file, write_json, write_jsonl


def test_conjunction_never_creates_acceptance_and_failure_reading_is_conservative():
    for stages in itertools.product(('accept', 'reject', 'technical_failure'), repeat=3):
        assert (conjunction(*stages) == 'accept') == all(s == 'accept' for s in stages)
        assert counted_rejection('synthetic_planted_defect', *stages) == ('reject' in stages)
        assert counted_rejection('clean', *stages) == any(s != 'accept' for s in stages)
    output = {**dict.fromkeys(CHECKS, 'pass'), 'reason': 'fixture', 'recomputed_answers': ['a']}
    assert arm_result('V0', output) == 'accept'
    output['question_preserved'] = 'uncertain'
    assert arm_result('V0', output) == 'reject'
    assert arm_result('V0', {'question_preserved': 'fail'}) == 'technical_failure'


def test_allowlisted_extraction_and_witness_inputs():
    case = {'question': 'FIXED_Q', 'text': 'RETAINED_TEXT', 'image_path': 'retained.png',
            'reference': 'REFERENCE_SECRET', 'label': 'LABEL_SECRET'}
    proof = {'kind': 'ambiguous', 'scenario_a': 'SCENARIO_A', 'scenario_b': 'SCENARIO_B',
             'answer_a': 'ANSWER_SECRET_A', 'answer_b': 'ANSWER_SECRET_B', 'reason': 'REASON_SECRET'}
    for removed in ('T', 'I'):
        kwargs = call_inputs(case, removed, 'extract', 'K', proof=proof)
        encoded = canonical_json(kwargs)
        assert not any(s in encoded for s in ('SECRET', 'SCENARIO'))
        assert (kwargs['image_path'] is not None) == (removed == 'T')
        assert (kwargs['text'] is not None) == (removed == 'I')
    check = call_inputs(case, 'T', 'V1', 'K', proof=proof, constraints={'constraints': []})
    encoded = canonical_json(check)
    assert 'SCENARIO_A' in encoded and 'SCENARIO_B' in encoded
    assert not any(s in encoded for s in ('SECRET', 'FIXED_Q', 'retained.png', 'RETAINED_TEXT'))
    assert check['question'] is None and check['text'] is None and check['image_path'] is None


def test_durable_budget_resume_no_duplicates_raw_failures_and_seal(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path/'prompts').mkdir(); (tmp_path/'configs').mkdir(); (tmp_path/'docs').mkdir()
    for name in TEMPLATES:
        shutil.copy(root/'prompts'/f'{name}.txt', tmp_path/'prompts')
    shutil.copy(root/'configs/necessity_assessment_v2.json', tmp_path/'configs')
    for name in ('VQG_Proposal.md', 'uv.lock', 'pyproject.toml'):
        (tmp_path/name).write_text('fixture')
    shutil.copy(root/'tests/fixtures/stage_a/images/m_thermo.png', tmp_path/'image.png')
    source = tmp_path/'outputs/real-scienceqa20-v2'; source.mkdir(parents=True)
    case = {'case_id': 'C', 'question': 'FIXED_Q', 'text': 'RETAINED_TEXT',
            'image_path': 'image.png', 'image_sha256': sha256_file(tmp_path/'image.png')}
    write_jsonl(source/'cases.jsonl', [case])
    rows = []
    proof = {'kind': 'unique', 'answer': 'a', 'reason': 'CANDIDATE_ONLY_SENTINEL'}
    for g in range(8):
        for version, label in [('clean', 'clean'), ('defect-terse', 'synthetic_planted_defect'), ('defect-elaborate', 'synthetic_planted_defect')]:
            rows.append({'item_id': f'{g}_{version}', 'source_item_id': str(g), 'case_id': 'C',
                'removed': 'I', 'version': version, 'sealed_label': label, 'candidate': dict(proof),
                'violation': None if label == 'clean' else {'quote': 'FIXED_Q', 'constraint': 'fixture'}})
    for g in range(8, 11):
        rows.append({**rows[1], 'item_id': str(g), 'source_item_id': str(g), 'sealed_label': 'real_defect', 'version': 'real-defect-unchanged'})
    write_json(tmp_path/'construction.json', {'items': rows})
    out = tmp_path/'outputs/diagnostic'
    provider = JudgeProvider(model='gpt-6-luna', base_url='https://fixture.invalid/v1', api_key='fixture-key',
        cache_dir=tmp_path/'cache', ledger_path=out/'requests.jsonl', max_requests=70,
        max_tokens=3072, api='responses', visual=True)
    posted = []
    def post(payload):
        posted.append(payload)
        prompt = payload['input'][0]['content']
        if '你只负责从固定输入独立列出' in prompt:
            assert 'CANDIDATE_ONLY_SENTINEL' not in prompt
            value = {'constraints': [{'id': 'c1', 'source': 'question', 'quote': 'FIXED_Q',
                      'constraint': 'fixture', 'type': 'constraint'}], 'referring_expressions': []}
        elif '你是范围有限的保留约束' in prompt:
            value = {'checks': [{'id': 'c1', 'result': 'pass', 'quote': 'a', 'reason': 'fixture'}], 'overall': 'pass', 'reason': 'fixture'}
        else:
            value = {**dict.fromkeys(CHECKS, 'pass'), 'reason': 'fixture', 'recomputed_answers': ['a']}
        raw = json.dumps(value)
        if len(posted) == 1: raw = '```json\n'+raw+'\n```'  # must remain malformed, no normalization
        return {'status': 'completed', 'model': 'gpt-6-luna', 'output': [{'type': 'message',
            'content': [{'type': 'output_text', 'text': raw}]}], 'usage': {'input_tokens': 2, 'output_tokens': 2, 'total_tokens': 4}}
    provider._post_json = post
    kw = dict(run_id='diagnostic', project_root=tmp_path, provider=provider)
    init_diagnostic(**kw, construction_path='construction.json')
    with pytest.raises(RequestBudgetExceeded): run_diagnostic(**kw, max_requests=2)
    assert len(posted) == 2
    assert sum(r['event'] == 'started' for r in read_jsonl(out/'requests.jsonl')) == 2
    with pytest.raises(ValueError, match='70'): run_diagnostic(**kw, max_requests=71)
    assert run_diagnostic(**kw, max_requests=70)['new_steps'] == 63
    assert len(posted) == 65
    before = {p.name: sha256_file(p) for p in (out/'calls.jsonl', out/'requests.jsonl', out/'records.jsonl', out/'extractions.jsonl')}
    assert run_diagnostic(**kw)['new_steps'] == 0
    assert before == {p.name: sha256_file(p) for p in (out/'calls.jsonl', out/'requests.jsonl', out/'records.jsonl', out/'extractions.jsonl')}
    assert len({r['call_id'] for r in read_jsonl(out/'calls.jsonl')}) == 65
    first = read_jsonl(out/'records.jsonl')[0]['V0']
    assert first['result'] == 'technical_failure' and 'strict_json_failure' in first['parse_error']
    assert read_jsonl(out/'calls.jsonl')[0]['raw_output'].startswith('```json')
    result = report_diagnostic(run_id='diagnostic', project_root=tmp_path)
    assert result['counts']['conjunction_clean_rejected_of_8'] == 1
    assert json.loads((out/'acceptance.json').read_text())['logged_inputs_reconstructed'] == 65
    (out/'prompts'/f'{TEMPLATES[1]}.txt').write_text('changed')
    with pytest.raises(ValueError, match='sealed file'): run_diagnostic(**kw)
