"""Ensemble necessity evaluation: decision table, material isolation, resume, failure handling, arbiter scope, denominator."""
import json
import shutil
from pathlib import Path

import pytest

from egvqg.necessity_ensemble import (ARBITER, Evaluation, GROUPS, NO_IMAGE_TEXT, TEMPLATES, init_ensemble, key, norm_value,
                                      report_ensemble, run_ensemble, vote)
from egvqg.providers import Completion
from egvqg.records import read_jsonl, sha256_file, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('raw,expected', [('1/2', '50'), ('50%', '50'), (' 0.5 ', '0.5'), ('Tomatoes', 'tomatoe'),
                                          ('Tomato', 'tomato'), (None, None), ('a - b', 'a b'), ('3/0', '3/0')])
def test_norm_value(raw, expected):
    assert norm_value(raw) == expected


@pytest.mark.parametrize('values,expected', [([True, True], True), ([True, False], None), ([False, False], False), ([True, None], None),
                                             ([True, False, True], True), ([True, None, None], None), ([False, True, False], False)])
def test_vote_requires_agreement_and_never_counts_undecided_as_support(values, expected):
    assert vote(values) is expected


def item(attempt='x', witnesses=True, reference='gamma'):
    pair = {'scenario_a': 'A', 'scenario_b': 'B'}
    return {'attempt_id': attempt, 'item_id': 'Q' + attempt, 'text': 'T', 'question': 'q', 'reference': reference, 'validity': 1,
            'witness_text_removed': pair if witnesses else None, 'witness_image_removed': pair if witnesses else None}


def record(parsed, strings=None):
    return {'parsed': parsed, 'strings': strings}


def slots(role, attempt, mapping):
    strings = sorted(mapping)
    return {key(role, 'SLOT', attempt): record({'items': [{'i': n, 'values': {'v': mapping[s]}} for n, s in enumerate(strings)]}, strings)}


def witness(a, b, ok=True):
    return {'s': [{'presup_ok': ok, 'consistent': True, 'answer': a}, {'presup_ok': ok, 'consistent': True, 'answer': b}]}


def table(roles, *, suff=None, wit=None, mapping=None, attempt='x'):
    """Records for the given roles. suff/wit map direction -> parsed output; mapping: answer string -> slot value."""
    out = {}
    for role in roles:
        for mod in ('T', 'I'):
            if suff and mod in suff:
                out[key(role, 'S', attempt, mod)] = record(suff[mod])
            if wit and mod in wit:
                out[key(role, 'W', attempt, mod)] = record(wit[mod])
        out.update(slots(role, attempt, mapping or {}))
    return out


MAP = {'gamma': 'g', 'alpha': 'a', 'beta': 'b', 'alpha too': 'a'}
NOT_SUFF = {'determinable': False, 'answer': None}


@pytest.mark.parametrize('name,suff,wit,expect', [
    ('dual: both witnesses valid, nothing sufficient', {'T': NOT_SUFF, 'I': NOT_SUFF}, {'T': witness('alpha', 'beta'), 'I': witness('alpha', 'beta')}, 1),
    ('text alone sufficient => image removable', {'T': {'determinable': True, 'answer': 'gamma'}, 'I': NOT_SUFF}, {'T': witness('alpha', 'beta'), 'I': witness('alpha', 'beta')}, 0),
    ('image alone sufficient => text removable', {'T': NOT_SUFF, 'I': {'determinable': True, 'answer': 'gamma'}}, {'T': witness('alpha', 'beta'), 'I': witness('alpha', 'beta')}, 0),
    ('witness answers equal after slot normalisation => not a witness', {'T': NOT_SUFF, 'I': NOT_SUFF}, {'T': witness('alpha', 'alpha too'), 'I': witness('alpha', 'beta')}, None),
    ('witness violating a presupposition is invalid', {'T': NOT_SUFF, 'I': NOT_SUFF}, {'T': witness('alpha', 'beta', ok=False), 'I': witness('alpha', 'beta')}, None),
    ('both witness answers equal the full answer => no difference', {'T': NOT_SUFF, 'I': NOT_SUFF}, {'T': witness('gamma', 'gamma'), 'I': witness('alpha', 'beta')}, None),
    ('sufficient answer that differs from the full answer is not a proof', {'T': {'determinable': True, 'answer': 'alpha'}, 'I': NOT_SUFF}, {'T': witness('alpha', 'beta'), 'I': witness('alpha', 'beta')}, 1),
])
def test_decision_table_two_groups(name, suff, wit, expect):
    records = table(GROUPS, suff=suff, wit=wit, mapping=MAP)
    result = Evaluation([item()], records).results['x']
    assert result['B'] is expect or result['B'] == expect, name


def test_split_between_groups_stays_undecided_until_the_arbiter_rules():
    base = {'T': NOT_SUFF, 'I': NOT_SUFF}
    good = {'T': witness('alpha', 'beta'), 'I': witness('alpha', 'beta')}
    bad = {'T': witness('alpha', 'beta', ok=False), 'I': witness('alpha', 'beta')}
    records = {**table(['a'], suff=base, wit=good, mapping=MAP), **table(['b'], suff=base, wit=bad, mapping=MAP)}
    evaluation = Evaluation([item()], records)
    assert evaluation.results['x']['B'] is None and [i['attempt_id'] for i in evaluation.undecided()] == ['x']
    records.update(table([ARBITER], suff=base, wit=good, mapping=MAP))
    assert Evaluation([item()], records).results['x']['B'] == 1            # two of three
    records.update(table([ARBITER], suff=base, wit=bad, mapping=MAP))
    assert Evaluation([item()], records).results['x']['B'] is None           # arbiter agrees with the dissenter: still undecided


def test_missing_witness_and_missing_slots_never_confirm():
    records = table(GROUPS, suff={'T': NOT_SUFF, 'I': NOT_SUFF}, wit={'T': witness('alpha', 'beta'), 'I': witness('alpha', 'beta')}, mapping=MAP)
    assert Evaluation([item(witnesses=False)], records).results['x']['B'] is None            # nothing to recompute in either direction
    no_slots = {k: v for k, v in records.items() if '|SLOT|' not in k}
    assert Evaluation([item()], no_slots).results['x']['B'] is None                            # equivalence undecidable


def test_invalid_or_unanswered_slots_are_not_evaluated():
    items = [item('a'), {**item('b'), 'validity': 0}, {**item('c'), 'validity': None}, {**item('d'), 'reference': None}]
    assert [i['attempt_id'] for i in Evaluation(items, {}).items] == ['a']


# ----------------------------------------------------------------------------- end to end with an offline mock

SENTINELS = {'text': 'TEXT_SENTINEL', 'ref': 'REF_SENTINEL'}


def build_source(tmp_path, answer=None):
    src = tmp_path / 'outputs/source'
    src.mkdir(parents=True)
    (tmp_path / 'img.png').write_bytes(b'\x89PNG\r\n\x1a\nfixture')
    digest = sha256_file(tmp_path / 'img.png')
    (src / 'manifest.json').write_text(json.dumps({'settings': {'assessment': {'knowledge': 'K_SENTINEL'}}}))
    pair = {'kind': 'ambiguous', 'scenario_a': 'SCENARIO_A', 'scenario_b': 'SCENARIO_B', 'answer_a': 'x', 'answer_b': 'y', 'reason': 'r'}
    rows = [('Q1', 1, True), ('Q2', 1, False), ('Q3', 0, False), ('Q4', None, False)]
    records, sources = [], []
    for n, (q, validity, has_witness) in enumerate(rows):
        a = f'attempt_{n}'
        records.append({'attempt_id': a, 'material_id': f'm{n}', 'question': {'normalized_text': q}, 'question_answerability': validity,
                        'both_required': None, 'outputs': {'analysis_T': {'output': pair}, 'analysis_I': {'output': pair}} if has_witness else {}})
        sources.append({'attempt_id': a, 'item_id': f'S{n}', 'material': {'text': f'{SENTINELS["text"]} {q}', 'image_path': 'img.png', 'image_sha256': digest},
                        'responses': {'TI': {'answer': answer or f'{SENTINELS["ref"]} gamma'}}})
    write_jsonl(src / 'records.jsonl', records)
    write_jsonl(src / 'source_questions.jsonl', sources)
    (tmp_path / 'prompts').mkdir()
    for name in TEMPLATES:
        shutil.copy(ROOT / f'prompts/{name}.txt', tmp_path / 'prompts')


class Mock:
    """Offline stand-in for the model groups. Q1 is a dual candidate the groups disagree on; Q2 is text-sufficient."""

    def __init__(self, fail=None, ref_disagree=False):
        self.calls, self.fail, self.ref_disagree = [], fail, ref_disagree

    def __call__(self, model):
        outer = self

        class Provider:
            def complete(self, request):
                outer.calls.append((model, request))
                prompt = request.prompt
                if outer.fail and outer.fail(model, request):
                    raise RuntimeError('mock failure')
                if '最小答案槽位' in prompt:
                    data = {'slots': [{'name': 'v', 'asks': 'x', 'granularity': 'g'}]}
                elif '答案槽位定义' in prompt:
                    lines = [line.split('] ', 1)[1] for line in prompt.splitlines() if line.startswith('[')]
                    data = {'items': [{'i': n, 'values': {'v': 'gamma' if 'gamma' in s else s.strip().lower()}} for n, s in enumerate(lines)]}
                elif '情形1' in prompt:
                    bad = model == 'model-b' and request.condition == 'T'
                    data = {'presuppositions': ['p'], 's': [{'presup_ok': not bad, 'consistent': True, 'answer': 'alpha'},
                                                           {'presup_ok': not bad, 'consistent': True, 'answer': 'beta'}]}
                elif request.condition is None and request.image_path and '情形1' not in prompt:        # reference answer with the full material
                    data = {'determinable': True, 'answer': 'delta' if (outer.ref_disagree and model == 'model-b') else 'gamma', 'reason': 'fixture'}
                elif prompt.split('固定问题：')[1].startswith('\nQ2') and request.condition == 'T':
                    data = {'determinable': True, 'answer': 'gamma', 'reason': 'fixture'}
                else:
                    data = {'determinable': False, 'answer': None, 'reason': 'fixture'}
                return Completion(json.dumps(data, ensure_ascii=False), model, {'total_tokens': 1})
        return Provider()


def go(tmp_path, mock, **kw):
    return run_ensemble(run_id='ens', project_root=tmp_path, provider_factory=mock, **kw)


def test_end_to_end_isolation_arbiter_scope_resume_and_denominator(tmp_path):
    build_source(tmp_path)
    assert init_ensemble(run_id='ens', source_run='source', model_a='model-a', model_b='model-b', arbiter='model-arb',
                         project_root=tmp_path, concurrency=4, candidates_only=False, reference='baseline')['n_slots'] == 4
    assert init_ensemble(run_id='ens', source_run='source', model_a='model-a', model_b='model-b', arbiter='model-arb',
                         project_root=tmp_path, concurrency=4, candidates_only=False, reference='baseline')['resumed']
    with pytest.raises(ValueError, match='frozen'):
        init_ensemble(run_id='ens', source_run='source', model_a='other', model_b='model-b', arbiter='model-arb', project_root=tmp_path, concurrency=4, candidates_only=False, reference='baseline')
    mock = Mock()
    go(tmp_path, mock)
    for model, r in mock.calls:                                           # material isolation and no reference leakage
        judging = '最小答案槽位' not in r.prompt and '答案槽位定义' not in r.prompt
        if judging:
            assert SENTINELS['ref'] not in r.prompt and 'K_SENTINEL' in r.prompt
        if judging and '情形1' not in r.prompt:
            if r.condition == 'T':
                assert r.image_path is None and SENTINELS['text'] in r.prompt
            else:
                assert r.image_path and NO_IMAGE_TEXT in r.prompt and SENTINELS['text'] not in r.prompt
        if '情形1' in r.prompt and r.condition == 'I':                      # image removed: text stays, image is not sent
            assert r.image_path is None and SENTINELS['text'] in r.prompt
        if '情形1' in r.prompt and r.condition == 'T':                      # text removed: image stays, original text is not sent
            assert r.image_path and SENTINELS['text'] not in r.prompt
        if '最小答案槽位' in r.prompt:
            assert SENTINELS['text'] not in r.prompt and r.image_path is None and SENTINELS['ref'] not in r.prompt
    assert {r.metadata['key'].split('|')[-1] for model, r in mock.calls if model == 'model-arb'} == {'attempt_0'}   # only the open item
    assert not any(model not in ('model-a', 'model-b', 'model-arb') for model, _ in mock.calls)
    calls = len(mock.calls)
    go(tmp_path, mock)
    assert len(mock.calls) == calls                                         # resume adds nothing
    result = report_ensemble(run_id='ens', project_root=tmp_path)
    assert result['n_slots'] == 4 and result['confirmed_dual'] + result['not_confirmed'] == 4
    final = json.loads((tmp_path / 'outputs/ens/final_items.json').read_text())
    assert final['S0']['B'] == 1 and final['S0']['arbitrated'] and final['S1']['B'] == 0 and not final['S1']['arbitrated']
    assert 'S2' not in final and 'S3' not in final                          # invalid / validity-pending slots keep their place in the denominator only
    assert result['detail']['invalid_question'] == 1 and result['detail']['validity_pending'] == 1


def test_failed_calls_are_recorded_not_cached_and_retried_on_resume(tmp_path):
    build_source(tmp_path)
    init_ensemble(run_id='ens', source_run='source', model_a='model-a', model_b='model-b', arbiter='model-arb', project_root=tmp_path, concurrency=2, candidates_only=False, reference='baseline')
    target = lambda model, r: model == 'model-b' and r.condition == 'T' and '情形1' not in r.prompt and '固定问题：\nQ2' in r.prompt
    go(tmp_path, Mock(fail=target), phase='screen')
    rows = read_jsonl(tmp_path / 'outputs/ens/results.jsonl')
    failed = [r for r in rows if r.get('error')]
    assert failed and all(r['attempts'] == 2 for r in failed)               # one retry, then the failure stays visible
    assert all(r['key'] not in {x['key'] for x in rows if not x.get('error')} for r in failed)
    retry = Mock()
    go(tmp_path, retry, phase='screen')
    assert {r.metadata['key'] for _, r in retry.calls if r.metadata['key'] in {x['key'] for x in failed}} == {x['key'] for x in failed}
    assert all(r.metadata['key'] in {x['key'] for x in failed} or r.metadata['kind'] == 'SLOT' for _, r in retry.calls)


def test_http_cap_stops_the_run(tmp_path):
    build_source(tmp_path)
    init_ensemble(run_id='ens', source_run='source', model_a='model-a', model_b='model-b', arbiter='model-arb', project_root=tmp_path,
                  concurrency=1, max_http=3, candidates_only=False, reference='baseline')
    with pytest.raises(RuntimeError, match='HTTP cap'):
        go(tmp_path, Mock())
    assert len(read_jsonl(tmp_path / 'outputs/ens/results.jsonl')) <= 3


def test_cli_exposes_the_three_commands_with_the_common_run_arguments():
    from egvqg.cli import build_parser
    parser = build_parser()
    init = parser.parse_args(['necessity-ensemble-init', '--run-id', 'r', '--source-run', 's'])
    assert (init.run_id, init.source_run, init.max_http) == ('r', 's', 16000) and init.output_root == 'outputs' and not init.baseline_reference
    assert parser.parse_args(['necessity-ensemble-run', '--run-id', 'r', '--phase', 'arbitrate', '--limit', '3']).phase == 'arbitrate'
    assert parser.parse_args(['necessity-ensemble-report', '--run-id', 'r']).command == 'necessity-ensemble-report'


def test_partial_run_reports_unevaluated_slots_separately(tmp_path):
    build_source(tmp_path)
    init_ensemble(run_id='ens', source_run='source', model_a='model-a', model_b='model-b', arbiter='model-arb', project_root=tmp_path, concurrency=2, candidates_only=False, reference='baseline')
    go(tmp_path, Mock(), phase='screen', limit=1)
    detail = report_ensemble(run_id='ens', project_root=tmp_path)['detail']
    assert detail['not_yet_evaluated'] == 1 and detail['necessity_undecided'] == 1 and detail['necessity_undecided_why']


def test_default_mode_runs_only_items_that_can_be_confirmed(tmp_path):
    build_source(tmp_path)
    init_ensemble(run_id='ens', source_run='source', model_a='model-a', model_b='model-b', arbiter='model-arb', project_root=tmp_path, concurrency=2, reference='baseline')
    mock = Mock()
    go(tmp_path, mock)
    assert {r.metadata['key'].split('|')[-1] for _, r in mock.calls} == {'attempt_0'}       # Q2 has no two-way witness: no call at all
    result = report_ensemble(run_id='ens', project_root=tmp_path)
    assert result['detail']['no_two_way_witness_not_run'] == 1 and result['confirmed_dual'] == 1
    assert result['confirmed_dual'] + result['not_confirmed'] == result['n_slots'] == 4


def test_independent_reference_replaces_the_evaluated_models_own_answer(tmp_path):
    build_source(tmp_path, answer='WRONG')       # the evaluated model answered every question wrongly
    init_ensemble(run_id='ens', source_run='source', model_a='model-a', model_b='model-b', arbiter='model-arb', project_root=tmp_path,
                  concurrency=2, candidates_only=False)           # default reference: independent
    mock = Mock()
    go(tmp_path, mock)
    refs = [r for _, r in mock.calls if r.metadata['kind'] == 'REF']
    assert refs and all(r.image_path and r.condition is None and SENTINELS['text'] in r.prompt and 'WRONG' not in r.prompt for r in refs)
    assert {m for m, r in mock.calls if r.metadata['kind'] == 'REF'} == {'model-a', 'model-b'}
    report_ensemble(run_id='ens', project_root=tmp_path)
    final = json.loads((tmp_path / 'outputs/ens/final_items.json').read_text())
    assert final['S1']['B'] == 0 and final['S0']['B'] == 1          # the text-sufficient item is still recognised: the reference is gamma, not WRONG


def test_wrong_baseline_answer_as_reference_cannot_refute_a_sufficient_item(tmp_path):
    build_source(tmp_path, answer='WRONG')
    init_ensemble(run_id='ens', source_run='source', model_a='model-a', model_b='model-b', arbiter='model-arb', project_root=tmp_path,
                  concurrency=2, candidates_only=False, reference='baseline')
    go(tmp_path, Mock())
    report_ensemble(run_id='ens', project_root=tmp_path)
    final = json.loads((tmp_path / 'outputs/ens/final_items.json').read_text())
    assert final['S1']['B'] is None                                  # 'gamma' (sufficient) differs from 'WRONG': not shown to be sufficient


def test_disagreeing_independent_references_leave_the_item_undecided(tmp_path):
    build_source(tmp_path)
    init_ensemble(run_id='ens', source_run='source', model_a='model-a', model_b='model-b', arbiter='model-arb', project_root=tmp_path,
                  concurrency=2, candidates_only=False)
    mock = Mock(ref_disagree=True)
    go(tmp_path, mock)
    result = report_ensemble(run_id='ens', project_root=tmp_path)
    assert result['confirmed_dual'] == 0 and result['detail']['necessity_undecided'] == 2
    assert any('reference undecided' in k for k in result['detail']['necessity_undecided_why'])
    assert not [1 for m, _ in mock.calls if m == 'model-arb']        # nothing to arbitrate without a reference
