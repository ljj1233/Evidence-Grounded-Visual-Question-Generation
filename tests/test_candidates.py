"""P1 candidate helpers: nested stratified selection, seeds, duplicate collapse, replay provider (no GPU)."""
from collections import Counter

from egvqg.candidates import ReplayProvider, candidate_seed, cjk_share, select_materials, unique_candidates
from egvqg.providers import CompletionRequest


def skills(n_skills=10, per=9):
    return {f's{a}m{b}': f'skill{a}' for a in range(n_skills) for b in range(per)} | {'tiny1': 'rare', 'tiny2': 'rare'}


def test_selection_is_stratified_nested_and_skips_rare_skills():
    small = select_materials(skills(), seed=1, n_skills=4, per_skill=3)
    large = select_materials(skills(), seed=1, n_skills=8, per_skill=6)
    assert len(small) == 12 and set(Counter(s for s, _ in small).values()) == {3}
    assert set(small) <= set(large) and all(s != 'rare' for s, _ in large)
    assert select_materials(skills(), seed=1, n_skills=4, per_skill=3) == small
    assert select_materials(skills(), seed=2, n_skills=4, per_skill=3) != small


def test_seed_is_deterministic_and_distinct_per_candidate():
    seeds = {candidate_seed(7, m, j) for m in ('a', 'b') for j in range(8)}
    assert len(seeds) == 16 and candidate_seed(7, 'a', 3) == candidate_seed(7, 'a', 3) and all(0 <= s < 2 ** 31 for s in seeds)


def test_identical_questions_of_one_material_collapse_but_keep_multiplicity():
    rows = [{'candidate_id': f'a~c{j}', 'material_id': 'a', 'question': q} for j, q in enumerate(['Q1', 'Q1 ', 'Q2'])]
    rows.append({'candidate_id': 'b~c0', 'material_id': 'b', 'question': 'Q1'})
    uniques = unique_candidates(rows)
    assert [len(u['candidate_ids']) for u in uniques] == [2, 1, 1] and len({u['pseudo_id'] for u in uniques}) == 3


def test_language_share():
    assert cjk_share('这两个磁铁？') == 1.0 and cjk_share('Which one?') == 0.0 and cjk_share('') == 0.0


def test_replay_returns_stored_question_for_generation_and_delegates_answering():
    class Local:
        model, identity = 'm', {'x': 1}
        def complete(self, request):
            return f'local:{request.task}'
    p = ReplayProvider('stored?', Local())
    gen = p.complete(CompletionRequest(task='generate_question', prompt='x'))
    assert gen.text == 'stored?' and gen.usage['network_requests'] == 0 and p.complete(CompletionRequest(task='assess_quality', prompt='y')) == 'local:assess_quality'


def test_cli_exposes_candidate_commands():
    from egvqg.cli import build_parser
    a = build_parser().parse_args(['candidates-capture', '--run-id', 'r', '--local-model-path', '/m', '--n-skills', '20', '--per-skill', '3'])
    assert (a.n_candidates, a.temperature, a.top_p, a.top_k, a.exclude_run) == (8, 0.7, 0.8, 20, None)
    assert build_parser().parse_args(['candidate-source-init', '--run-id', 's', '--capture-run', 'r']).capture_run == 'r'


def test_a_materials_file_is_taken_as_is_and_can_be_subsampled_reproducibly():
    from egvqg.candidates import choose_materials
    mats = {f'm{i}': {'material_id': f'm{i}', 'group': 'g1' if i % 2 else 'g2'} for i in range(10)}
    kw = dict(materials_file='x.jsonl', group_field='group', n_skills=0, per_skill=0, run_seed=1, root=None)
    every = choose_materials(mats, select='all', max_materials=None, **kw)
    assert [m for _, m in every] == sorted(mats) and {g for g, _ in every} == {'g1', 'g2'}
    a = choose_materials(mats, select='sample', max_materials=4, **kw)
    assert len(a) == 4 and a == choose_materials(mats, select='sample', max_materials=4, **kw)
    assert choose_materials({'m': {'material_id': 'm'}}, select='all', max_materials=None, **kw) == [('all', 'm')]


def test_plain_v2_prompt_is_neutral_and_keeps_the_span_markers(tmp_path):
    from pathlib import Path
    from egvqg.prompts import PromptSet
    text = (Path(__file__).resolve().parents[1] / 'prompts' / 'plain-v2.txt').read_text(encoding='utf-8')
    assert '<T>$T</T>' in text and text.count('$T') == 1
    lowered = text.lower()
    for framing in ('educational', 'clear', 'natural', 'answerable', 'based on the material', 'provided as', 'both', 'must'):
        assert framing not in lowered
    assert 'context' in lowered and 'image' in lowered and 'only the question' in lowered
    rendered = PromptSet(Path(__file__).resolve().parents[1] / 'prompts', names=('plain-v2',)).render('plain-v2', T='some context')
    assert '<T>some context</T>' in rendered
