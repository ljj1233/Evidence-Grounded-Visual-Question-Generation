"""White-box capture: which generated step holds which position, span location, reference comparison, CLI wiring (no GPU)."""
import pytest

from egvqg.generate.capture import capture_plan, compare_to_reference, locate_spans, token_span


@pytest.mark.parametrize('n,ended,k,early,question_end,n_question', [
    (10, True, 4, [(1, 0), (2, 0), (3, 0), (4, 0)], (9, 0), 9),     # 9 question tokens + end token; token i is visible at step i+1
    (3, True, 4, [(1, 0), (2, 0)], (2, 0), 2),                       # question shorter than K
    (1, True, 4, [], None, 0),                                       # end token straight away: no question state exists
    (6, False, 4, [(1, 0), (2, 0), (3, 0), (4, 0)], (5, 0), 6),      # cut by the length limit: last token never fed back
    (0, False, 4, [], None, 0),
])
def test_capture_plan_indexes_the_step_that_actually_holds_each_state(n, ended, k, early, question_end, n_question):
    plan = capture_plan(n, ended, k)
    assert plan['prompt_end'] == (0, -1) and plan['early'] == early and plan['question_end'] == question_end
    assert plan['n_question_tokens'] == n_question and plan['complete'] is ended
    assert all(step <= max(n - 1, 0) for step, _ in plan['early'])


def test_capture_plan_rejects_bad_k():
    with pytest.raises(ValueError):
        capture_plan(5, True, 0)


def toy(rendered):
    """Character-level 'tokenizer': one token per character; markers stand in for special tokens by single characters."""
    return list(rendered), [(i, i + 1) for i in range(len(rendered))]


def test_spans_separate_image_text_user_and_generation_prompt():
    # S/E = vision start/end, '#' = image pad (token id 99); the template mimics <|im_start|>user\n ... <|im_end|>
    text = 'Pick one.'
    rendered = '<|im_start|>user\nS###E<T>' + text + '</T>\nOnly the question.<|im_end|>\n<|im_start|>assistant\n'
    chars, offsets = toy(rendered)
    ids = [99 if c == '#' else 1 if c == 'S' else 2 if c == 'E' else 7 for c in chars]
    spans = locate_spans(ids, offsets, rendered, text, image_id=99, vision_start=1, vision_end=2)
    cut = lambda name: rendered[spans[name][0]:spans[name][1]]
    assert cut('image') == '###' and cut('image_block') == 'S###E'
    assert cut('text') == text and cut('text_block') == f'<T>{text}</T>'
    assert cut('user').startswith('S###E<T>') and cut('user').endswith('Only the question.')
    assert cut('generation_prompt') == '<|im_start|>assistant\n' and spans['n_prompt_tokens'] == len(ids)
    assert spans['image'][1] <= spans['text'][0]                     # image first, then text


def test_spans_reject_ambiguous_or_missing_text_and_broken_image_block():
    text = 'abc'
    rendered = '<|im_start|>user\nS#E<T>abc</T><T>abc</T><|im_end|>\n<|im_start|>assistant\n'
    chars, offsets = toy(rendered)
    ids = [99 if c == '#' else 1 if c == 'S' else 2 if c == 'E' else 7 for c in chars]
    with pytest.raises(ValueError):
        locate_spans(ids, offsets, rendered, text, image_id=99, vision_start=1, vision_end=2)       # text twice
    with pytest.raises(ValueError):
        locate_spans(ids, offsets, rendered, 'zzz', image_id=99, vision_start=1, vision_end=2)      # text absent
    ids_bad = [7 if i == chars.index('E') else t for i, t in enumerate(ids)]
    with pytest.raises(ValueError):
        locate_spans(ids_bad, offsets, rendered, text, image_id=99, vision_start=1, vision_end=2)   # no end marker


def test_token_span_uses_overlap():
    assert token_span([(0, 3), (3, 5), (5, 9)], 4, 6) == (1, 3)
    with pytest.raises(ValueError):
        token_span([(0, 3)], 5, 6)


@pytest.mark.parametrize('a,b,match', [('Q?', 'Q?', 'exact'), ('Q? ', 'Q?', 'whitespace_only'), ('Which?', 'What?', 'different')])
def test_reference_comparison(a, b, match):
    result = compare_to_reference(a, b)
    assert result['match'] == match and (result['first_diff_char'] == 2) == (match == 'different')


def test_cli_exposes_whitebox_capture():
    from egvqg.cli import build_parser
    args = build_parser().parse_args(['whitebox-capture', '--run-id', 'r', '--local-model-path', '/m', '--limit', '8'])
    assert (args.limit, args.k, args.device, args.no_check, args.max_new_tokens) == (8, 4, 'cuda:0', False, 1024)


def test_compare_captures_reports_prompt_and_state_differences(tmp_path):
    torch = pytest.importorskip('torch')
    import json
    from safetensors.torch import save_file
    from egvqg.generate.capture import compare_captures

    def run(name, prompt_tokens, scale):
        d = tmp_path / name
        (d / 'states').mkdir(parents=True)
        states = {'prompt_end': torch.ones(3, 4) * scale, 'question_end': torch.ones(3, 4) * scale}
        save_file(states, str(d / 'states' / 'm1.safetensors'))
        row = {'material_id': 'm1', 'n_prompt_tokens': prompt_tokens, 'image_grid_thw': [[1, 2, 2]], 'generated_token_ids': [1, 2, 3], 'question': 'q',
               'states_file': 'states/m1.safetensors'}
        (d / 'captures.jsonl').write_text(json.dumps(row) + '\n')
    run('a', 100, 1.0)
    run('b', 100, 1.1)
    run('c', 120, 1.0)
    same = compare_captures(run_a='a', run_b='b', project_root=str(tmp_path), output_root='.')
    assert same['same_prompt_tokens'] == 1 and same['same_question_tokens'] == 1
    cmp_b = json.loads((tmp_path / 'b' / 'compare_to_a.json').read_text())
    assert cmp_b['rows'][0]['prompt_end_mean_rel'] == pytest.approx(0.1, abs=1e-3)
    changed = compare_captures(run_a='a', run_b='c', project_root=str(tmp_path), output_root='.')
    assert changed['same_prompt_tokens'] == 0
    assert 'prompt_end_mean_rel' not in json.loads((tmp_path / 'c' / 'compare_to_a.json').read_text())['rows'][0]      # not comparable position by position


def test_forward_extras_pads_the_token_type_ids_for_appended_text_tokens():
    torch = pytest.importorskip('torch')
    from egvqg.generate.capture import forward_extras
    inputs = {'input_ids': torch.zeros(1, 5, dtype=torch.long), 'attention_mask': torch.ones(1, 5, dtype=torch.long),
              'mm_token_type_ids': torch.tensor([[0, 1, 1, 0, 0]]), 'pixel_values': torch.zeros(3, 2)}
    extra = forward_extras(inputs, 3)
    assert extra['mm_token_type_ids'].tolist() == [[0, 1, 1, 0, 0, 0, 0, 0]] and 'input_ids' not in extra and 'pixel_values' in extra
    assert 'mm_token_type_ids' not in forward_extras({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}, 3)
