import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from egvqg.studies import copy_circuit


ROOT = Path(copy_circuit.__file__).resolve().parents[3]


def test_render_and_text_positions():
    root = ROOT
    assert copy_circuit._render(root, 'Only items with value at least 6 count.') == (
        'Based on the context and the image, ask one question.\nContext:\n'
        '<T>Only items with value at least 6 count.</T>\nOutput only the question.\n')
    inputs = {'input_ids': torch.tensor([[5, 6, 7, 8, 9]]),
              'mm_token_type_ids': torch.tensor([[0, 0, 1, 1, 1]])}
    assert copy_circuit._text_positions(inputs) == [0, 1]


def test_key_block_bias_applies_at_decode_step():
    block = copy_circuit._KeyBlock()
    assert block.bias(1, 8) is None
    block.spec = {'keys': [2, 5], 'query_start': 8, 'strength': -1e4}
    # decode step: one query row, which is the generated token
    bias = block.bias(1, 8)
    assert bias.shape == (1, 1, 1, 8)
    assert bias[0, 0, 0, 2].item() == -1e4 and bias[0, 0, 0, 5].item() == -1e4
    assert bias[0, 0, 0, 3].item() == 0
    # the prompt prefill is never blocked (query_start == query_len)
    assert block.bias(8, 8) is None
    prefill = block.bias(9, 9)
    assert prefill[0, 0, :8].abs().sum().item() == 0 and prefill[0, 0, 8, 2].item() == -1e4


def test_summarize_counts_faithful_threshold():
    rows = [
        {'condition': 'control', 'material_id': 'm1', 'question': 'Which items have a value of at least 6?',
         'has_number': True, 'mentions_threshold': True, 'generated_label': 'single_image'},
        {'condition': 'block', 'material_id': 'm1', 'question': 'Which items have a value of at least 10?',
         'has_number': True, 'mentions_threshold': True, 'generated_label': 'single_image'},
        {'condition': 'block', 'material_id': 'm2', 'question': 'Which is the largest value?',
         'has_number': False, 'mentions_threshold': False, 'generated_label': 'single_image'},
    ]
    materials = {f'm{i}': {'material_id': f'm{i}', 'threshold': 6} for i in (1, 2)}
    summary = copy_circuit._summarize_ablations('r', rows, {'conditions': ['control', 'block']}, materials)
    assert summary['conditions']['control']['faithful_threshold_rate'] == 1.0
    assert summary['conditions']['block']['faithful_threshold_rate'] == 0.0
    assert summary['conditions']['block']['first_numbers'] == {10: 1}


def test_mask_report_merges_runs(tmp_path):
    materials = tmp_path / 'materials.jsonl'
    materials.write_text(json.dumps({'material_id': 'm1', 'threshold': 6,
                                     'text': 'Only items with value at least 6 count.'}) + '\n')
    for run, condition, question in (('a', 'control', 'Which items have a value of at least 6?'),
                                     ('b', 'block_text_span', 'Which is the largest value?')):
        directory = tmp_path / 'outputs' / run
        directory.mkdir(parents=True)
        (directory / 'masked.jsonl').write_text(json.dumps({
            'condition': condition, 'material_id': 'm1', 'question': question,
            'has_number': 'least 6' in question, 'mentions_threshold': 'least' in question,
            'generated_label': 'single_image'}) + '\n')
        (directory / 'mask_summary.json').write_text(json.dumps({'manifest': {
            'conditions': [condition], 'samples': 1, 'temperature': 0.7, 'max_new_tokens': 8,
            'category': 'derived', 'strength': -1e4}}))
    summary = copy_circuit.mask_report(run_id='merged', sources=['a', 'b'], materials_path=str(materials),
                                       project_root=str(tmp_path))
    assert list(summary['conditions']) == ['control', 'block_text_span']
    assert summary['conditions']['control']['faithful_threshold_rate'] == 1.0
    assert (tmp_path / 'outputs' / 'merged' / 'mask_report.md').exists()
