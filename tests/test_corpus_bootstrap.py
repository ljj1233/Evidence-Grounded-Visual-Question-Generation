"""Corpus bootstrap must recompute nonlinear scores from sampled counts."""
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from vvqg.evaluation.metrics.bleu import corpus_bleu4


def test_report_bleu_interval_estimate_is_corpus_not_sentence_mean():
    spec = importlib.util.spec_from_file_location('phase23_report', Path('scripts/vvqg/phase23_report.py'))
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    predictions = [{'image_id': 'a', 'question': 'a cat sits on the mat'}, {'image_id': 'b', 'question': 'dog'}]
    refs = {'a': ['a cat sits on the mat'], 'b': ['the dog runs in the large park']}
    result = report.traditional(predictions, refs)
    assert result['bleu4_bootstrap_ci']['estimate'] == pytest.approx(result['corpus_bleu4'])


def test_rollout_unique_rate_is_mean_unique_fraction_not_image_indicator():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from vvqg.evaluation.metrics.diversity import diversity_summary
    rows = [
        {'image_id': 'a', 'question': 'Q one'}, {'image_id': 'a', 'question': 'Q one'},
        {'image_id': 'a', 'question': 'Q two'}, {'image_id': 'a', 'question': 'Q three'},
        {'image_id': 'b', 'question': 'Same'}, {'image_id': 'b', 'question': 'Same'},
        {'image_id': 'b', 'question': 'Same'}, {'image_id': 'b', 'question': 'Same'},
    ]
    result = diversity_summary(rows)
    assert result['rollout_unique_rate'] == pytest.approx((3 / 4 + 1 / 4) / 2)
    assert result['p_at_least2_normalized_unique'] == pytest.approx(0.5)
